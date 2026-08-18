"""The nightly scheduler: step order, the failure gate, and lock coordination.

Fast and fully synthetic — every step's child process is faked, so this suite
exercises the *orchestration* (what runs, in what order, under which lock, and
what a failure leaves behind) without touching the data warehouse or spending an
hour on a real batch.

The integration that actually matters is the last one here: the daemon and the
dashboard must agree on the lock files and the failure markers, or a manual
"Sync from Data Warehouse" click at 00:05 and the nightly run will each start
their own ten-minute query, and a failed night will leave the app quietly
serving stale data — the exact outcome ``joblocks.record_failure`` exists to
prevent.
"""

import os
import time

import pytest

import joblocks
import scheduler


# --------------------------------------------------------------------------- #
# Harness                                                                     #
# --------------------------------------------------------------------------- #
class FakePopen:
    """Stand-in for a step's child process: records the call, returns a code."""

    def __init__(self, argv, *, cwd=None, env=None, stdout=None, **_kw):
        self.argv = argv
        self.cwd = cwd
        self.env = env
        self.pid = 424242
        self._code = FakePopen.codes.get(FakePopen.step_of(argv), 0)
        FakePopen.calls.append(self)
        # Real children write into the log handle they inherit; some tests read
        # it back through joblocks.last_error_line, so write something plausible.
        if stdout is not None and self._code:
            stdout.write("2026-08-18 00:00:03,001 ERROR   Database error: "
                         "('IM002', 'Data source name not found')\n")
            stdout.flush()

    def wait(self):
        return self._code

    # -- test controls ------------------------------------------------------
    calls = []
    codes = {}

    @staticmethod
    def step_of(argv):
        """'demand' / 'warehouse' / 'key_skus' / 'batch' from the argv."""
        joined = " ".join(argv)
        for key, needle in (("demand", "extract_demand_details"),
                            ("warehouse", "extract_warehouse_projections"),
                            ("key_skus", "extract_key_skus"),
                            ("batch", "agent.batch")):
            if needle in joined:
                return key
        return "?"

    @classmethod
    def ran(cls):
        return [cls.step_of(c.argv) for c in cls.calls]


@pytest.fixture
def locks(tmp_path, monkeypatch):
    """Point every step's lock at a tmp dir and fake out the child processes.

    Returns ``{step_key: lock_path}``. ``log_config.LOG_ROOT`` is already
    redirected by the autouse fixture in conftest, so the per-pull logs land in
    a tmp tree too.
    """
    paths = {}
    for key, fn in (("demand", "_demand_lock"),
                    ("warehouse", "_warehouse_lock"),
                    ("key_skus", "_key_skus_lock"),
                    ("batch", "_batch_lock")):
        folder = tmp_path / key
        folder.mkdir()
        paths[key] = str(folder / joblocks.LOCK_NAME)
        monkeypatch.setattr(scheduler, fn, lambda p=paths[key]: p)

    FakePopen.calls = []
    FakePopen.codes = {}
    monkeypatch.setattr(scheduler.subprocess, "Popen", FakePopen)
    # build_steps pins DEMAND_RAW_DIR/WAREHOUSE_RAW_DIR from the real data_io
    # resolution; harmless here (no child really runs) but keep it off the share.
    monkeypatch.setenv("DEMAND_RAW_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("WAREHOUSE_RAW_DIR", str(tmp_path / "wh"))
    return paths


# --------------------------------------------------------------------------- #
# The schedule                                                                #
# --------------------------------------------------------------------------- #
def test_default_schedule_is_midnight():
    from apscheduler.triggers.cron import CronTrigger

    trigger = CronTrigger(hour=scheduler.DEFAULT_HOUR,
                          minute=scheduler.DEFAULT_MINUTE)
    nxt = trigger.get_next_fire_time(None, _now())

    assert (nxt.hour, nxt.minute, nxt.second) == (0, 0, 0)


def test_next_fire_is_within_a_day():
    """A daily trigger, not a one-shot: the next fire is always < 24h out."""
    from apscheduler.triggers.cron import CronTrigger

    now = _now()
    nxt = CronTrigger(hour=0, minute=0).get_next_fire_time(None, now)

    assert 0 < (nxt - now).total_seconds() <= 24 * 3600


def _now():
    import datetime

    from tzlocal import get_localzone

    return datetime.datetime.now(get_localzone())


# --------------------------------------------------------------------------- #
# What each step runs                                                         #
# --------------------------------------------------------------------------- #
def test_nightly_demand_pull_is_the_full_one():
    """--incremental is the BUTTON's mode. The nightly pull is the self-healing
    36-month baseline that picks up restated actuals and item renames; making it
    incremental would quietly remove the only thing that repairs the snapshot."""
    demand = {s.key: s for s in scheduler.build_steps()}["demand"]

    assert "--incremental" not in demand.argv
    assert any("extract_demand_details" in a for a in demand.argv)


def test_batch_runs_all_models_for_every_view_by_default():
    """No --views: Optimized Projections needs a summary for every group."""
    batch = {s.key: s for s in scheduler.build_steps()}["batch"]

    assert batch.argv[-2:] == ["-m", "agent.batch"]
    assert "--views" not in batch.argv
    assert "--no-llm" not in batch.argv


def test_batch_flags_are_passed_through():
    batch = {s.key: s
             for s in scheduler.build_steps(no_llm=True, workers=6)}["batch"]

    assert "--no-llm" in batch.argv
    assert batch.argv[batch.argv.index("--workers") + 1] == "6"


def test_provider_is_dropped_when_the_llm_is_off():
    """--provider is meaningless with --no-llm and agent.batch ignores it; not
    passing it keeps the command line honest about what the run will do."""
    batch = {s.key: s for s in
             scheduler.build_steps(no_llm=True, provider="anthropic")}["batch"]

    assert "--provider" not in batch.argv


def test_batch_runs_from_src_so_the_module_resolves():
    """`python -m agent.batch` only resolves with src/ as the working dir."""
    batch = {s.key: s for s in scheduler.build_steps()}["batch"]

    assert os.path.basename(batch.cwd) == "src"


# --------------------------------------------------------------------------- #
# Step order and the failure gate                                             #
# --------------------------------------------------------------------------- #
def test_all_four_steps_run_in_order_on_a_good_night(locks):
    assert scheduler.nightly_refresh() == 0
    assert FakePopen.ran() == ["demand", "warehouse", "key_skus", "batch"]


def test_a_failed_demand_pull_skips_the_all_models_run(locks):
    """An hour of backtesting against yesterday's snapshot would publish
    recommendations dated today over unchanged data."""
    FakePopen.codes = {"demand": 1}

    code = scheduler.nightly_refresh()

    assert "batch" not in FakePopen.ran()
    assert code == 1


def test_the_independent_pulls_run_even_when_the_demand_pull_fails(locks):
    """Warehouse projections and the key-SKU list are separate data sets."""
    FakePopen.codes = {"demand": 1}

    scheduler.nightly_refresh()

    assert FakePopen.ran() == ["demand", "warehouse", "key_skus"]


def test_worst_exit_code_wins(locks):
    FakePopen.codes = {"warehouse": 2, "key_skus": 1}

    assert scheduler.nightly_refresh() == 2


def test_a_failing_batch_is_reported(locks):
    FakePopen.codes = {"batch": 1}

    assert scheduler.nightly_refresh() == 1


def test_dry_run_launches_nothing_and_writes_no_lock(locks):
    assert scheduler.nightly_refresh(dry_run=True) == 0
    assert FakePopen.calls == []
    assert not any(os.path.exists(p) for p in locks.values())


# --------------------------------------------------------------------------- #
# Locks                                                                       #
# --------------------------------------------------------------------------- #
def test_a_live_lock_makes_the_step_skip_rather_than_collide(locks):
    """Someone clicked Sync at 00:05, or a second daemon is running on the other
    host that mounts this share. Two concurrent 10-minute queries is the worst
    outcome available; using the data that run produces is the best."""
    joblocks.acquire(locks["demand"], joblocks.now_stamp())

    code = scheduler.nightly_refresh()

    assert "demand" not in FakePopen.ran()
    assert code == 0                       # skipped is not failed...
    assert "batch" in FakePopen.ran()      # ...so the batch still runs


def test_a_stale_lock_is_taken_over(locks):
    """A crashed run leaves its lock behind; the night must not be lost to it."""
    joblocks.acquire(locks["demand"], "2026-08-17 00:00:00")
    old = time.time() - joblocks.REFRESH_STALE_SECONDS - 60
    os.utime(locks["demand"], (old, old))

    scheduler.nightly_refresh()

    assert "demand" in FakePopen.ran()


def test_locks_are_released_when_the_night_ends(locks):
    FakePopen.codes = {"warehouse": 1}

    scheduler.nightly_refresh()

    # Including the failed one: the marker records the failure, the lock does
    # not, or the next night would skip the step it most needs to retry.
    assert not any(os.path.exists(p) for p in locks.values())


def test_the_batch_lock_records_the_childs_pid(locks, monkeypatch):
    """``refresh._pid_running_batch`` confirms the recorded PID's command line
    contains "agent.batch" before believing the run is alive. Recording the
    scheduler's own PID would read as "already finished" and the dashboard would
    clear the lock out from under a live run."""
    seen = {}

    class Capturing(FakePopen):
        def wait(self):
            if FakePopen.step_of(self.argv) == "batch":
                seen["lock"] = open(locks["batch"], encoding="utf-8").read()
            return super().wait()

    monkeypatch.setattr(scheduler.subprocess, "Popen", Capturing)
    scheduler.nightly_refresh()

    started, pid = _parse_lock_text(seen["lock"])
    assert pid == 424242                # the child's, not os.getpid()
    assert started == started.strip()


def test_single_pull_locks_stay_one_line(locks, monkeypatch):
    """``_refresh_state`` shows ``f.read().strip()`` to the user as the start
    time, so a second line would reach the UI as part of the timestamp."""
    seen = {}

    class Capturing(FakePopen):
        def wait(self):
            key = FakePopen.step_of(self.argv)
            seen[key] = open(locks[key], encoding="utf-8").read()
            return super().wait()

    monkeypatch.setattr(scheduler.subprocess, "Popen", Capturing)
    scheduler.nightly_refresh()

    for key in ("demand", "warehouse", "key_skus"):
        assert "\n" not in seen[key], key


def _parse_lock_text(text):
    lines = text.splitlines()
    return lines[0], int(lines[1])


# --------------------------------------------------------------------------- #
# The dashboard has to see what the night left behind                         #
# --------------------------------------------------------------------------- #
def test_a_failed_step_leaves_the_marker_the_dashboard_banner_reads(locks,
                                                                    monkeypatch):
    """End-to-end on the one seam that matters: a step fails at 00:00 with
    nobody watching, and by morning the dashboard says so on its own."""
    from dashboard_app import refresh as rf

    FakePopen.codes = {"demand": 1}
    scheduler.nightly_refresh()

    # The dashboard reads its OWN lock paths; point them at the same files the
    # daemon just used (in production they resolve to these already).
    monkeypatch.setattr(rf, "_refresh_lock_path", lambda: locks["demand"])
    monkeypatch.setattr(rf, "_wh_refresh_lock_path", lambda: locks["warehouse"])
    monkeypatch.setattr(rf, "_key_skus_lock_path", lambda: locks["key_skus"])

    failures = rf.sync_failures()

    assert [label for label, *_ in failures] == ["Demand snapshot"]
    _label, when, host, detail = failures[0]
    assert when and host
    # The child's real error, not a generic "it failed" — that specificity is
    # what turned a six-day outage into something diagnosable.
    assert "IM002" in detail


def test_a_good_run_leaves_no_failure_marker(locks):
    joblocks.record_failure(locks["demand"], "Nightly DW refresh (full)",
                            str(locks["demand"]) + ".log", "last night broke")
    assert joblocks.read_failure(locks["demand"]) is not None

    scheduler.nightly_refresh()

    assert joblocks.read_failure(locks["demand"]) is None


def test_the_run_header_scopes_the_error_scan_to_this_run(locks):
    """Every launcher writes the same header, because ``last_error_line`` only
    looks at lines after the newest one — a shared log with a private header
    format would attribute one run's error to another."""
    log = joblocks.refresh_log_path(joblocks.DEMAND_LOG)
    with open(log, "a", encoding="utf-8") as f:
        f.write(joblocks.run_header("An earlier run", "2026-08-17 00:00:00"))
        f.write("2026-08-17 00:00:01 ERROR   yesterday's problem\n")

    FakePopen.codes = {"demand": 1}
    scheduler.nightly_refresh()

    _when, _host, detail = joblocks.read_failure(locks["demand"])
    assert "IM002" in detail
    assert "yesterday" not in detail


# --------------------------------------------------------------------------- #
# The joblocks extraction must not have moved anything the dashboard uses     #
# --------------------------------------------------------------------------- #
def test_refresh_still_exposes_its_original_helpers():
    from dashboard_app import refresh as rf

    assert rf._failure_path is joblocks.failure_path
    assert rf._read_failure is joblocks.read_failure
    assert rf._read_lock is joblocks.read_lock
    assert rf._clear_lock is joblocks.release
    assert (rf.DEMAND_LOG, rf.WAREHOUSE_LOG, rf.KEY_SKUS_LOG) == (
        joblocks.DEMAND_LOG, joblocks.WAREHOUSE_LOG, joblocks.KEY_SKUS_LOG)


def test_scheduler_and_dashboard_resolve_the_same_locks():
    """If these ever drift, a click and the nightly run stop seeing each other
    and both start a ten-minute query against the warehouse."""
    from dashboard_app import refresh as rf

    for theirs, ours in ((rf._refresh_lock_path(), scheduler._demand_lock()),
                         (rf._wh_refresh_lock_path(), scheduler._warehouse_lock()),
                         (rf._key_skus_lock_path(), scheduler._key_skus_lock()),
                         (rf._batch_lock_path(), scheduler._batch_lock())):
        assert os.path.normcase(os.path.realpath(theirs)) == \
               os.path.normcase(os.path.realpath(ours))


def test_the_batch_stale_window_clears_a_real_run(locks):
    """A measured full batch is 48-58 minutes. The lock self-clearing UNDER a
    live run is worse than a stale lock: the dashboard reports the batch
    finished while it is still writing summaries."""
    assert joblocks.BATCH_STALE_SECONDS > 2 * 58 * 60


def test_the_scheduler_module_never_imports_streamlit():
    """It runs with no ScriptRunContext, and ``refresh``'s batch helpers read
    ``st.session_state``. That is the whole reason ``joblocks`` exists."""
    import ast

    for module in (scheduler, joblocks):
        tree = ast.parse(open(module.__file__, encoding="utf-8").read())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any(n.split(".")[0] in ("streamlit", "dashboard_app")
                           for n in names), (module.__name__, names)
