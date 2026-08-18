"""Nightly scheduler: full data-warehouse sync, then all models for every view.

Runs as a long-lived daemon. At 00:00 local time it performs, in order:

  1. ``extract_demand_details.py`` — the **full** 36-month pull (deliberately no
     ``--incremental``). This is the self-healing baseline that picks up restated
     actuals, item renames and customer remaps; the dashboard's button runs the
     fast incremental pull instead.
  2. ``extract_warehouse_projections.py`` — the five regional files behind the
     missing-projections table.
  3. ``extract_key_skus.py`` — the Exceptions view's "Key SKUs" watchlist.
  4. ``python -m agent.batch`` — backtests all five models across every view and
     publishes ``outputs/agent_summary_<view>.json``, which is what **Optimized
     Projections** reads to pick each customer group's model. It also warms
     ``forecast_cache``, so the first planner in each morning reads forecasts off
     disk instead of recomputing them.

Steps 2 and 3 run even if step 1 failed — they are independent data sets. Step 4
runs **only if step 1 succeeded**: a failed pull leaves no fresh data worth
spending an hour of backtesting on. The process exits with the worst step's code.

Start it with::

    python src/scheduler.py                  # daemon: sleep until 00:00, repeat
    python src/scheduler.py --once           # run the job now and exit
    python src/scheduler.py --once --dry-run # print the plan, touch nothing

``start_scheduler.ps1`` is the boot wrapper (it resolves the interpreter the same
way ``refresh_demand_data.ps1`` does); register that with Task Scheduler on a
``/sc onstart`` trigger so the daemon comes back after a reboot.

Why a daemon rather than four Task Scheduler entries: the ordering and the
"only if the demand pull succeeded" gate live in Python, where the repo's test
suite can reach them, and every step writes the same lock files, failure markers
and dated logs the dashboard already reads (see ``joblocks``). A nightly failure
therefore raises the dashboard's existing red banner by morning instead of going
unnoticed — which is the failure mode that once left the app serving six-day-old
data.

**This module must not import ``dashboard_app.refresh``** (or anything else that
imports ``streamlit``): it runs with no ScriptRunContext, and that module's
agent-batch helpers read ``st.session_state``. The half both need lives in
``joblocks``.
"""

import os
import sys
import atexit
import socket
import logging
import argparse
import subprocess

# Support both `python src/scheduler.py` and `python -m scheduler` from src/.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from dotenv import load_dotenv

import joblocks
from log_config import DateFolderHandler
from agent import data_io

REPO_ROOT = os.path.dirname(HERE)

# The extracts and agent.batch each read their own settings from .env (the SQL_*
# connection vars, ANTHROPIC_API_KEY for the batch's narrative nodes). They call
# load_dotenv() themselves and inherit our environment either way, but loading it
# here too means the daemon's own startup log can report which server it will hit
# instead of failing opaquely an hour later.
load_dotenv(os.path.join(REPO_ROOT, ".env"))

logger = logging.getLogger("demand_scheduler")

DEFAULT_HOUR = int(os.environ.get("DEMAND_SCHEDULE_HOUR", 0))
DEFAULT_MINUTE = int(os.environ.get("DEMAND_SCHEDULE_MINUTE", 0))

# A missed fire (the box was asleep, or the daemon was down for a deploy) still
# runs if we come back within the hour — stale data is the thing we are fixing,
# so a 00:40 start beats skipping the night. Past that, wait for tomorrow rather
# than kicking off a 90-minute job into the working day.
MISFIRE_GRACE_SECONDS = 60 * 60


# --------------------------------------------------------------------------- #
# The four steps                                                              #
# --------------------------------------------------------------------------- #
class Step:
    """One nightly step: what to run, which lock guards it, where it logs.

    ``lock_dir`` and the argv are resolved as *callables* rather than at import,
    because both depend on env vars (``DEMAND_RAW_DIR``, ``WAREHOUSE_RAW_DIR``)
    that a test — or an operator running one step by hand — may redirect between
    the daemon starting and a job firing.
    """

    def __init__(self, key, label, argv, lock, log_name, stale_seconds,
                 cwd=HERE, env=None, records_pid=False):
        self.key = key
        self.label = label
        self.argv = argv
        self.lock = lock
        self.log_name = log_name
        self.stale_seconds = stale_seconds
        self.cwd = cwd
        self.env = env or {}
        self.records_pid = records_pid


def _demand_lock():
    """The demand pull's lock, in the folder the extract writes snapshots to.

    Resolved through ``data_io`` rather than ``dashboard_app.datasources`` (which
    imports streamlit). Both honour ``DEMAND_RAW_DIR`` and both read the model
    files' identical ``RAW_INPUTS_FOLDER``, so the dashboard and this daemon land
    on the same file — which is the whole point of the lock.
    """
    return joblocks.lock_path(data_io._raw_dir(data_io.default_pipeline()))


def _warehouse_lock():
    return joblocks.lock_path(data_io._warehouse_dir())


def _key_skus_lock():
    # Derived from the discovery glob so the lock cannot drift away from the
    # folder the dashboard actually scans for key-SKU lists.
    return joblocks.lock_path(os.path.dirname(data_io.key_skus_glob()))


def _batch_lock():
    return joblocks.batch_lock_path(REPO_ROOT)


def build_steps(no_llm=False, workers=None, provider=None):
    """The nightly step list, in run order."""
    py = sys.executable

    batch_argv = [py, "-m", "agent.batch"]
    if no_llm:
        batch_argv.append("--no-llm")
    if workers:
        batch_argv += ["--workers", str(workers)]
    if provider and not no_llm:
        batch_argv += ["--provider", provider]

    return [
        Step(
            "demand", "Nightly DW refresh (full)",
            # No --incremental: the nightly pull is the self-healing baseline.
            [py, os.path.join(HERE, "extract_demand_details.py")],
            _demand_lock, joblocks.DEMAND_LOG, joblocks.REFRESH_STALE_SECONDS,
            # Pin the output folder to the one we resolved, so the child writes
            # exactly where the dashboard looks regardless of its own CWD.
            env={"DEMAND_RAW_DIR": data_io._raw_dir(data_io.default_pipeline())},
        ),
        Step(
            "warehouse", "Nightly warehouse refresh",
            [py, os.path.join(HERE, "extract_warehouse_projections.py")],
            _warehouse_lock, joblocks.WAREHOUSE_LOG,
            joblocks.REFRESH_STALE_SECONDS,
            env={"WAREHOUSE_RAW_DIR": data_io._warehouse_dir()},
        ),
        Step(
            "key_skus", "Nightly key-SKU refresh",
            [py, os.path.join(HERE, "extract_key_skus.py")],
            _key_skus_lock, joblocks.KEY_SKUS_LOG,
            joblocks.KEY_SKUS_STALE_SECONDS,
        ),
        Step(
            "batch", "Agent batch (all views)",
            batch_argv, _batch_lock, joblocks.BATCH_LOG,
            joblocks.BATCH_STALE_SECONDS,
            # `-m agent.batch` resolves because cwd=HERE is src/ (agent is a
            # package under src/) — the same invocation the dashboard uses.
            cwd=HERE,
            # The dashboard's liveness check confirms the recorded PID's command
            # line contains "agent.batch", so this step must record the CHILD's
            # PID; our own would read as "the batch already finished" and the
            # dashboard would clear the lock out from under a live run.
            records_pid=True,
        ),
    ]


# --------------------------------------------------------------------------- #
# Running one step                                                            #
# --------------------------------------------------------------------------- #
def run_step(step, dry_run=False):
    """Run one step to completion. Returns its exit code (0 if skipped).

    Blocking, unlike ``refresh._launch_refresh`` — that one deliberately detaches
    so a button click returns instantly, which is the opposite of what an ordered
    nightly job needs.

    A step whose lock is already live is **skipped, not failed**: someone clicked
    Sync at 00:05, or a second daemon is running on the other host that shares
    this network folder. Starting a concurrent 10-minute warehouse query would be
    strictly worse than using the data that run is about to produce.
    """
    lock = step.lock()
    live, started = joblocks.lock_is_live(lock, step.stale_seconds)
    if live:
        logger.warning("%s: skipped — a run started at %s still holds %s",
                       step.label, started or "an unknown time", lock)
        return 0

    if dry_run:
        logger.info("%s: would run %s (cwd=%s, lock=%s, log=%s)", step.label,
                    " ".join(step.argv), step.cwd, lock,
                    joblocks.refresh_log_path(step.log_name))
        return 0

    now = joblocks.now_stamp()
    joblocks.acquire(lock, now)
    # A new attempt supersedes the previous outcome, so drop the stale banner.
    joblocks.release(joblocks.failure_path(lock))

    log_path = joblocks.refresh_log_path(step.log_name)
    logger.info("%s: starting (log: %s)", step.label, log_path)

    code = 1
    try:
        with open(log_path, "a", encoding="utf-8") as logf:
            logf.write(joblocks.run_header(step.label, now))
            logf.flush()
            proc = subprocess.Popen(
                step.argv,
                cwd=step.cwd,
                env=joblocks.child_env(**step.env),
                stdout=logf,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=joblocks.bg_creationflags(),
                close_fds=True,
            )
            if step.records_pid:
                joblocks.acquire(lock, now, pid=proc.pid)
            code = proc.wait()
    except Exception as exc:  # noqa: BLE001 — one bad step must not kill the job
        logger.exception("%s: could not run", step.label)
        joblocks.record_failure(lock, step.label, log_path,
                                f"could not start: {exc}")
        joblocks.release(lock)
        return 1

    if code:
        detail = joblocks.record_failure(
            lock, step.label, log_path,
            f"exited with code {code} without writing new data",
        )
        logger.error("%s: FAILED (exit %s) — %s", step.label, code, detail)
    else:
        logger.info("%s: finished OK", step.label)

    joblocks.release(lock)
    return code


# --------------------------------------------------------------------------- #
# The nightly job                                                             #
# --------------------------------------------------------------------------- #
def nightly_refresh(no_llm=False, workers=None, provider=None, dry_run=False):
    """Run all four steps in order. Returns the worst exit code.

    Mirrors ``refresh_demand_data.ps1``: the two independent pulls run whatever
    the demand pull did, the all-models batch runs only after a good pull, and
    the worst code wins so a supervisor can tell any failure happened.
    """
    steps = {s.key: s for s in build_steps(no_llm, workers, provider)}
    codes = {}

    logger.info("Nightly refresh starting on %s%s",
                socket.gethostname(), " (dry run)" if dry_run else "")

    for key in ("demand", "warehouse", "key_skus"):
        codes[key] = run_step(steps[key], dry_run=dry_run)

    if codes["demand"] == 0:
        codes["batch"] = run_step(steps["batch"], dry_run=dry_run)
    else:
        # Backtesting 114 views against yesterday's snapshot would burn an hour
        # and publish recommendations dated today over unchanged data — worse
        # than leaving last night's summaries in place with a visible failure.
        logger.error("Skipping the all-models run: the demand pull failed, so "
                     "there is no fresh snapshot to backtest against.")
        codes["batch"] = 0

    worst = max(codes.values())
    logger.info("Nightly refresh finished: %s (worst exit code %s)",
                ", ".join(f"{k}={v}" for k, v in codes.items()), worst)
    return worst


def _job(**kwargs):
    """Scheduler entry point: never let a job raise into APScheduler's loop.

    An unhandled exception would be logged by APScheduler and the job would
    survive, but the daemon's own log is where anyone will look — so catch, log,
    and let the next night try again.
    """
    try:
        nightly_refresh(**kwargs)
    except Exception:  # noqa: BLE001 — last line of defence for a daemon
        logger.exception("Nightly refresh raised")


# --------------------------------------------------------------------------- #
# Single-instance guard                                                       #
# --------------------------------------------------------------------------- #
def _scheduler_lock_path():
    return os.path.join(REPO_ROOT, "outputs", ".scheduler.lock")


def _pid_alive(pid):
    """True/False/None ("can't tell", psutil missing) for a local PID."""
    if not pid:
        return None
    try:
        import psutil
    except ImportError:
        return None
    try:
        return psutil.Process(int(pid)).is_running()
    except Exception:  # noqa: BLE001 — NoSuchProcess and friends
        return False


def claim_scheduler_lock(force=False):
    """Record this daemon as the schedule owner. Returns True if we may run.

    The repo lives on a network share that more than one host mounts, so two
    people can each start a daemon without knowing. The per-step locks already
    stop duplicate *pulls*, but two schedules is still a misconfiguration worth
    naming: this refuses to start a second daemon on the SAME host (where the
    recorded PID is checkable) and warns loudly about one on another host (where
    it isn't).
    """
    path = _scheduler_lock_path()
    started, pid = joblocks.read_lock(path)
    host = joblocks.read_note(path)
    me = socket.gethostname()

    if started and not force:
        if host == me:
            alive = _pid_alive(pid)
            if alive:
                logger.error(
                    "A scheduler (pid %s) has been running on this host since "
                    "%s. Not starting a second one — pass --force to override.",
                    pid, started)
                return False
            logger.info("Taking over a stale scheduler lock from pid %s (%s).",
                        pid, started)
        else:
            logger.warning(
                "A scheduler was registered on %s at %s, and this folder is "
                "shared between hosts. Starting anyway (a remote PID cannot be "
                "checked), but you almost certainly want only one daemon: the "
                "step locks will make the loser skip its work every night.",
                host or "another host", started)

    joblocks.acquire(path, joblocks.now_stamp(), pid=os.getpid(), note=me)
    atexit.register(joblocks.release, path)
    return True


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _configure_logging(verbose=False):
    """Log to the console AND logs/<date>/logs_scheduler.txt.

    DateFolderHandler re-files each line under the day it was written, which
    matters more here than anywhere else in the app: this process is expected to
    run for months, and its most interesting lines are written at midnight.
    """
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if logger.handlers:
        return
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    for handler in (DateFolderHandler(joblocks.SCHEDULER_LOG),
                    logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    # joblocks logs failures under the dashboard's logger; mirror them here so
    # the scheduler log is a complete account of the night.
    joblocks.logger.addHandler(logger.handlers[0])


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Nightly full DW sync + all-models optimized projections."
    )
    ap.add_argument("--once", action="store_true",
                    help="Run the job immediately and exit, instead of "
                         "scheduling it. Exits with the worst step's code.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Log what each step would run, resolved paths and all, "
                         "without touching the warehouse or the data folders.")
    ap.add_argument("--hour", type=int, default=DEFAULT_HOUR,
                    help=f"Hour to run (default {DEFAULT_HOUR}, or "
                         "DEMAND_SCHEDULE_HOUR).")
    ap.add_argument("--minute", type=int, default=DEFAULT_MINUTE,
                    help=f"Minute to run (default {DEFAULT_MINUTE}, or "
                         "DEMAND_SCHEDULE_MINUTE).")
    ap.add_argument("--timezone", default=os.environ.get("SCHEDULER_TIMEZONE"),
                    help="IANA timezone for the schedule (default: this "
                         "machine's local time, or SCHEDULER_TIMEZONE).")
    ap.add_argument("--no-llm", action="store_true",
                    help="Run the batch without its LLM narrative nodes — "
                         "faster and free, but Optimized Projections loses its "
                         "written anomaly/summary prose.")
    ap.add_argument("--provider", choices=["anthropic", "local"], default=None,
                    help="LLM provider for the batch (overrides LLM_PROVIDER).")
    ap.add_argument("--workers", type=int, default=None,
                    help="Batch worker processes (default: CPU count - 1).")
    ap.add_argument("--force", action="store_true",
                    help="Start even if another daemon holds the scheduler "
                         "lock on this host.")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    _configure_logging(args.verbose)

    job_kwargs = dict(no_llm=args.no_llm, workers=args.workers,
                      provider=args.provider, dry_run=args.dry_run)

    if args.once:
        return nightly_refresh(**job_kwargs)

    # Imported here, not at module scope, so --once and the tests work on a
    # machine that hasn't installed APScheduler yet.
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    if not claim_scheduler_lock(force=args.force):
        return 1

    scheduler = BlockingScheduler(timezone=args.timezone)
    trigger = CronTrigger(hour=args.hour, minute=args.minute,
                          timezone=args.timezone)
    scheduler.add_job(
        _job, trigger, kwargs=job_kwargs,
        id="nightly_refresh", name="Nightly DW sync + all-models run",
        # The job can run for ~80 minutes. max_instances=1 stops tomorrow's run
        # from starting on top of one that overran; coalesce collapses a backlog
        # of missed fires (a laptop closed over a long weekend) into one run.
        max_instances=1, coalesce=True,
        misfire_grace_time=MISFIRE_GRACE_SECONDS,
    )

    logger.info("Scheduler started on %s — next run %s. Steps: full demand "
                "pull, warehouse, key SKUs, then all models for every view.",
                socket.gethostname(), trigger.get_next_fire_time(None, None))
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
