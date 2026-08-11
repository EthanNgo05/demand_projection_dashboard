"""Data-warehouse integration: snapshot pruning, atomic write, refresh lock.

Covers the pieces added when the ~10-minute SQL pull was moved out of the
request path (extract_demand_details.py writes the snapshot; the dashboard
serves it and can trigger a background refresh):

  1. ``prune_old_snapshots`` keeps only the newest N dated workbooks.
  2. ``write_powerbi_xlsx`` writes atomically (no temp litter) and the result is
     readable by the dashboard's own reader.
  3. The dashboard's refresh lock state-machine — idle / running / self-heals on
     completion or after going stale — and the double-launch guard.

None of these touch SQL Server, so they run in the normal (fast) suite.
"""

import glob
import os
import re
import sys
import time

import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

pytest.importorskip("pyodbc")  # extract_demand_details imports it at module load
import extract_demand_details as extract  # noqa: E402


def _make_snapshot(folder, date_str, mtime=None):
    """Create an empty dated snapshot workbook; optionally pin its mtime."""
    path = os.path.join(folder, f"all_demand_projections_{date_str}.xlsx")
    with open(path, "w", encoding="utf-8") as f:
        f.write("x")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


# --------------------------------------------------------------------------- #
# 1. Pruning                                                                  #
# --------------------------------------------------------------------------- #
def test_prune_keeps_newest_n_by_date(tmp_path):
    for d in ["2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04", "2026-07-05"]:
        _make_snapshot(str(tmp_path), d)

    removed = extract.prune_old_snapshots(str(tmp_path), keep=3)

    remaining = sorted(p.name for p in tmp_path.glob("all_demand_projections_*.xlsx"))
    assert remaining == [
        "all_demand_projections_2026-07-03.xlsx",
        "all_demand_projections_2026-07-04.xlsx",
        "all_demand_projections_2026-07-05.xlsx",
    ]
    assert len(removed) == 2  # the two oldest dates


def test_prune_counts_dates_not_files(tmp_path):
    # A warehouse snapshot is 5 region files sharing one date; ``keep`` counts
    # distinct dates so a snapshot lives or dies as a set.
    for d in ["2026-07-01", "2026-07-02", "2026-07-03"]:
        for region in ["AU", "CA", "EU", "JP", "US"]:
            path = tmp_path / f"{region}_warehouse_projections_{d}.xlsx"
            path.write_text("x", encoding="utf-8")

    removed = extract.prune_old_snapshots(
        str(tmp_path), keep=2, pattern="*_warehouse_projections_*.xlsx"
    )

    assert len(removed) == 5  # the whole 2026-07-01 set
    remaining = {p.name for p in tmp_path.glob("*.xlsx")}
    assert all("2026-07-01" not in n for n in remaining)
    assert len(remaining) == 10


def test_prune_disabled_when_keep_not_positive(tmp_path):
    for d in ["2026-07-01", "2026-07-02", "2026-07-03"]:
        _make_snapshot(str(tmp_path), d)

    assert extract.prune_old_snapshots(str(tmp_path), keep=0) == []
    assert len(list(tmp_path.glob("all_demand_projections_*.xlsx"))) == 3


def test_prune_never_deletes_undated_files(tmp_path):
    # A file without a YYYY-MM-DD in its name must never be auto-deleted.
    undated = tmp_path / "all_demand_projections_final.xlsx"
    undated.write_text("keep me")
    for d in ["2026-07-01", "2026-07-02"]:
        _make_snapshot(str(tmp_path), d)

    extract.prune_old_snapshots(str(tmp_path), keep=1)

    assert undated.exists()
    # Of the dated files only the newest survives.
    dated = sorted(
        p.name for p in tmp_path.glob("all_demand_projections_*.xlsx")
        if re.search(r"\d{4}-\d{2}-\d{2}", p.name)
    )
    assert dated == ["all_demand_projections_2026-07-02.xlsx"]


# --------------------------------------------------------------------------- #
# 2. Atomic write                                                             #
# --------------------------------------------------------------------------- #
def test_write_powerbi_xlsx_is_atomic_and_readable(tmp_path):
    from agent import data_io

    df = pd.DataFrame({
        "'Demand'[DisplaySKU]": ["ST1001", "ST1002"],
        "Description": ["Widget", "Gadget"],
        "Custnmbr": ["CUST1", "CUST2"],
        "WeekDate": ["2026-07-05", "2026-07-05"],
        "POS": [10, 20],
        "Projection": [12, 18],
        "Sum of Quantity": [5, 7],
    })
    out = tmp_path / "all_demand_projections_2026-07-05.xlsx"

    extract.write_powerbi_xlsx(df, str(out))

    # The only file left is the final workbook — no stray temp file from mkstemp.
    assert [p.name for p in tmp_path.iterdir()] == [out.name]

    clean = data_io.load_raw(str(out))
    assert list(clean["SKU"]) == ["ST1001", "ST1002"]
    assert list(clean["Orders"]) == [5, 7]  # 'Sum of Quantity' -> Orders


# --------------------------------------------------------------------------- #
# 3. Dashboard refresh lock state-machine                                     #
# --------------------------------------------------------------------------- #
pytest.importorskip("streamlit")


@pytest.fixture
def dash(monkeypatch, tmp_path):
    """Import the dashboard with its raw folder pointed at a temp dir.

    ``_raw_dir`` and ``discover_raw_files`` are the only two seams the refresh
    functions touch the filesystem through, so patching them isolates the lock
    logic from a real snapshot folder / pipeline load.
    """
    import dashboard

    folder = str(tmp_path)

    def _discover():
        out = []
        for p in glob.glob(os.path.join(folder, "all_demand_projections_*.xlsx")):
            m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(p))
            if m:
                out.append((m.group(1), p))
        return sorted(out, reverse=True)

    # These helpers moved into the dashboard_app package during the refactor;
    # refresh functions resolve them in their own module namespace, so patch the
    # modules that actually define/consume them (the dashboard facade only
    # re-exports copies, which the refresh code no longer reads).
    from dashboard_app import datasources as _ds, refresh as _rf
    from agent import data_io
    monkeypatch.setattr(_ds, "_raw_dir", lambda: folder)
    monkeypatch.setattr(_ds, "discover_raw_files", _discover)
    monkeypatch.setattr(_rf, "_refresh_log_path",
                        lambda name=_rf.DEMAND_LOG: os.path.join(folder, name))
    # sync_failures() reads all three pulls' markers, so pin the warehouse and
    # key-SKU folders here too — otherwise a real failure file in the repo would
    # leak into these assertions. Separate subfolders, because each pull's lock
    # is ".refresh.lock" inside its own folder.
    for name in ("wh", "ks"):
        os.makedirs(os.path.join(folder, name), exist_ok=True)
    monkeypatch.setattr(data_io, "_warehouse_dir",
                        lambda warehouse_dir=None: os.path.join(folder, "wh"))
    monkeypatch.setattr(_rf, "_key_skus_dir",
                        lambda: os.path.join(folder, "ks"))
    return dashboard, folder


def test_refresh_idle_when_no_lock(dash):
    dashboard, _ = dash
    assert dashboard.refresh_in_progress() == (False, None)


def test_refresh_running_with_fresh_lock(dash):
    dashboard, _ = dash
    with open(dashboard._refresh_lock_path(), "w", encoding="utf-8") as f:
        f.write("2026-07-10 09:00:00")

    running, started = dashboard.refresh_in_progress()
    assert running is True
    assert started == "2026-07-10 09:00:00"


def test_refresh_completion_clears_lock(dash):
    dashboard, folder = dash
    lock = dashboard._refresh_lock_path()
    with open(lock, "w", encoding="utf-8") as f:
        f.write("2026-07-10 09:00:00")
    t0 = 1_000_000.0
    os.utime(lock, (t0, t0))
    # A snapshot written AFTER the lock means the pull finished.
    _make_snapshot(folder, "2026-07-10", mtime=t0 + 100)

    running, _ = dashboard.refresh_in_progress()
    assert running is False
    assert not os.path.exists(lock)  # self-healed


def test_refresh_stale_lock_is_cleared(dash):
    dashboard, folder = dash
    lock = dashboard._refresh_lock_path()
    with open(lock, "w", encoding="utf-8") as f:
        f.write("old run")
    old = time.time() - (dashboard.REFRESH_STALE_SECONDS + 60)
    os.utime(lock, (old, old))
    # Only an OLDER snapshot exists, so it's not a completion — it's a crash.
    _make_snapshot(folder, "2026-07-01", mtime=old - 100)

    running, _ = dashboard.refresh_in_progress()
    assert running is False
    assert not os.path.exists(lock)


def test_start_refresh_blocks_when_already_running(dash):
    dashboard, _ = dash
    # Fresh lock, no newer snapshot -> a pull is in flight.
    with open(dashboard._refresh_lock_path(), "w", encoding="utf-8") as f:
        f.write("2026-07-10 10:00:00")

    ok, msg = dashboard.start_refresh()
    assert ok is False
    assert "already running" in msg


def test_start_refresh_launches_and_writes_lock(dash, monkeypatch):
    dashboard, folder = dash
    calls = {}

    class _FakePopen:
        def __init__(self, args, **kwargs):
            calls["args"] = args
            calls["kwargs"] = kwargs

    monkeypatch.setattr(dashboard.subprocess, "Popen", _FakePopen)

    ok, started = dashboard.start_refresh()

    assert ok is True
    assert os.path.exists(dashboard._refresh_lock_path())
    # Launched with THIS interpreter + the extract script...
    assert calls["args"][0] == sys.executable
    assert calls["args"][1] == dashboard.EXTRACT_SCRIPT
    # ...as the fast incremental pull (the nightly task does the full one)...
    assert calls["args"][2] == "--incremental"
    # ...and DEMAND_RAW_DIR pinned to the folder the dashboard reads.
    assert calls["kwargs"]["env"]["DEMAND_RAW_DIR"] == folder


def test_start_refresh_full_pull_when_incremental_disabled(dash, monkeypatch):
    dashboard, _ = dash
    calls = {}

    class _FakePopen:
        def __init__(self, args, **kwargs):
            calls["args"] = args

    monkeypatch.setattr(dashboard.subprocess, "Popen", _FakePopen)

    ok, _ = dashboard.start_refresh(incremental=False)

    assert ok is True
    assert "--incremental" not in calls["args"]


# --------------------------------------------------------------------------- #
# 4. Warehouse refresh lock state-machine                                     #
# --------------------------------------------------------------------------- #
# Same self-healing lock as above, except a warehouse snapshot is a FIVE-file
# set: completion must wait for every region, not just the first new file.
REGIONS = ["AU", "CA", "EU", "JP", "US"]


def _make_wh_files(folder, date_str, regions, mtime=None):
    paths = []
    for region in regions:
        path = os.path.join(
            folder, f"{region}_warehouse_projections_{date_str}.xlsx"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write("x")
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        paths.append(path)
    return paths


@pytest.fixture
def wh_dash(monkeypatch, tmp_path):
    """Import the dashboard with the warehouse folder pointed at a temp dir.

    The warehouse refresh functions reach the filesystem only through
    ``data_io._warehouse_dir`` (lock path + snapshot discovery), so patching it
    isolates the lock logic.
    """
    import dashboard
    from agent import data_io

    folder = str(tmp_path)
    monkeypatch.setattr(data_io, "_warehouse_dir",
                        lambda warehouse_dir=None: folder)
    from dashboard_app import refresh as _rf
    monkeypatch.setattr(_rf, "_refresh_log_path",
                        lambda name=_rf.DEMAND_LOG: os.path.join(folder, name))
    return dashboard, folder


def test_wh_refresh_idle_when_no_lock(wh_dash):
    dashboard, _ = wh_dash
    assert dashboard.warehouse_refresh_in_progress() == (False, None)


def test_wh_refresh_still_running_on_partial_snapshot(wh_dash):
    # 3 of 5 region files newer than the lock: the set is incomplete, so the
    # pull is still "running" — the dashboard must not serve a partial snapshot.
    dashboard, folder = wh_dash
    lock = dashboard._wh_refresh_lock_path()
    with open(lock, "w", encoding="utf-8") as f:
        f.write("2026-07-14 09:00:00")
    t0 = time.time()  # fresh lock — must not trip the staleness check
    os.utime(lock, (t0, t0))
    _make_wh_files(folder, "2026-07-14", ["AU", "CA", "EU"], mtime=t0 + 100)

    running, started = dashboard.warehouse_refresh_in_progress()
    assert running is True
    assert started == "2026-07-14 09:00:00"
    assert os.path.exists(lock)


def test_wh_refresh_completes_when_all_regions_land(wh_dash):
    dashboard, folder = wh_dash
    lock = dashboard._wh_refresh_lock_path()
    with open(lock, "w", encoding="utf-8") as f:
        f.write("2026-07-14 09:00:00")
    t0 = 1_000_000.0
    os.utime(lock, (t0, t0))
    _make_wh_files(folder, "2026-07-14", REGIONS, mtime=t0 + 100)

    running, _ = dashboard.warehouse_refresh_in_progress()
    assert running is False
    assert not os.path.exists(lock)  # self-healed


def test_wh_refresh_ignores_older_snapshot_group(wh_dash):
    # Yesterday's complete set predates the lock: not a completion.
    dashboard, folder = wh_dash
    lock = dashboard._wh_refresh_lock_path()
    with open(lock, "w", encoding="utf-8") as f:
        f.write("2026-07-14 09:00:00")
    t0 = time.time()  # fresh lock — must not trip the staleness check
    os.utime(lock, (t0, t0))
    _make_wh_files(folder, "2026-07-13", REGIONS, mtime=t0 - 100)

    running, _ = dashboard.warehouse_refresh_in_progress()
    assert running is True


def test_wh_refresh_stale_lock_is_cleared(wh_dash):
    dashboard, folder = wh_dash
    lock = dashboard._wh_refresh_lock_path()
    with open(lock, "w", encoding="utf-8") as f:
        f.write("old run")
    old = time.time() - (dashboard.REFRESH_STALE_SECONDS + 60)
    os.utime(lock, (old, old))

    running, _ = dashboard.warehouse_refresh_in_progress()
    assert running is False
    assert not os.path.exists(lock)


def test_start_warehouse_refresh_launches_and_writes_lock(wh_dash, monkeypatch):
    dashboard, folder = wh_dash
    calls = {}

    class _FakePopen:
        def __init__(self, args, **kwargs):
            calls["args"] = args
            calls["kwargs"] = kwargs

    monkeypatch.setattr(dashboard.subprocess, "Popen", _FakePopen)

    ok, _ = dashboard.start_warehouse_refresh()

    assert ok is True
    assert os.path.exists(dashboard._wh_refresh_lock_path())
    assert calls["args"][0] == sys.executable
    assert calls["args"][1] == dashboard.WAREHOUSE_EXTRACT_SCRIPT
    # WAREHOUSE_RAW_DIR pinned to the folder the dashboard reads.
    assert calls["kwargs"]["env"]["WAREHOUSE_RAW_DIR"] == folder


def test_start_warehouse_refresh_blocks_when_already_running(wh_dash):
    dashboard, _ = wh_dash
    with open(dashboard._wh_refresh_lock_path(), "w", encoding="utf-8") as f:
        f.write("2026-07-14 10:00:00")

    ok, msg = dashboard.start_warehouse_refresh()
    assert ok is False
    assert "already running" in msg


# --------------------------------------------------------------------------- #
# 5. ODBC driver resolution                                                   #
# --------------------------------------------------------------------------- #
# Regression cover for the 2026-08-04..08-10 outage: the connection string
# hardcoded "ODBC Driver 18 for SQL Server", which is installed on some hosts
# and not others (the sh-sw-dev dev server that serves the shared dashboard has
# only 13/17). Every pull launched there died with an opaque
#   IM002 ... Data source name not found and no default driver specified
# and, because .env lives on the shared network path, no single pinned driver
# name works for every host. The driver is therefore resolved at runtime.


@pytest.fixture
def drivers(monkeypatch):
    """Control what ``pyodbc.drivers()`` reports, and clear SQL_DRIVER."""
    monkeypatch.delenv("SQL_DRIVER", raising=False)
    monkeypatch.delenv("SQL_SERVER_CERT", raising=False)

    def _set(*names):
        monkeypatch.setattr(extract.pyodbc, "drivers", lambda: list(names))
    return _set


def test_resolve_driver_prefers_newest_installed(drivers):
    drivers("SQL Server", "ODBC Driver 13 for SQL Server",
            "ODBC Driver 17 for SQL Server", "ODBC Driver 18 for SQL Server")
    assert extract._resolve_driver() == "ODBC Driver 18 for SQL Server"


def test_resolve_driver_falls_back_to_17_when_18_absent(drivers):
    # Exactly the sh-sw-dev case that caused the outage.
    drivers("SQL Server", "ODBC Driver 13 for SQL Server",
            "SQL Server Native Client 11.0", "ODBC Driver 17 for SQL Server")
    assert extract._resolve_driver() == "ODBC Driver 17 for SQL Server"


def test_resolve_driver_ignores_non_sql_server_drivers(drivers):
    drivers("MySQL ODBC 9.4 Unicode Driver",
            "Microsoft Excel Driver (*.xls, *.xlsx, *.xlsm, *.xlsb)",
            "ODBC Driver 17 for SQL Server")
    assert extract._resolve_driver() == "ODBC Driver 17 for SQL Server"


def test_resolve_driver_falls_back_to_native_client_then_legacy(drivers):
    drivers("SQL Server", "SQL Server Native Client 11.0")
    assert extract._resolve_driver() == "SQL Server Native Client 11.0"

    drivers("SQL Server")
    assert extract._resolve_driver() == "SQL Server"


def test_resolve_driver_honours_explicit_sql_driver(drivers, monkeypatch):
    drivers("ODBC Driver 17 for SQL Server", "ODBC Driver 18 for SQL Server")
    monkeypatch.setenv("SQL_DRIVER", "ODBC Driver 17 for SQL Server")
    assert extract._resolve_driver() == "ODBC Driver 17 for SQL Server"


def test_resolve_driver_rejects_explicit_driver_that_is_not_installed(
    drivers, monkeypatch
):
    # The failure must name what was asked for AND what is available, instead of
    # the driver manager's opaque IM002.
    drivers("ODBC Driver 17 for SQL Server", "ODBC Driver 13 for SQL Server")
    monkeypatch.setenv("SQL_DRIVER", "ODBC Driver 18 for SQL Server")

    with pytest.raises(ValueError) as exc:
        extract._resolve_driver()

    msg = str(exc.value)
    assert "ODBC Driver 18 for SQL Server" in msg   # what was requested
    assert "ODBC Driver 17 for SQL Server" in msg   # what is installed
    assert "SQL_DRIVER" in msg


def test_resolve_driver_errors_clearly_when_none_installed(drivers):
    drivers("Microsoft Access Driver (*.mdb, *.accdb)")

    with pytest.raises(ValueError) as exc:
        extract._resolve_driver()

    msg = str(exc.value)
    assert "No SQL Server ODBC driver" in msg
    assert os.environ.get("COMPUTERNAME", "").lower() in msg.lower() or "host" in msg.lower()


def test_connection_string_uses_resolved_driver(drivers, monkeypatch):
    drivers("ODBC Driver 17 for SQL Server")
    monkeypatch.setenv("SQL_SERVER", "datawarehouse")
    monkeypatch.setenv("SQL_DATABASE", "SHSTGDB")
    monkeypatch.delenv("SQL_USER", raising=False)

    assert "DRIVER={ODBC Driver 17 for SQL Server};" in extract.connection_string()


def test_server_cert_pinning_rejected_on_pre_18_driver(drivers, monkeypatch, tmp_path):
    # ServerCertificate is a Driver 18+ keyword; older drivers ignore it, which
    # would silently downgrade certificate validation. Fail loudly instead.
    cert = tmp_path / "dw.cer"
    cert.write_text("x", encoding="utf-8")
    drivers("ODBC Driver 17 for SQL Server")
    monkeypatch.setenv("SQL_SERVER", "datawarehouse")
    monkeypatch.setenv("SQL_DATABASE", "SHSTGDB")
    monkeypatch.setenv("SQL_SERVER_CERT", str(cert))

    with pytest.raises(ValueError) as exc:
        extract.connection_string()

    assert "SQL_SERVER_CERT" in str(exc.value)
    assert "18" in str(exc.value)


def test_security_warning_is_logged_once_per_process(drivers, monkeypatch, caplog):
    # Every pull builds the connection string twice — once via
    # redacted_connection_string() for the "Connecting:" log line, once inside
    # connect() — so the SQL_TRUST_CERT security warning appeared twice per run
    # in logs_refresh.txt. Warn once per process instead.
    drivers("ODBC Driver 18 for SQL Server")
    monkeypatch.setenv("SQL_SERVER", "datawarehouse")
    monkeypatch.setenv("SQL_DATABASE", "SHSTGDB")
    monkeypatch.setenv("SQL_TRUST_CERT", "yes")
    monkeypatch.setattr(extract, "_WARNED_ONCE", set())

    with caplog.at_level("WARNING", logger="extract_demand_details"):
        extract.redacted_connection_string()
        extract.connection_string()

    warnings = [r for r in caplog.records if "SQL_TRUST_CERT" in r.getMessage()]
    assert len(warnings) == 1


# --------------------------------------------------------------------------- #
# 6. Failed pulls are reported, not silently swallowed                        #
# --------------------------------------------------------------------------- #
# The other half of the 2026-08 outage: _launch_refresh returned ok=True as soon
# as Popen succeeded and nobody ever looked at the child's exit code, so a pull
# that died on an ODBC error looked exactly like an idle button. Six days of
# failures went unnoticed. A failed pull must now leave a durable, visible trace.


def _write_run_log(folder, log_name, *lines):
    """A pull log shaped like a real one: run header, then the child's output."""
    path = os.path.join(folder, log_name)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n===== DW refresh (incremental) started 2026-08-10 09:41:26 "
                "on SOMEHOST =====\n")
        for line in lines:
            f.write(line + "\n")
    return path


def test_each_pull_writes_its_own_log(dash, monkeypatch):
    # Three children sharing one append handle is what corrupted the evidence.
    dashboard, _ = dash
    from dashboard_app import refresh as _rf

    names = {_rf.DEMAND_LOG, _rf.WAREHOUSE_LOG, _rf.KEY_SKUS_LOG}
    assert len(names) == 3


def test_failed_pull_is_recorded_and_reported(dash, monkeypatch):
    dashboard, folder = dash
    from dashboard_app import refresh as _rf

    lock = dashboard._refresh_lock_path()
    with open(lock, "w", encoding="utf-8") as f:
        f.write("2026-08-10 09:41:26")
    _write_run_log(
        folder, _rf.DEMAND_LOG,
        "2026-08-10 09:41:28,735 ERROR   Database error: ('IM002', "
        "'[IM002] [Microsoft][ODBC Driver Manager] Data source name not found "
        "and no default driver specified (0) (SQLDriverConnect)')",
    )

    running, _ = dashboard.refresh_in_progress()

    assert running is False           # not left spinning until the stale timeout
    assert not os.path.exists(lock)   # lock released
    failures = _rf.sync_failures()
    assert len(failures) == 1
    label, _when, _host, detail = failures[0]
    assert label == "Demand snapshot"
    assert "IM002" in detail          # the real cause reaches the user


def test_dead_child_reports_failure_without_waiting_for_stale_timeout(dash):
    dashboard, folder = dash
    from dashboard_app import refresh as _rf

    lock = dashboard._refresh_lock_path()
    with open(lock, "w", encoding="utf-8") as f:
        f.write("2026-08-10 09:41:26")

    class _DeadChild:
        returncode = 1

        def poll(self):
            return 1

    _rf._CHILDREN[lock] = _DeadChild()
    try:
        running, _ = dashboard.refresh_in_progress()
    finally:
        _rf._CHILDREN.pop(lock, None)

    assert running is False
    failures = _rf.sync_failures()
    assert failures and "exited with code 1" in failures[0][3]


def test_successful_pull_clears_a_previous_failure(dash):
    dashboard, folder = dash
    from dashboard_app import refresh as _rf

    lock = dashboard._refresh_lock_path()
    with open(_rf._failure_path(lock), "w", encoding="utf-8") as f:
        f.write("2026-08-10 09:41:26\tSOMEHOST\told IM002 failure")
    assert _rf.sync_failures()

    with open(lock, "w", encoding="utf-8") as f:
        f.write("2026-08-10 10:00:00")
    t0 = 1_000_000.0
    os.utime(lock, (t0, t0))
    _make_snapshot(folder, "2026-08-10", mtime=t0 + 100)

    running, _ = dashboard.refresh_in_progress()

    assert running is False
    assert _rf.sync_failures() == []   # banner goes away on a good run


def test_error_from_an_earlier_run_does_not_fail_the_current_one(dash):
    # Two runs land in the same day's file; only the current run's block counts.
    dashboard, folder = dash
    from dashboard_app import refresh as _rf

    _write_run_log(folder, _rf.DEMAND_LOG,
                   "2026-08-10 09:41:28 ERROR   Database error: IM002 boom")
    _write_run_log(folder, _rf.DEMAND_LOG,
                   "2026-08-10 10:05:00 INFO    Pulled 713,627 rows")

    lock = dashboard._refresh_lock_path()
    with open(lock, "w", encoding="utf-8") as f:
        f.write("2026-08-10 10:05:00")

    running, started = dashboard.refresh_in_progress()

    assert running is True             # still going, not falsely failed
    assert started == "2026-08-10 10:05:00"
    assert _rf.sync_failures() == []


def test_clear_sync_failures_dismisses_the_banner(dash):
    dashboard, _ = dash
    from dashboard_app import refresh as _rf

    with open(_rf._failure_path(dashboard._refresh_lock_path()), "w",
              encoding="utf-8") as f:
        f.write("2026-08-10 09:41:26\tSOMEHOST\tIM002")
    assert _rf.sync_failures()

    _rf.clear_sync_failures()

    assert _rf.sync_failures() == []


def test_clean_exit_with_lagging_output_is_not_reported_as_failure(dash):
    # The snapshot folder is a network share: a child can exit 0 a moment before
    # its file is visible. That must not be reported as a failed sync.
    dashboard, folder = dash
    from dashboard_app import refresh as _rf

    lock = dashboard._refresh_lock_path()
    with open(lock, "w", encoding="utf-8") as f:
        f.write("2026-08-10 10:00:00")

    class _CleanChild:
        returncode = 0

        def poll(self):
            return 0

    _rf._CHILDREN[lock] = _CleanChild()
    try:
        running, _ = dashboard.refresh_in_progress()
    finally:
        _rf._CHILDREN.pop(lock, None)

    assert running is True          # still pending, not failed
    assert _rf.sync_failures() == []
