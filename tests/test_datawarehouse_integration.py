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
    monkeypatch.setattr(_ds, "_raw_dir", lambda: folder)
    monkeypatch.setattr(_ds, "discover_raw_files", _discover)
    monkeypatch.setattr(_rf, "_refresh_log_path",
                        lambda: os.path.join(folder, "logs_refresh.txt"))
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
                        lambda: os.path.join(folder, "logs_refresh.txt"))
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
