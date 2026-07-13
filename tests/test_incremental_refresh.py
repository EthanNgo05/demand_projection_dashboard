"""Incremental refresh: SQL window override, snapshot merge, main() fallbacks.

The dashboard's refresh button runs ``extract_demand_details.py --incremental``:
only the last few weeks of actuals (plus all forward projections) are pulled
from the warehouse and merged into the newest existing snapshot, instead of
re-pulling 36 months. These tests cover the pure pieces — none touch SQL Server.
"""

import os
import re
from datetime import date, timedelta

import pandas as pd
import pytest

pytest.importorskip("pyodbc")  # extract_demand_details imports it at module load
import extract_demand_details as extract  # noqa: E402

SAMPLE_SQL = """\
declare @StartSunday date, @MonthBack int = 36;

select @StartSunday = TheStartingSunday
from pbi.calendar
where TheDate = cast(dateadd(month, -1 * @MonthBack, GETDATE()) as date);

-- INCREMENTAL_START_OVERRIDE (line replaced by extract_demand_details.py --incremental; do not remove)

select * from #gp_pos;
"""


def _frame(rows):
    """Build a snapshot-shaped DataFrame from (sku, cust, weekdate, pos) rows."""
    return pd.DataFrame(
        [
            {
                "'Demand'[DisplaySKU]": sku,
                "Description": f"Desc {sku}",
                "Custnmbr": cust,
                "WeekDate": week,
                "POS": pos,
                "Projection": pos + 1,
            }
            for sku, cust, week, pos in rows
        ]
    )


# --------------------------------------------------------------------------- #
# 1. Cutoff computation                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("weekday_offset", range(7))
def test_incremental_start_sunday_snaps_to_sunday(weekday_offset):
    today = date(2026, 7, 5) + timedelta(days=weekday_offset)  # Sun..Sat
    got = extract.incremental_start_sunday(3, today=today)
    assert got.weekday() == 6  # Sunday
    # At most 6 days before (today - 3 weeks), never after it.
    target = today - timedelta(weeks=3)
    assert timedelta(0) <= target - got <= timedelta(days=6)


def test_incremental_start_sunday_identity_on_sunday():
    # 2026-07-05 is a Sunday: 3 weeks back is also a Sunday, no snapping needed.
    assert extract.incremental_start_sunday(3, today=date(2026, 7, 5)) == date(
        2026, 6, 14
    )


# --------------------------------------------------------------------------- #
# 2. SQL rewrite                                                              #
# --------------------------------------------------------------------------- #
def test_build_incremental_sql_replaces_marker():
    out = extract.build_incremental_sql(SAMPLE_SQL, date(2026, 6, 14))

    assert extract.INCREMENTAL_MARKER not in out
    assert (
        "select @StartSunday = TheStartingSunday "
        "from pbi.calendar where TheDate = '2026-06-14';" in out
    )
    # The original 36-month assignment still runs first; the override wins by
    # coming after it. Everything else in the batch is untouched.
    assert "dateadd(month, -1 * @MonthBack, GETDATE())" in out
    assert "select * from #gp_pos;" in out
    assert len(out.splitlines()) == len(SAMPLE_SQL.splitlines())


def test_build_incremental_sql_raises_without_marker():
    no_marker = SAMPLE_SQL.replace(extract.INCREMENTAL_MARKER, "-- plain comment")
    with pytest.raises(ValueError, match="INCREMENTAL_START_OVERRIDE"):
        extract.build_incremental_sql(no_marker, date(2026, 6, 14))


def test_default_sql_file_contains_marker():
    # Guards against the marker line being dropped when someone edits the .sql:
    # without it --incremental fails loudly instead of running the full pull.
    assert extract.INCREMENTAL_MARKER in extract.read_sql_file(extract.DEFAULT_SQL)


# --------------------------------------------------------------------------- #
# 3. Merge                                                                    #
# --------------------------------------------------------------------------- #
CUTOFF = date(2026, 6, 14)


def test_merge_partitions_at_cutoff():
    previous = _frame([
        ("ST1", "CUST1", "2026-05-31", 10),   # before cutoff -> kept
        ("ST1", "CUST1", "2026-06-14", 99),   # at cutoff -> replaced by fresh
        ("ST1", "CUST1", "2026-06-21", 98),   # after cutoff -> replaced by fresh
    ])
    fresh = _frame([
        ("ST1", "CUST1", "2026-06-14", 11),
        ("ST1", "CUST1", "2026-06-21", 12),
    ])

    merged = extract.merge_snapshots(previous, fresh, CUTOFF)

    assert list(merged["POS"]) == [10, 11, 12]
    assert list(merged["WeekDate"]) == [
        pd.Timestamp("2026-05-31"),
        pd.Timestamp("2026-06-14"),
        pd.Timestamp("2026-06-21"),
    ]


def test_merge_drops_fresh_rows_below_cutoff():
    previous = _frame([("ST1", "CUST1", "2026-05-31", 10)])
    fresh = _frame([
        ("ST1", "CUST1", "2026-05-31", 55),   # shouldn't happen; dropped
        ("ST1", "CUST1", "2026-06-14", 11),
    ])

    merged = extract.merge_snapshots(previous, fresh, CUTOFF)

    assert list(merged["POS"]) == [10, 11]


def test_merge_column_set_follows_fresh():
    previous = _frame([("ST1", "CUST1", "2026-05-31", 10)])
    previous["Stale Col"] = "old"             # dropped: not in fresh
    fresh = _frame([("ST1", "CUST1", "2026-06-14", 11)])
    fresh["New Col"] = "new"                  # appears; NaN for old rows

    merged = extract.merge_snapshots(previous, fresh, CUTOFF)

    assert list(merged.columns) == list(fresh.columns)
    assert "Stale Col" not in merged.columns
    assert pd.isna(merged.loc[0, "New Col"]) and merged.loc[1, "New Col"] == "new"


def test_merge_normalizes_weekdate_dtypes():
    # Excel round-trips give datetime64/str; SQL gives datetime.date objects.
    previous = _frame([("ST1", "CUST1", "2026-05-31", 10)])          # str
    fresh = _frame([("ST1", "CUST1", date(2026, 6, 14), 11)])        # date obj

    merged = extract.merge_snapshots(previous, fresh, CUTOFF)

    assert pd.api.types.is_datetime64_any_dtype(merged["WeekDate"])
    assert len(merged) == 2


def test_merge_sorts_deterministically():
    previous = _frame([
        ("ST2", "CUST1", "2026-05-31", 2),
        ("ST1", "CUST2", "2026-05-31", 1),
        ("ST1", "CUST1", "2026-05-24", 0),
    ])
    fresh = _frame([("ST1", "CUST1", "2026-06-14", 3)])

    merged = extract.merge_snapshots(previous, fresh, CUTOFF)

    keys = list(
        zip(merged["WeekDate"].dt.date.astype(str),
            merged["'Demand'[DisplaySKU]"], merged["Custnmbr"])
    )
    assert keys == sorted(keys)


# --------------------------------------------------------------------------- #
# 4. Previous-snapshot discovery / loading                                    #
# --------------------------------------------------------------------------- #
def test_find_previous_snapshot_newest_by_filename_date(tmp_path):
    for d in ["2026-07-01", "2026-07-03", "2026-07-02"]:
        (tmp_path / f"all_demand_projections_{d}.xlsx").write_text("x")
    (tmp_path / "all_demand_projections_final.xlsx").write_text("undated")

    got = extract.find_previous_snapshot(str(tmp_path))

    assert os.path.basename(got) == "all_demand_projections_2026-07-03.xlsx"


def test_find_previous_snapshot_none_when_empty(tmp_path):
    assert extract.find_previous_snapshot(str(tmp_path)) is None


def test_load_previous_snapshot_roundtrip(tmp_path):
    df = _frame([("ST1", "CUST1", "2026-05-31", 10)])
    path = tmp_path / "all_demand_projections_2026-07-01.xlsx"
    extract.write_powerbi_xlsx(df, str(path))

    got = extract.load_previous_snapshot(str(path))

    assert got is not None
    assert pd.api.types.is_datetime64_any_dtype(got["WeekDate"])
    assert list(got["'Demand'[DisplaySKU]"]) == ["ST1"]


def test_load_previous_snapshot_rejects_missing_columns(tmp_path):
    df = _frame([("ST1", "CUST1", "2026-05-31", 10)]).drop(columns=["Projection"])
    path = tmp_path / "all_demand_projections_2026-07-01.xlsx"
    extract.write_powerbi_xlsx(df, str(path))

    assert extract.load_previous_snapshot(str(path)) is None


def test_load_previous_snapshot_rejects_unreadable_file(tmp_path):
    path = tmp_path / "all_demand_projections_2026-07-01.xlsx"
    path.write_text("not an xlsx")

    assert extract.load_previous_snapshot(str(path)) is None


# --------------------------------------------------------------------------- #
# 5. Locked-destination retry (same-day overwrite vs a reader/Excel)          #
# --------------------------------------------------------------------------- #
def test_replace_retries_through_transient_lock(tmp_path, monkeypatch):
    src = tmp_path / "src.xlsx"
    dst = tmp_path / "dst.xlsx"
    src.write_text("new")
    dst.write_text("old")
    real_replace, tries = os.replace, []

    def flaky(a, b):
        tries.append(1)
        if len(tries) < 3:
            raise PermissionError("locked")
        real_replace(a, b)

    monkeypatch.setattr(extract.os, "replace", flaky)
    monkeypatch.setattr(extract.time, "sleep", lambda s: None)

    extract._replace_with_retry(str(src), str(dst))

    assert dst.read_text() == "new"
    assert len(tries) == 3


def test_replace_gives_clear_error_when_lock_persists(tmp_path, monkeypatch):
    src = tmp_path / "src.xlsx"
    src.write_text("new")
    monkeypatch.setattr(
        extract.os, "replace",
        lambda a, b: (_ for _ in ()).throw(PermissionError("locked")),
    )
    monkeypatch.setattr(extract.time, "sleep", lambda s: None)

    with pytest.raises(ValueError, match="stayed locked"):
        extract._replace_with_retry(str(src), str(tmp_path / "dst.xlsx"))


# --------------------------------------------------------------------------- #
# 6. main() end-to-end (SQL mocked)                                           #
# --------------------------------------------------------------------------- #
@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    """Point the extract's raw folder at a temp dir (also used by out path)."""
    folder = str(tmp_path)
    monkeypatch.setattr(extract, "_default_raw_dir", lambda: folder)
    return tmp_path


def _fake_loader(monkeypatch, result):
    """Stub load_demand_details; record the sql_transform each call got."""
    calls = []

    def fake(path=extract.DEFAULT_SQL, sql_transform=None):
        calls.append(sql_transform)
        return result.copy()

    monkeypatch.setattr(extract, "load_demand_details", fake)
    return calls


def test_main_incremental_merges_and_overwrites_same_day_file(raw_dir, monkeypatch):
    cutoff = extract.incremental_start_sunday(3)
    before, after = cutoff - timedelta(days=7), cutoff + timedelta(days=7)
    # Previous file is TODAY's snapshot -> out path == previous path (the
    # same-day overwrite case: read fully first, then atomic-replace).
    prev_path = raw_dir / f"all_demand_projections_{date.today():%Y-%m-%d}.xlsx"
    extract.write_powerbi_xlsx(
        _frame([
            ("ST1", "CUST1", str(before), 10),
            ("ST1", "CUST1", str(cutoff), 99),  # gets replaced by fresh
        ]),
        str(prev_path),
    )
    calls = _fake_loader(
        monkeypatch,
        _frame([("ST1", "CUST1", str(cutoff), 11), ("ST1", "CUST1", str(after), 12)]),
    )

    assert extract.main(["--incremental"]) == 0

    assert calls and calls[0] is not None  # incremental transform was applied
    out = pd.read_excel(prev_path, header=2)
    assert list(out["POS"]) == [10, 11, 12]


def test_main_incremental_falls_back_to_full_without_previous(raw_dir, monkeypatch):
    calls = _fake_loader(
        monkeypatch, _frame([("ST1", "CUST1", "2026-06-14", 11)])
    )

    assert extract.main(["--incremental"]) == 0

    assert calls == [None]  # ran the un-transformed (full) batch
    assert list(raw_dir.glob("all_demand_projections_*.xlsx"))


def test_main_incremental_falls_back_when_previous_unusable(raw_dir, monkeypatch):
    bad = raw_dir / "all_demand_projections_2026-07-01.xlsx"
    extract.write_powerbi_xlsx(
        _frame([("ST1", "CUST1", "2026-05-31", 10)]).drop(columns=["Projection"]),
        str(bad),
    )
    calls = _fake_loader(
        monkeypatch, _frame([("ST1", "CUST1", "2026-06-14", 11)])
    )

    assert extract.main(["--incremental"]) == 0
    assert calls == [None]


def test_main_incremental_refuses_empty_pull(raw_dir, monkeypatch):
    prev_path = raw_dir / "all_demand_projections_2026-07-01.xlsx"
    extract.write_powerbi_xlsx(
        _frame([("ST1", "CUST1", "2026-05-31", 10)]), str(prev_path)
    )
    mtime = os.path.getmtime(prev_path)
    _fake_loader(monkeypatch, _frame([]).reindex(
        columns=_frame([("x", "y", "2026-01-04", 0)]).columns
    ))

    assert extract.main(["--incremental"]) == 1

    # Previous snapshot untouched; no new file written.
    assert os.path.getmtime(prev_path) == mtime
    assert len(list(raw_dir.glob("all_demand_projections_*.xlsx"))) == 1


def test_main_default_is_full_pull(raw_dir, monkeypatch):
    calls = _fake_loader(
        monkeypatch, _frame([("ST1", "CUST1", "2026-06-14", 11)])
    )

    assert extract.main([]) == 0

    assert calls == [None]  # nightly path: no transform, plain full batch
