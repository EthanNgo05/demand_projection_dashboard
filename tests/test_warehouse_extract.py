"""Warehouse extract: raw SQL rows -> per-region PowerBI-layout snapshot files.

Covers the pure-python half of ``extract_warehouse_projections.py`` (no SQL
Server): the region transform (warehouse mapping, ProjType collapsing, key
stripping, zero-total dropping) and the regional writer whose files must
round-trip through the dashboard's own reader and snapshot discovery.
"""

import os
import sys

import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

pytest.importorskip("pyodbc")  # extract_demand_details imports it at module load
import extract_warehouse_projections as wex  # noqa: E402

from agent import data_io  # noqa: E402

W1 = pd.Timestamp("2026-07-19")
W2 = pd.Timestamp("2026-07-26")


def raw_rows(rows):
    """Build a raw-pull frame from (CUSTNMBR, ITEMNMBR, WeekDate, Warehouse,
    ProjType, Proj) tuples."""
    return pd.DataFrame(rows, columns=list(wex.REQUIRED_COLUMNS))


def test_us_sums_both_warehouses_and_projtypes():
    frames = wex.transform_to_regions(raw_rows([
        ("CUSTA", "SKU1", W1, "LBC", "Projection", 3),
        ("CUSTA", "SKU1", W1, "NJ", "Projection", 4),
        ("CUSTA", "SKU1", W1, "LBC", "PromoProj", 2),
    ]))

    us = frames["US"]
    assert list(us.columns) == wex.PBI_COLUMNS
    assert len(us) == 1
    assert us["Sum of Proj"].iloc[0] == 9  # LBC+NJ, promo+base collapsed


def test_each_warehouse_lands_in_its_region():
    frames = wex.transform_to_regions(raw_rows([
        ("CUSTA", "SKU1", W1, "SH-CTS", "Projection", 1),
        ("CUSTA", "SKU1", W1, "ACR", "Projection", 2),
        ("CUSTA", "SKU1", W1, "YYZ5", "Projection", 3),
        ("CUSTA", "SKU1", W1, "NETDEPOT", "Projection", 4),
    ]))

    totals = {r: f["Sum of Proj"].sum() for r, f in frames.items()}
    assert totals == {"US": 0, "EU": 1, "AU": 2, "CA": 3, "JP": 4}
    # Every region present even when empty — a snapshot is always 5 files.
    assert set(frames) == set(wex.REGION_WAREHOUSES)
    assert frames["US"].empty and list(frames["US"].columns) == wex.PBI_COLUMNS


def test_unmapped_warehouse_fails_loudly():
    with pytest.raises(ValueError, match="TX9"):
        wex.transform_to_regions(raw_rows([
            ("CUSTA", "SKU1", W1, "TX9", "Projection", 5),
        ]))


def test_padded_keys_are_stripped():
    # GP CHAR columns come back space-padded; padded and clean variants of the
    # same key must aggregate together and come out clean.
    frames = wex.transform_to_regions(raw_rows([
        ("CUSTA   ", "SKU1  ", W1, "LBC", "Projection", 3),
        ("CUSTA", "SKU1", W1, "NJ ", "Projection", 4),
    ]))

    us = frames["US"]
    assert len(us) == 1
    assert us[wex.PBI_COLUMNS[0]].iloc[0] == "SKU1"
    assert us["CUSTNMBR"].iloc[0] == "CUSTA"
    assert us["Sum of Proj"].iloc[0] == 7


def test_zero_totals_dropped_even_when_offsetting():
    # The SQL filters Proj <> 0, but offsetting promo/base rows can still net
    # to zero after aggregation — and zero means "missing", so the row goes.
    frames = wex.transform_to_regions(raw_rows([
        ("CUSTA", "SKU1", W1, "LBC", "Projection", 5),
        ("CUSTA", "SKU1", W1, "LBC", "PromoProj", -5),
        ("CUSTA", "SKU1", W2, "LBC", "Projection", 2),
    ]))

    us = frames["US"]
    assert list(us["WeekDate"]) == [W2]


def test_written_snapshot_round_trips_through_dashboard_reader(tmp_path):
    frames = wex.transform_to_regions(raw_rows([
        ("CUSTA", "SKU1", W1, "LBC", "Projection", 3),
        ("CUSTA", "SKU1", W2, "NJ", "Projection", 4),
        ("CUSTB", "SKU2", W1, "NETDEPOT", "Projection", 6),
    ]))
    written = wex.write_region_files(frames, str(tmp_path), pd.Timestamp("2026-07-14"))

    assert len(written) == 5
    # Snapshot discovery groups the set under its date.
    snapshots = data_io.discover_warehouse_files(str(tmp_path))
    assert list(snapshots) == ["2026-07-14"]
    assert len(snapshots["2026-07-14"]) == 5

    # The dashboard reader parses the files and reconstructs missing cells:
    # SKU2/CUSTB (JP) has a W1 row only -> NaN at W2 across the week union.
    combined = data_io.combine_warehouse_projections(
        [(p, p) for p in snapshots["2026-07-14"]]
    )
    assert set(combined["Location"]) == {"US", "JP"}
    jp_w2 = combined[(combined["Location"] == "JP") & (combined["WeekDate"] == W2)]
    assert jp_w2["Projection"].isna().all()
    us_vals = combined[combined["Location"] == "US"].dropna(subset=["Projection"])
    assert us_vals["Projection"].sum() == 7
