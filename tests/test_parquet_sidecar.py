"""The Parquet sidecar: it must actually get written, and it must match the xlsx.

Background — this is a regression suite for a bug that ran unnoticed for at least
ten days. ``run_query`` builds the frame with ``pd.DataFrame.from_records`` over
pyodbc rows, and pyodbc returns ``decimal.Decimal`` for SQL ``decimal``/``numeric``
columns, which pandas stores as **object** dtype. pyarrow refuses to convert
Decimal to double, so ``write_parquet_sidecar`` failed on *every* run:

    Could not write Parquet sidecar ...: ("Could not convert Decimal('24.00000000')
    with type decimal.Decimal: tried to convert to double", 'Conversion failed for
    column Sum of Quantity with type object')

The ``.xlsx`` write never noticed (openpyxl serialises Decimal as a number), so the
only symptom was slowness: with no fresh sidecar, ``agent.data_io.read_raw_frame``
re-parsed the 34 MB workbook — 66s instead of 0.15s — on the first page load after
every refresh, and ``load_previous_snapshot`` paid the same 66s inside every
incremental pull.

The load-bearing test here is ``test_sidecar_matches_xlsx_roundtrip``: because
``read_raw_frame`` treats the two files as interchangeable, a sidecar that differs
from the workbook would make displayed numbers depend on which file was read.
"""

import logging
import os
from decimal import Decimal
from io import BytesIO

import numpy as np
import pandas as pd
import pytest

import extract_demand_details as extract

RAW_SQL_COLUMNS = list(extract.SQL_TO_POWERBI_FORMAT)


def sql_shaped_frame():
    """A frame shaped exactly as ``run_query`` produces one.

    Measure columns are object dtype holding ``Decimal``/``None`` (what pyodbc
    gives for SQL decimal columns); keys carry the fixed-width space padding the
    warehouse export is known for; Description includes a blank and a None.
    Column names are the RAW SQL names so ``select_and_rename`` can process it.
    """
    return pd.DataFrame({
        "DisplaySKU": ["BT1028", "BT1028", "ZZ9  ", "K1", "AS100"],
        "LongName": ["twin wall mount pumps", None, "", "   ", "widget"],
        "Custnmbr": ["AMAZON-DC", "MARVAL-FBM      ", "Others - US", "COSTCO", "X"],
        "WeekDate": pd.to_datetime(["2026-07-19"] * 5),
        "SalesUnits": [191.0, np.nan, 0.0, 3.0, 7.0],
        "ProjQty": [220.0, 10.0, np.nan, 0.0, 1.0],
        "PromoProj": [0.0, 0.0, 0.0, 0.0, 0.0],
        "OnHand": [Decimal("361.00000000"), Decimal("-5.25000000"), None,
                   Decimal("0.00000001"), Decimal("1E+3")],
        "OnOrder": [9.0, 0.0, 96.0, 0.0, 0.0],
        # The column that actually broke: SQL decimal -> Decimal -> object.
        "Quantity": [Decimal("24.00000000"), None, Decimal("0E-8"),
                     Decimal("1.50000000"), Decimal("3.00000000")],
        "InStock": [0.0, 0.0, 0.0, 1.0, 0.0],
        "StoreCount": [1.0, np.nan, 94.0, 0.0, 2.0],
    })[RAW_SQL_COLUMNS]


# --------------------------------------------------------------------------
# 1. The dtype normalisation (root cause)
# --------------------------------------------------------------------------
def test_select_and_rename_returns_float64_measures():
    """Every measure column must be float64 — object dtype is what broke Parquet."""
    out = extract.select_and_rename(sql_shaped_frame())
    for col in extract.NUMERIC_COLUMNS:
        assert out[col].dtype == "float64", f"{col} is {out[col].dtype}, not float64"


def test_decimal_values_survive_conversion_exactly():
    out = extract.select_and_rename(sql_shaped_frame())
    got = out["Sum of Quantity"].tolist()
    assert got[0] == float(Decimal("24.00000000")) == 24.0
    assert pd.isna(got[1])                      # None -> NaN
    assert got[2] == 0.0                        # Decimal('0E-8') -> 0.0
    assert got[3] == 1.5
    assert out["On Hand"].tolist()[1] == -5.25  # negatives intact


def test_text_and_date_columns_are_untouched():
    """Coercion must not reach the keys — the space padding is load-bearing
    (agent/data_io._clean strips it, and it must still be there to strip)."""
    out = extract.select_and_rename(sql_shaped_frame())
    assert out["Custnmbr"].tolist()[1] == "MARVAL-FBM      "
    assert out["'Demand'[DisplaySKU]"].tolist()[2] == "ZZ9  "
    assert out["LongName" if "LongName" in out.columns else "Description"].tolist()[2] == ""
    assert pd.api.types.is_datetime64_any_dtype(out["WeekDate"])


def test_non_numeric_measure_is_coerced_and_logged(caplog):
    """A surprise string must not sink the pull, but must not vanish silently."""
    raw = sql_shaped_frame()
    raw["Quantity"] = ["24", "not-a-number", None, "1.5", "3"]
    with caplog.at_level(logging.WARNING, logger=extract.log.name):
        out = extract.select_and_rename(raw)
    assert out["Sum of Quantity"].dtype == "float64"
    assert out["Sum of Quantity"].tolist()[0] == 24.0     # numeric strings convert
    assert pd.isna(out["Sum of Quantity"].tolist()[1])    # the bad one -> NaN
    assert any("non-numeric" in r.getMessage() for r in caplog.records), \
        "the coerced value was not reported"


def test_no_warning_when_everything_converts(caplog):
    with caplog.at_level(logging.WARNING, logger=extract.log.name):
        extract.select_and_rename(sql_shaped_frame())
    assert not [r for r in caplog.records if "non-numeric" in r.getMessage()]


def test_coerce_is_a_noop_on_an_already_clean_frame():
    """Idempotent — the incremental path effectively runs it twice."""
    once = extract.select_and_rename(sql_shaped_frame())
    twice = extract.coerce_numeric_columns(once)
    pd.testing.assert_frame_equal(once, twice, check_exact=True, check_dtype=True)


def test_coerce_tolerates_a_missing_optional_column():
    raw = sql_shaped_frame().drop(columns=["StoreCount"])
    out = extract.select_and_rename(raw)
    assert "Store Count" not in out.columns      # absent, not invented
    assert out["Sum of Quantity"].dtype == "float64"


# --------------------------------------------------------------------------
# 2. The sidecar actually gets written
# --------------------------------------------------------------------------
def test_sidecar_write_succeeds_on_a_decimal_frame(tmp_path, caplog):
    """THE regression test: this exact frame used to fail the write every run."""
    xlsx = str(tmp_path / "all_demand_projections_2026-07-29.xlsx")
    df = extract.select_and_rename(sql_shaped_frame())
    with caplog.at_level(logging.WARNING, logger=extract.log.name):
        out = extract.write_parquet_sidecar(df, xlsx)
    assert out is not None, "sidecar write still failing"
    assert os.path.exists(out)
    assert not [r for r in caplog.records if "Could not write Parquet sidecar" in r.getMessage()]


def test_sidecar_leaves_no_temp_file(tmp_path):
    xlsx = str(tmp_path / "snap.xlsx")
    extract.write_parquet_sidecar(extract.select_and_rename(sql_shaped_frame()), xlsx)
    assert [p.name for p in tmp_path.iterdir()] == ["snap.parquet"]


def test_sidecar_write_stays_best_effort(tmp_path, monkeypatch, caplog):
    """A write failure must still be swallowed — the xlsx is the source of truth."""
    def boom(*a, **k):
        raise RuntimeError("no parquet engine")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with caplog.at_level(logging.WARNING, logger=extract.log.name):
        out = extract.write_parquet_sidecar(
            extract.select_and_rename(sql_shaped_frame()), str(tmp_path / "s.xlsx"))
    assert out is None
    assert any("Could not write Parquet sidecar" in r.getMessage() for r in caplog.records)
    assert list(tmp_path.iterdir()) == []        # no half-written temp left behind


# --------------------------------------------------------------------------
# 3. Fidelity — the load-bearing test
# --------------------------------------------------------------------------
def _as_float_measures(df):
    """Cast measure columns to float64 so a comparison tests VALUES, not pandas'
    read_excel dtype inference (see test_xlsx_infers_int64_for_whole_columns)."""
    out = df.copy()
    for col in extract.NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("float64")
    return out


def test_sidecar_matches_xlsx_roundtrip(tmp_path):
    """Every value in the sidecar must equal ``pd.read_excel(path, header=2)``.

    ``agent.data_io.read_raw_frame`` serves whichever of the two is current, so a
    value divergence would mean the dashboard's numbers depend on which file it
    happened to read. This is the load-bearing test of the sidecar work.

    Text and date columns are compared with dtypes pinned. Measures are compared
    as float64 on both sides, because read_excel's integer inference is
    data-dependent (see the next test) while the sidecar's float64 is not — the
    thing that must not drift is the numbers. ``rtol`` covers only Excel's
    ~15-significant-digit storage limit, documented in ``write_parquet_sidecar``;
    a genuine difference is far larger.
    """
    xlsx = str(tmp_path / "all_demand_projections_2026-07-29.xlsx")
    df = extract.select_and_rename(sql_shaped_frame())

    extract.write_powerbi_xlsx(df, xlsx)
    sidecar = extract.write_parquet_sidecar(df, xlsx)
    assert sidecar is not None

    via_xlsx = pd.read_excel(xlsx, header=2)
    via_parquet = pd.read_parquet(sidecar)

    assert list(via_parquet.columns) == list(via_xlsx.columns)
    # Text + date columns: dtypes and values both pinned.
    for col in extract.TEXT_COLUMNS:
        pd.testing.assert_series_equal(via_parquet[col], via_xlsx[col],
                                       check_exact=True, check_dtype=True)
    # Measures: values pinned, dtype normalised on both sides.
    pd.testing.assert_frame_equal(
        _as_float_measures(via_parquet), _as_float_measures(via_xlsx),
        check_dtype=True, check_exact=False, rtol=1e-12,
    )


def test_sidecar_measures_are_always_float64(tmp_path):
    """The sidecar's measure dtypes must not depend on the data.

    read_excel gives int64 for a column that happens to hold only whole numbers
    with no nulls, and float64 otherwise — so a workbook's dtypes shift from
    snapshot to snapshot. The sidecar is float64 unconditionally, which is the
    more predictable of the two and what `_clean` gets today anyway (POS /
    Projection / Sum of Quantity always carry nulls in real data).
    """
    xlsx = str(tmp_path / "snap.xlsx")
    df = extract.select_and_rename(sql_shaped_frame())
    extract.write_powerbi_xlsx(df, xlsx)
    via_parquet = pd.read_parquet(extract.write_parquet_sidecar(df, xlsx))
    for col in extract.NUMERIC_COLUMNS:
        assert via_parquet[col].dtype == "float64", f"{col} is {via_parquet[col].dtype}"


def test_xlsx_infers_int64_for_whole_columns(tmp_path):
    """Document the one remaining dtype difference, and prove it is value-only.

    If a future pandas stops inferring int64 here, this test fails and the
    dtype-flexibility in ``test_sidecar_matches_xlsx_roundtrip`` can be tightened
    rather than being carried forever on faith.
    """
    xlsx = str(tmp_path / "snap.xlsx")
    df = extract.select_and_rename(sql_shaped_frame())
    extract.write_powerbi_xlsx(df, xlsx)
    sidecar = extract.write_parquet_sidecar(df, xlsx)

    via_xlsx = pd.read_excel(xlsx, header=2)
    via_parquet = pd.read_parquet(sidecar)

    # "Promo Qty" is all zeros with no nulls -> Excel round-trip infers int64.
    assert via_xlsx["Promo Qty"].dtype == "int64"
    assert via_parquet["Promo Qty"].dtype == "float64"
    # ...and every value still matches, which is what actually matters.
    for col in extract.NUMERIC_COLUMNS:
        assert via_parquet[col].astype("float64").equals(
            via_xlsx[col].astype("float64")
        ), f"{col} values diverged"


def test_blank_text_becomes_na_like_excel_does(tmp_path):
    """Pin the specific normalisation the fidelity test depends on: Excel writes a
    zero-length string as an empty cell, which reads back as NaN."""
    xlsx = str(tmp_path / "snap.xlsx")
    df = extract.select_and_rename(sql_shaped_frame())
    extract.write_powerbi_xlsx(df, xlsx)
    sidecar = extract.write_parquet_sidecar(df, xlsx)

    desc_x = pd.read_excel(xlsx, header=2)["Description"]
    desc_p = pd.read_parquet(sidecar)["Description"]
    assert pd.isna(desc_x.iloc[2]) and pd.isna(desc_p.iloc[2])   # "" -> NaN both ways
    assert desc_x.iloc[3] == desc_p.iloc[3] == "   "             # whitespace kept
    assert desc_p.iloc[0] == "twin wall mount pumps"


def test_blank_masking_does_not_touch_the_xlsx_frame():
    """``_blank_text_to_na`` must not mutate its input.

    The frame handed to ``write_parquet_sidecar`` is the SAME object main() passes
    to ``write_powerbi_xlsx`` and (before that) to ``_apply_output_filters``, which
    drops rows on ``Custnmbr.notna()``. Masking in place would silently change
    which rows ship.
    """
    df = extract.select_and_rename(sql_shaped_frame())
    before = df.copy()
    extract._blank_text_to_na(df)
    pd.testing.assert_frame_equal(df, before, check_exact=True, check_dtype=True)


def test_output_filters_keep_the_same_rows_with_a_blank_customer():
    """Pins the boundary of change #2: an empty-string Custnmbr must still be KEPT
    by _apply_output_filters, exactly as it was before the sidecar work."""
    raw = sql_shaped_frame()
    raw["Custnmbr"] = ["AMAZON-DC", "", "Others - US", "COSTCO", "X"]
    df = extract.select_and_rename(raw)
    kept = extract._apply_output_filters(df)
    assert kept["Custnmbr"].tolist() == ["AMAZON-DC", "", "Others - US", "COSTCO"]
    # (the 'AS100' SKU is the only row dropped, per FILTER_BANNER)


# --------------------------------------------------------------------------
# 4. The incremental path
# --------------------------------------------------------------------------
def test_merge_produces_float64_not_object():
    """The incremental-specific failure: previous (xlsx, float64) concatenated with
    fresh (SQL, object) used to yield an object column."""
    fresh = extract.select_and_rename(sql_shaped_frame())
    previous = fresh.copy()
    previous["WeekDate"] = pd.to_datetime(["2026-06-07"] * len(previous))
    merged = extract.merge_snapshots(previous, fresh, pd.Timestamp("2026-07-12").date())
    for col in extract.NUMERIC_COLUMNS:
        assert merged[col].dtype == "float64", f"{col} is {merged[col].dtype}"


def _write_pair(tmp_path, name="all_demand_projections_2026-07-29.xlsx"):
    """A snapshot workbook plus its sidecar, with the sidecar newer."""
    xlsx = str(tmp_path / name)
    df = extract.select_and_rename(sql_shaped_frame())
    extract.write_powerbi_xlsx(df, xlsx)
    sidecar = extract.write_parquet_sidecar(df, xlsx)
    os.utime(xlsx, (1000, 1000))
    os.utime(sidecar, (2000, 2000))
    return xlsx, sidecar


def test_read_snapshot_frame_prefers_a_current_sidecar(tmp_path, monkeypatch):
    xlsx, _ = _write_pair(tmp_path)

    def fail(*a, **k):
        raise AssertionError("read_excel was called despite a current sidecar")

    monkeypatch.setattr(pd, "read_excel", fail)
    got = extract._read_snapshot_frame(xlsx)
    assert len(got) == 5


def test_read_snapshot_frame_accepts_an_equal_mtime_sidecar(tmp_path, monkeypatch):
    """The rule is ``>=`` (mirroring data_io.read_raw_frame), not ``>``."""
    xlsx, sidecar = _write_pair(tmp_path)
    os.utime(xlsx, (1500, 1500))
    os.utime(sidecar, (1500, 1500))
    monkeypatch.setattr(pd, "read_excel",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("used xlsx")))
    assert len(extract._read_snapshot_frame(xlsx)) == 5


def test_read_snapshot_frame_ignores_a_stale_sidecar(tmp_path):
    """A sidecar older than the workbook is stale — the workbook wins."""
    xlsx, sidecar = _write_pair(tmp_path)
    os.utime(sidecar, (500, 500))            # now older than the xlsx
    os.utime(xlsx, (1000, 1000))
    # Make the two distinguishable, then confirm we got the workbook's content.
    pd.DataFrame({"marker": [1]}).to_parquet(sidecar, index=False)
    os.utime(sidecar, (500, 500))
    got = extract._read_snapshot_frame(xlsx)
    assert "marker" not in got.columns
    assert len(got) == 5


def test_read_snapshot_frame_falls_back_when_sidecar_is_corrupt(tmp_path, caplog):
    xlsx, sidecar = _write_pair(tmp_path)
    with open(sidecar, "wb") as fh:
        fh.write(b"not parquet at all")
    with caplog.at_level(logging.WARNING, logger=extract.log.name):
        got = extract._read_snapshot_frame(xlsx)
    assert len(got) == 5                      # served from the workbook
    assert any("Could not read Parquet sidecar" in r.getMessage() for r in caplog.records)


def test_read_snapshot_frame_falls_back_when_sidecar_absent(tmp_path):
    xlsx = str(tmp_path / "snap.xlsx")
    extract.write_powerbi_xlsx(extract.select_and_rename(sql_shaped_frame()), xlsx)
    assert len(extract._read_snapshot_frame(xlsx)) == 5


def test_load_previous_snapshot_reads_via_the_sidecar(tmp_path):
    """The public contract still holds when the fast path is taken."""
    xlsx, _ = _write_pair(tmp_path)
    got = extract.load_previous_snapshot(xlsx)
    assert got is not None
    assert pd.api.types.is_datetime64_any_dtype(got["WeekDate"])
    for col in extract.NUMERIC_COLUMNS:
        assert got[col].dtype == "float64"


def test_load_previous_snapshot_still_rejects_a_bad_sidecar_frame(tmp_path):
    """A sidecar missing required columns must return None (-> full pull), not
    a half-built snapshot."""
    xlsx = str(tmp_path / "snap.xlsx")
    extract.write_powerbi_xlsx(pd.DataFrame({"nope": [1]}), xlsx)
    pd.DataFrame({"nope": [1]}).to_parquet(extract.parquet_sidecar_path(xlsx), index=False)
    os.utime(xlsx, (1000, 1000))
    os.utime(extract.parquet_sidecar_path(xlsx), (2000, 2000))
    assert extract.load_previous_snapshot(xlsx) is None


def test_sidecar_roundtrip_is_faster_than_the_workbook(tmp_path):
    """Sanity check on the whole point of the sidecar. Not a benchmark — just
    proof the fast path is in fact the fast one on identical content."""
    import time

    xlsx, sidecar = _write_pair(tmp_path)
    t0 = time.perf_counter(); pd.read_parquet(sidecar); pq = time.perf_counter() - t0
    t0 = time.perf_counter(); pd.read_excel(xlsx, header=2); xl = time.perf_counter() - t0
    assert pq < xl, f"parquet {pq:.4f}s was not faster than xlsx {xl:.4f}s"
