"""Phase 2 node-level unit tests — see docs/agentic_workflow/02-deterministic-pipeline-nodes.md.

Run with: pytest tests/test_phase2_pipeline.py -v
"""

import pandas as pd

from agent.config import ALL_CUSTOMERS_VIEW, MODEL_OPTIONS
from agent.nodes.forecast import run_all_models
from agent.nodes.ingest import ingest

TODAY = pd.Timestamp("2026-07-01")


def test_ingest_no_files_returns_error(monkeypatch):
    monkeypatch.setattr("agent.data_io.discover_raw_files", lambda: [])
    result = ingest({})
    assert result["errors"], "expected an error when no raw files are found"


def test_ingest_missing_price_file_is_non_fatal(monkeypatch, sample_raw_path):
    # With no price source at all — the Plytix feed disabled AND no local file —
    # ingest must not raise: prices come back None and downstream nodes simply
    # skip the list-price columns. Disabling the feed also keeps this unit test
    # hermetic (no network call to the default feed URL).
    monkeypatch.setattr(
        "agent.data_io.discover_raw_files", lambda: [("2026-07-01", sample_raw_path)]
    )
    monkeypatch.setattr("agent.data_io.PLYTIX_FEED_URL", "")
    monkeypatch.setattr("agent.data_io.discover_price_file", lambda: None)
    result = ingest({})
    assert result["prices"] is None
    assert not result.get("errors")
    assert not result["cleaned_df"].empty


def test_ingest_honours_pinned_raw_path(sample_raw_path):
    # parity tests pin the input file in the initial state; discovery is skipped
    result = ingest({"raw_path": sample_raw_path, "price_path": None})
    assert result["raw_path"] == sample_raw_path
    assert result["prices"] is None
    assert not result["cleaned_df"].empty


def test_ingest_filters_ignored_customers(sample_cleaned_df):
    # Others - UK is in CUSTOMERS_TO_IGNORE and must be dropped by _clean
    assert "Others - UK" not in set(sample_cleaned_df["CUSTNMBR"])
    # grouping fold: AMAZON-DS rows map into the AMAZON-DC customer group
    ds = sample_cleaned_df[sample_cleaned_df["CUSTNMBR"] == "AMAZON-DS"]
    assert (ds["Customer Grouping"] == "AMAZON-DC").all()


def test_clean_strips_sku_whitespace():
    """The fixed-width warehouse export space-pads every SKU (e.g.
    'BT1028                    '). _clean must strip that surrounding whitespace
    so the SKU matches the (stripped) list-price index and Plytix SKU sets —
    otherwise revenue risk is blank and the active-in/discontinued checks never
    fire. Regression test for the "revenue risk left blank" bug.
    """
    from agent import data_io
    from agent.model_loader import load_pipeline

    P = load_pipeline(next(iter(MODEL_OPTIONS.values())))
    raw = pd.DataFrame(
        [["BT1028                         ", "Padded widget", "AMAZON-DS",
          pd.Timestamp("2026-06-07"), 10, 12, 11]],
        columns=["'Demand'[DisplaySKU]", "Description", "Custnmbr", "WeekDate",
                 "POS", "Sum of Quantity", "Projection"],
    )
    cleaned = data_io._clean(raw, P)
    assert cleaned["SKU"].tolist() == ["BT1028"]


def test_clean_strips_custnmbr_whitespace_so_groups_fold():
    """The export also space-pads CUSTNMBR (e.g. 'AMAZON-DS      '). Left padded,
    it misses COMBINED_GROUPING, so AMAZON-DS never folds into the AMAZON-DC
    group and the group fragments across padded/clean spellings. _clean must
    strip CUSTNMBR before the ignore filter and the grouping map.
    """
    from agent import data_io
    from agent.model_loader import load_pipeline

    P = load_pipeline(next(iter(MODEL_OPTIONS.values())))
    raw = pd.DataFrame(
        [["SKU-001", "Widget", "AMAZON-DS      ",
          pd.Timestamp("2026-06-07"), 10, 12, 11]],
        columns=["'Demand'[DisplaySKU]", "Description", "Custnmbr", "WeekDate",
                 "POS", "Sum of Quantity", "Projection"],
    )
    cleaned = data_io._clean(raw, P)
    assert cleaned["CUSTNMBR"].tolist() == ["AMAZON-DS"]
    assert (cleaned["Customer Grouping"] == "AMAZON-DC").all()


def test_run_all_models_produces_all_labels(sample_cleaned_df):
    state = {
        "cleaned_df": sample_cleaned_df,
        "view": ALL_CUSTOMERS_VIEW,
        "today_ts": TODAY,
        "prices": None,
        "errors": [],
    }
    out = run_all_models(state)
    assert not out["errors"], out["errors"]
    assert set(out["results"].keys()) == set(MODEL_OPTIONS.keys())
    for label, r in out["results"].items():
        assert not r["summary_df"].empty, f"{label} produced an empty summary"
        assert not r["weekly_df"].empty, f"{label} produced an empty weekly frame"


def test_run_all_models_individual_group(sample_cleaned_df):
    state = {
        "cleaned_df": sample_cleaned_df,
        "view": "AMAZON-DC",
        "today_ts": TODAY,
        "prices": None,
        "errors": [],
    }
    out = run_all_models(state)
    assert not out["errors"], out["errors"]
    assert set(out["results"].keys()) == set(MODEL_OPTIONS.keys())
    for label, r in out["results"].items():
        assert not r["summary_df"].empty, f"{label} produced an empty summary"
