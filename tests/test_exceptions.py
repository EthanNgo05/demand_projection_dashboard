"""Unit tests for the Exceptions view's pure compute layer.

Builds a tiny hand-controlled cleaned frame with known POS/Projection so the
gap / % / $ derivations and the edge cases (no plan, no recent sales, trivial
swing, no gap, no list price) can be asserted exactly. Uses the real default
pipeline for the model-agnostic helpers (aggregate_to_sku_week / week_anchors /
region_for_group).
"""
import numpy as np
import pandas as pd
import pytest

from dashboard_app.config import DEFAULT_MODEL, MODEL_OPTIONS, PRICE_COL
from dashboard_app.pipeline import load_pipeline
from dashboard_app.exceptions import (
    DIRECTION_COL, FLAG_COL, GAP_COL, IMPACT_COL, OVER, PCT_COL, PROJ_COL,
    RECENT_COL, UNDER, _apply_thresholds, compute_exceptions,
)

TODAY = pd.Timestamp("2026-07-22")          # Wednesday
# week_anchors(TODAY): current week starts Sun 2026-07-19, last complete week
# 2026-07-12, so the 8-week window is 2026-05-24 .. 2026-07-12 and the 15 forward
# weeks start 2026-07-19.
HIST_WEEKS = pd.date_range("2026-05-24", periods=8, freq="W-SUN")
FWD_WEEKS = pd.date_range("2026-07-19", periods=15, freq="W-SUN")
GROUP = "AMAZON-DC"


@pytest.fixture(scope="module")
def P():
    return load_pipeline(MODEL_OPTIONS[DEFAULT_MODEL])


def _rows(sku, recent=None, proj=None):
    """Weekly cleaned-frame rows for one SKU: a flat POS run-rate over the 8-week
    window (``recent``) and/or a flat system Projection over the 15 forward weeks
    (``proj``). Passing None omits that side entirely."""
    rows = []
    if recent is not None:
        for wk in HIST_WEEKS:
            rows.append({"SKU": sku, "Description": f"Widget {sku}", "Customer": "AMZ",
                         "WeekDate": wk, "POS": float(recent), "Orders": np.nan,
                         "Projection": np.nan, "Customer Grouping": GROUP})
    if proj is not None:
        for wk in FWD_WEEKS:
            rows.append({"SKU": sku, "Description": f"Widget {sku}", "Customer": "AMZ",
                         "WeekDate": wk, "POS": np.nan, "Orders": np.nan,
                         "Projection": float(proj), "Customer Grouping": GROUP})
    return rows


@pytest.fixture
def sample_df():
    rows = []
    rows += _rows("UNDER-CLEAR", recent=100, proj=20)   # gap +80, +400%
    rows += _rows("OVER-CLEAR", recent=20, proj=100)    # gap -80, -80%
    rows += _rows("NOPLAN", recent=50, proj=None)       # selling, no plan of record
    rows += _rows("DEADPLAN", recent=None, proj=100)    # planned, no recent sales
    rows += _rows("TRIVIAL", recent=100, proj=98)       # gap +2, ~+2% (noise)
    rows += _rows("NOGAP", recent=50, proj=50)          # exactly on plan
    rows += _rows("NOPRICE", recent=100, proj=20)       # big swing, no list price
    return pd.DataFrame(rows)


PRICES = {"UNDER-CLEAR": 10.0, "OVER-CLEAR": 10.0, "NOPLAN": 10.0,
          "DEADPLAN": 10.0, "TRIVIAL": 10.0, "NOGAP": 10.0}  # NOPRICE deliberately absent


def _by_sku(frame):
    return frame.set_index("SKU")


def test_flags_the_right_skus(sample_df, P):
    out = compute_exceptions(sample_df, TODAY, PRICES, P)
    # NOGAP (recent == proj) is dropped; everything else with a gap is flagged.
    assert set(out["SKU"]) == {
        "UNDER-CLEAR", "OVER-CLEAR", "NOPLAN", "DEADPLAN", "TRIVIAL", "NOPRICE"
    }


def test_gap_pct_and_impact(sample_df, P):
    out = _by_sku(compute_exceptions(sample_df, TODAY, PRICES, P))

    assert out.loc["UNDER-CLEAR", RECENT_COL] == 100.0
    assert out.loc["UNDER-CLEAR", PROJ_COL] == 20
    assert out.loc["UNDER-CLEAR", GAP_COL] == 80
    assert out.loc["UNDER-CLEAR", PCT_COL] == 400.0
    assert out.loc["UNDER-CLEAR", IMPACT_COL] == 800.0
    assert out.loc["UNDER-CLEAR", DIRECTION_COL] == UNDER

    assert out.loc["OVER-CLEAR", GAP_COL] == -80
    assert out.loc["OVER-CLEAR", PCT_COL] == -80.0
    assert out.loc["OVER-CLEAR", IMPACT_COL] == -800.0
    assert out.loc["OVER-CLEAR", DIRECTION_COL] == OVER


def test_edge_cases(sample_df, P):
    out = _by_sku(compute_exceptions(sample_df, TODAY, PRICES, P))

    # No plan of record: proj treated as 0, % undefined, still under-projected.
    assert out.loc["NOPLAN", FLAG_COL] == "no plan"
    assert out.loc["NOPLAN", PROJ_COL] == 0
    assert pd.isna(out.loc["NOPLAN", PCT_COL])
    assert out.loc["NOPLAN", DIRECTION_COL] == UNDER

    # Planned but nothing selling recently: recent 0, a full -100% over-projection.
    assert out.loc["DEADPLAN", FLAG_COL] == "no recent sales"
    assert out.loc["DEADPLAN", RECENT_COL] == 0.0
    assert out.loc["DEADPLAN", PCT_COL] == -100.0
    assert out.loc["DEADPLAN", DIRECTION_COL] == OVER

    # No list price: impact is unknown (NaN), not zero.
    assert pd.isna(out.loc["NOPRICE", IMPACT_COL])


def test_region_and_data_source(sample_df, P):
    out = _by_sku(compute_exceptions(sample_df, TODAY, PRICES, P))
    assert out.loc["UNDER-CLEAR", "Region"] == str(P.region_for_group(GROUP))
    assert out.loc["UNDER-CLEAR", "Data Source"] == "POS"


def test_pct_threshold_filters_trivial(sample_df, P):
    out = compute_exceptions(sample_df, TODAY, PRICES, P)
    kept = _apply_thresholds(out, min_pct=0.5, min_dollar=0)
    # TRIVIAL (~2%) drops; the no-plan row (NaN %) is kept as inherently extreme.
    assert "TRIVIAL" not in set(kept["SKU"])
    assert "NOPLAN" in set(kept["SKU"])


def test_dollar_threshold(sample_df, P):
    out = compute_exceptions(sample_df, TODAY, PRICES, P)
    kept = set(_apply_thresholds(out, min_pct=0.5, min_dollar=900)["SKU"])
    # Only DEADPLAN's |impact| ($1000) clears $900; NOPRICE (unknown $) is kept too.
    assert kept == {"DEADPLAN", "NOPRICE"}


def test_empty_frame_returns_empty(P):
    empty = compute_exceptions(pd.DataFrame(), TODAY, PRICES, P)
    assert empty.empty
