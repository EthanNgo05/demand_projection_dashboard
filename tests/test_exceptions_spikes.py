"""Unit tests for the Exceptions view's "Recent spikes in POS/Orders with no
projections" table (``compute_spikes``).

Builds a tiny hand-controlled cleaned frame with known weekly Orders and system
Projection, runs it through the real ``sku_week_by_group`` aggregation (as the
dashboard does), and asserts the spike derivations exactly: which SKUs surface,
the onset week / weeks-since, and Revenue Risk = list price × cumulative spike
units. Uses the real default pipeline for the model-agnostic helpers.
"""
import numpy as np
import pandas as pd
import pytest

from dashboard_app.config import DEFAULT_MODEL, MODEL_OPTIONS
from dashboard_app.pipeline import load_pipeline
from dashboard_app.exceptions import (
    CONTAINER_IMPACT_COL, RECENT_COL, SPIKE_FIRST_WEEK_COL,
    SPIKE_WEEKS_SINCE_COL, WOS_COL, compute_spikes, sku_week_by_group,
)

TODAY = pd.Timestamp("2026-07-22")          # Wednesday
# week_anchors(TODAY): current week starts Sun 2026-07-19, last complete week
# 2026-07-12, so the 8-week window is 2026-05-24 .. 2026-07-12.
HIST_WEEKS = pd.date_range("2026-05-24", periods=8, freq="W-SUN")
FWD_WEEKS = pd.date_range("2026-07-19", periods=15, freq="W-SUN")
GROUP = "AMAZON-DC"                          # region "US (LBC+NJ)" → code "US"


@pytest.fixture(scope="module")
def P():
    return load_pipeline(MODEL_OPTIONS[DEFAULT_MODEL])


def _rows(sku, orders_by_week, proj=None, group=GROUP):
    """Weekly cleaned-frame rows for one SKU: an explicit per-week Orders series
    over the 8-week window, and optionally a flat system Projection over the 15
    forward weeks (``proj=None`` omits the projection side entirely). ``group``
    sets the Customer Grouping so a SKU can span multiple customers."""
    rows = []
    for wk, v in zip(HIST_WEEKS, orders_by_week):
        rows.append({"SKU": sku, "Description": f"Widget {sku}", "Customer": group,
                     "WeekDate": wk, "POS": np.nan, "Orders": float(v),
                     "Projection": np.nan, "Customer Grouping": group})
    if proj is not None:
        for wk in FWD_WEEKS:
            rows.append({"SKU": sku, "Description": f"Widget {sku}", "Customer": group,
                         "WeekDate": wk, "POS": np.nan, "Orders": np.nan,
                         "Projection": float(proj), "Customer Grouping": group})
    return rows


# Orders ramp: dead for 5 weeks, then 10/wk for the last 3 (weeks 5..7 =
# 2026-06-28, 07-05, 07-12). First selling week (spike onset) → 2026-06-28.
SPIKE_ORDERS = [0, 0, 0, 0, 0, 10, 10, 10]
FIRST_SPIKE_WK = pd.Timestamp("2026-06-28")


@pytest.fixture
def sample_df():
    rows = []
    rows += _rows("SPIKE", SPIKE_ORDERS, proj=None)          # 0 plan, selling → flagged
    rows += _rows("SPIKE-ZERO", SPIKE_ORDERS, proj=0)        # explicit 0 projection → flagged
    rows += _rows("HAS-PROJ", SPIKE_ORDERS, proj=50)         # real plan → excluded
    rows += _rows("SMALL", [0, 0, 0, 0, 0, 2, 2, 2], proj=0)  # sells a little → still flagged
    rows += _rows("NOSALES", [0] * 8, proj=0)                # 0 plan, no sales → excluded
    return pd.DataFrame(rows)


PRICES = {"SPIKE": 10.0, "SPIKE-ZERO": 10.0, "HAS-PROJ": 10.0, "SMALL": 10.0}
ACTIVE_IN = {"SPIKE": "US,CA,EU"}


def _agg(df, P):
    return sku_week_by_group(df, P)


def _by_sku(frame):
    return frame.set_index("SKU")


def test_flags_zero_projection_sellers(sample_df, P):
    out = compute_spikes(_agg(sample_df, P), TODAY, PRICES, P, ACTIVE_IN)
    # Every 0/absent-projection SKU with ANY recent sales surfaces (the container
    # threshold defaults off); the SKU with a real forward plan and the one with no
    # sales do not.
    assert set(out["SKU"]) == {"SPIKE", "SPIKE-ZERO", "SMALL"}


def test_onset_week_and_weeks_since(sample_df, P):
    out = _by_sku(compute_spikes(_agg(sample_df, P), TODAY, PRICES, P, ACTIVE_IN))
    row = out.loc["SPIKE"]
    assert pd.Timestamp(row[SPIKE_FIRST_WEEK_COL]) == FIRST_SPIKE_WK
    # current week 2026-07-19 − first spike 2026-06-28 = 21 days = 3 weeks.
    assert int(row[SPIKE_WEEKS_SINCE_COL]) == 3
    assert row["Data Source"] == "Orders"
    assert row["Region Code"] == "US"
    assert row["Active in"] == "US,CA,EU"


def test_output_carries_description_for_card_title(sample_df, P):
    out = _by_sku(compute_spikes(_agg(sample_df, P), TODAY, PRICES, P, ACTIVE_IN))
    assert "Description" in out.columns
    assert out.loc["SPIKE", "Description"] == "Widget SPIKE"


def test_empty_input_returns_empty_frame(P):
    out = compute_spikes(pd.DataFrame(), TODAY, PRICES, P)
    assert out.empty
    assert CONTAINER_IMPACT_COL in out.columns


# --------------------------------------------------------------------------- #
# Container Impact + WOS Impact (both SKU-level: constant across a SKU's rows)  #
# --------------------------------------------------------------------------- #
GROUP2 = "COSTCO"       # a second customer grouping for the same SKU
GROUP3 = "TARGET-DS"    # a third — carries the SKU's real projection


@pytest.fixture
def multi_df():
    """A SKU spiking (0 projection) at two customer groups, plus a third group that
    carries a real projection (contributing to the SKU's total projection but not a
    spike row), so WOS has a nonzero denominator."""
    rows = []
    # MULTI: spikes at two customers (30 and 60 cumulative units), projection 85 at
    # a third → total spike units 90, total projection 85.
    rows += _rows("MULTI", SPIKE_ORDERS, proj=0, group=GROUP)                  # 30 units
    rows += _rows("MULTI", [0, 0, 0, 0, 0, 20, 20, 20], proj=0, group=GROUP2)  # 60 units
    rows += _rows("MULTI", [0] * 8, proj=85, group=GROUP3)                     # plan only
    # LT1009: spiking with 0 On Hand and total projection 85 → WOS 0.
    rows += _rows("LT1009", SPIKE_ORDERS, proj=0, group=GROUP)
    rows += _rows("LT1009", [0] * 8, proj=85, group=GROUP3)
    return pd.DataFrame(rows)


CONTAINER_LOAD = {"MULTI": 30.0, "LT1009": 393.0}
ONHAND = {"MULTI": 170.0, "LT1009": 0.0}


def test_container_impact_is_sku_total_spike_units_over_load(multi_df, P):
    out = compute_spikes(_agg(multi_df, P), TODAY, PRICES, P, container_load=CONTAINER_LOAD,
                         onhand_by_sku=ONHAND)
    multi = out[out["SKU"] == "MULTI"]
    # Two flagged rows (GROUP + GROUP2); both show the SAME SKU-level value.
    assert set(multi["Customer Grouping"]) == {GROUP, GROUP2}
    # (30 + 60) units / 30 per container = 3.0 containers.
    assert multi[CONTAINER_IMPACT_COL].tolist() == [pytest.approx(3.0)] * len(multi)


def test_wos_is_sku_total_onhand_over_total_projection(multi_df, P):
    out = compute_spikes(_agg(multi_df, P), TODAY, PRICES, P, container_load=CONTAINER_LOAD,
                         onhand_by_sku=ONHAND)
    by_sku = out.set_index(["SKU", "Customer Grouping"])
    # MULTI: On Hand 170 / total projection 85 = 2.0, identical on both rows.
    assert by_sku.loc[("MULTI", GROUP), WOS_COL] == pytest.approx(2.0)
    assert by_sku.loc[("MULTI", GROUP2), WOS_COL] == pytest.approx(2.0)
    # LT1009 (the user's example): On Hand 0 / 85 = 0.
    assert by_sku.loc[("LT1009", GROUP), WOS_COL] == pytest.approx(0.0)


def test_container_and_wos_blank_without_inputs(multi_df, P):
    # No container-load / on-hand maps → both columns present but all NaN.
    out = compute_spikes(_agg(multi_df, P), TODAY, PRICES, P)
    assert out[CONTAINER_IMPACT_COL].isna().all()
    assert out[WOS_COL].isna().all()


def test_wos_blank_when_total_projection_zero(P):
    # A SKU flagged with 0 projection everywhere → total projection 0 → WOS blank
    # (never a divide-by-zero), even though On Hand is known.
    df = pd.DataFrame(_rows("ZEROPROJ", SPIKE_ORDERS, proj=0, group=GROUP))
    out = compute_spikes(_agg(df, P), TODAY, PRICES, P, container_load={"ZEROPROJ": 10.0},
                         onhand_by_sku={"ZEROPROJ": 500.0})
    assert out[WOS_COL].isna().all()
    # Container Impact still computes (30 units / 10 = 3.0).
    assert out[CONTAINER_IMPACT_COL].tolist() == [pytest.approx(3.0)]


def test_min_container_impact_filters_by_sku_level_value(multi_df, P):
    # MULTI's container impact is 3.0 (90 units / 30 load); LT1009's is ~0.08.
    agg = _agg(multi_df, P)
    lo = compute_spikes(agg, TODAY, PRICES, P, container_load=CONTAINER_LOAD,
                        onhand_by_sku=ONHAND, min_container_impact=2.0)
    assert "MULTI" in set(lo["SKU"]) and "LT1009" not in set(lo["SKU"])
    hi = compute_spikes(agg, TODAY, PRICES, P, container_load=CONTAINER_LOAD,
                        onhand_by_sku=ONHAND, min_container_impact=5.0)
    assert hi.empty


def test_min_container_impact_keeps_unknown_load_skus(P):
    # A SKU with sales + 0 projection but no Container Load → impact unknown (NaN);
    # a nonzero threshold must NOT hide it (unknown ≠ below-threshold).
    df = pd.DataFrame(_rows("NOCL", SPIKE_ORDERS, proj=0, group=GROUP))
    out = compute_spikes(_agg(df, P), TODAY, PRICES, P, container_load={},
                         min_container_impact=99.0)
    assert "NOCL" in set(out["SKU"])
    assert pd.isna(out.set_index("SKU").loc["NOCL", CONTAINER_IMPACT_COL])


# --------------------------------------------------------------------------- #
# data_io helpers                                                              #
# --------------------------------------------------------------------------- #
def test_container_load_from_plytix():
    from agent.data_io import container_load_from_plytix
    plytix = pd.DataFrame({"SKU": ["ST2030", "ZERO*", "NOLOAD"],
                           "Container Load": [393, 100, np.nan]})
    cl = container_load_from_plytix(plytix)
    assert cl["ST2030"] == 393
    assert cl["ZERO"] == 100          # trailing '*' stripped to match demand SKUs
    assert "NOLOAD" not in cl.index   # blank load dropped
    assert container_load_from_plytix(pd.DataFrame({"SKU": ["X"]})) is None


def test_onhand_by_sku_takes_latest_week_and_sums_customers():
    from agent.data_io import onhand_by_sku
    raw = pd.DataFrame({
        "'Demand'[DisplaySKU]": ["AA", "AA", "AA", "BB"],
        "Custnmbr": ["C1", "C1", "C2", "C1"],
        "WeekDate": pd.to_datetime(["2026-06-28", "2026-07-05", "2026-07-05", "2026-07-05"]),
        "On Hand": [10, 40, 5, np.nan],   # C1's latest (07-05) = 40; C2 = 5
    })
    oh = onhand_by_sku(raw)
    assert oh["AA"] == 45              # 40 (C1 latest) + 5 (C2)
    assert "BB" not in oh.index        # only NaN On Hand → dropped
    assert onhand_by_sku(pd.DataFrame({"x": [1]})) is None
