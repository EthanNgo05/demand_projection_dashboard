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

from dashboard_app.config import DEFAULT_MODEL, MODEL_OPTIONS, PRICE_COL
from dashboard_app.pipeline import load_pipeline
from dashboard_app.exceptions import (
    CONTAINER_IMPACT_COL, DIRECTION_COL, FLAG_COL, GAP_COL, GROUP_CUSTOMER,
    GROUP_DETAIL, GROUP_REGION, GROUP_SKU, IMPACT_COL, OVER, PCT_COL, PROJ_COL,
    RECENT_COL, SPIKE_FIRST_WEEK_COL, SPIKE_WEEKS_SINCE_COL, STATUS_COL, UNDER,
    WEEKS_COL, WOS_COL, aggregate_exceptions, aggregate_spikes, compute_spikes,
    sku_week_by_group,
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


# --------------------------------------------------------------------------- #
# "Group by" roll-up: aggregate_exceptions / aggregate_spikes                  #
# --------------------------------------------------------------------------- #
def _exc_frame():
    """A hand-built compute_exceptions-style frame: 2 SKUs × 2 customers in 2
    regions (each customer in one region), with known recent/projection/price so
    the roll-up sums and re-derivations are exact."""
    def row(sku, cust, region, recent, proj, price):
        gap = recent - proj
        return {
            "SKU": sku, "Description": f"Widget {sku}", "Customer Grouping": cust,
            "Region": region, "Data Source": "Orders",
            RECENT_COL: recent, PROJ_COL: proj, WEEKS_COL: 8, GAP_COL: gap,
            PCT_COL: (gap / proj * 100) if proj else np.nan,
            IMPACT_COL: float(gap * price), FLAG_COL: "", PRICE_COL: float(price),
            STATUS_COL: "x", DIRECTION_COL: "x", "_sort": abs(gap * price),
        }
    df = pd.DataFrame([
        row("SKUA", "CUST1", "US", 100, 50, 2.0),   # under
        row("SKUA", "CUST2", "EU", 10, 40, 2.0),    # over
        row("SKUB", "CUST1", "US", 0, 20, 3.0),     # over ("No recent sales")
        row("SKUB", "CUST2", "EU", 5, 0, 3.0),      # under ("No forecasts given")
    ])
    for c in (RECENT_COL, PROJ_COL, WEEKS_COL, GAP_COL):
        df[c] = df[c].astype("Int64")
    return df


def test_aggregate_exceptions_detail_is_identity():
    frame = _exc_frame()
    assert aggregate_exceptions(frame, GROUP_DETAIL) is frame


def test_aggregate_exceptions_by_sku_sums_and_rederives():
    out = aggregate_exceptions(_exc_frame(), GROUP_SKU).set_index("SKU")
    assert len(out) == 2
    a = out.loc["SKUA"]
    # recent 100+10=110, proj 50+40=90 → gap 20 (under); impact 50·2 + (−30·2)=40.
    assert int(a[RECENT_COL]) == 110 and int(a[PROJ_COL]) == 90
    assert int(a[GAP_COL]) == 20 and a[DIRECTION_COL] == UNDER
    assert a[IMPACT_COL] == pytest.approx(40.0)
    assert a[PCT_COL] == pytest.approx(round(20 / 90 * 100, 2))
    assert a[PRICE_COL] == pytest.approx(2.0)          # price kept at the SKU grain
    assert a["Region"] == "EU, US"                     # sorted comma-separated regions
    assert a["Customer Grouping"] == "2 customers"
    b = out.loc["SKUB"]
    # recent 0+5=5, proj 20+0=20 → gap −15 (over); impact (−20·3)+(5·3)=−45.
    assert int(b[GAP_COL]) == -15 and b[DIRECTION_COL] == OVER
    assert b[IMPACT_COL] == pytest.approx(-45.0)


def test_aggregate_exceptions_by_customer_and_region():
    for grain, key in ((GROUP_CUSTOMER, "Customer Grouping"), (GROUP_REGION, "Region")):
        out = aggregate_exceptions(_exc_frame(), grain).set_index(key)
        assert len(out) == 2
        # CUST1/US: recent 100+0=100, proj 50+20=70 → gap 30 (under); impact 100−60=40.
        one = out.loc["CUST1" if grain == GROUP_CUSTOMER else "US"]
        assert int(one[RECENT_COL]) == 100 and int(one[PROJ_COL]) == 70
        assert int(one[GAP_COL]) == 30 and one[DIRECTION_COL] == UNDER
        assert one[IMPACT_COL] == pytest.approx(40.0)
        assert pd.isna(one[PRICE_COL])                 # no blended price off the SKU grain


def _spike_frame():
    """A hand-built compute_spikes-style frame: SKUX spiking at two customers in
    two regions (SKU-level Container Impact 3.0 / WOS 2.0) + SKUY at one customer."""
    def row(sku, cust, region, recent, first_wk, weeks_since, ci, wos):
        return {
            "SKU": sku, "Description": f"Widget {sku}", "Region": region,
            "Region Code": region, "Active in": "US", "Customer Grouping": cust,
            "Data Source": "Orders",
            SPIKE_FIRST_WEEK_COL: pd.Timestamp(first_wk).date(),
            SPIKE_WEEKS_SINCE_COL: weeks_since, RECENT_COL: float(recent),
            PROJ_COL: 0, PRICE_COL: 10.0, CONTAINER_IMPACT_COL: ci, WOS_COL: wos,
            "_sort": ci,
        }
    return pd.DataFrame([
        row("SKUX", "CUST1", "US", 30, "2026-06-28", 3, 3.0, 2.0),
        row("SKUX", "CUST2", "EU", 60, "2026-06-21", 4, 3.0, 2.0),
        row("SKUY", "CUST1", "US", 10, "2026-07-05", 2, 1.0, 0.5),
    ])


def test_aggregate_spikes_by_sku():
    out = aggregate_spikes(_spike_frame(), GROUP_SKU).set_index("SKU")
    x = out.loc["SKUX"]
    assert x[RECENT_COL] == pytest.approx(90.0)                 # 30 + 60
    assert x[CONTAINER_IMPACT_COL] == pytest.approx(3.0)        # SKU-level, one SKU
    assert x[WOS_COL] == pytest.approx(2.0)                     # kept at SKU grain
    assert pd.Timestamp(x[SPIKE_FIRST_WEEK_COL]) == pd.Timestamp("2026-06-21")  # min onset
    assert int(x[SPIKE_WEEKS_SINCE_COL]) == 4                   # max weeks-since


def test_aggregate_spikes_by_customer_sums_distinct_sku_container_impact():
    out = aggregate_spikes(_spike_frame(), GROUP_CUSTOMER).set_index("Customer Grouping")
    # CUST1: distinct SKUs SKUX(3.0)+SKUY(1.0)=4.0; WOS blanked off the SKU grain.
    assert out.loc["CUST1", CONTAINER_IMPACT_COL] == pytest.approx(4.0)
    assert pd.isna(out.loc["CUST1", WOS_COL])
    # CUST2: only SKUX → 3.0.
    assert out.loc["CUST2", CONTAINER_IMPACT_COL] == pytest.approx(3.0)


def test_aggregate_spikes_by_region():
    out = aggregate_spikes(_spike_frame(), GROUP_REGION).set_index("Region")
    assert out.loc["US", CONTAINER_IMPACT_COL] == pytest.approx(4.0)   # SKUX + SKUY
    assert out.loc["EU", CONTAINER_IMPACT_COL] == pytest.approx(3.0)
    assert out.loc["US", RECENT_COL] == pytest.approx(40.0)            # 30 + 10


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
