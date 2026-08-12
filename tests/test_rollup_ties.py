"""A view total must equal the sum of its SKU x customer parts.

Forecasts are made per (SKU, Customer Grouping). Every view-level figure the UI
shows — the KPI row, the total-demand chart, the SKU-detail tiles, the by-SKU
table — is therefore the SUM of those rows and must tie to them exactly.

Quick Projections used to fit a SECOND, separate series per SKU from the
customer-summed history and show that as the total, which on the live snapshot ran
+19.9% away from the table underneath it. 90% of that gap was demand the combined
fit could not see at all: POS-vs-Orders is chosen per series, so a SKU with POS at
one customer picked POS for the whole aggregate and silently dropped every
customer that only reports Orders. These tests pin the invariant that replaced it.

Fast and synthetic — no fixture workbook, no model fit — so the arithmetic is
checked directly and a regression is obvious rather than buried in a live total.
"""

import glob
import os

import numpy as np
import pandas as pd
import pytest

from dashboard_app.compute import (
    attach_current_projection, roll_up_summary, roll_up_to_sku_week,
    sku_grain_demand_frame,
)
from dashboard_app.config import MIXED_SOURCE, PRICE_COL, RISK_COL
from dashboard_app.summaries import historical_window, resolve_demand

VIEW = "All customers (combined)"
TODAY = pd.Timestamp("2026-08-11")
# Eight completed weeks ending 2026-08-02, then 15 forecast weeks — the anchors the
# models use, hard-coded so the test doesn't depend on a pipeline import.
LCW = pd.Timestamp("2026-08-02")
LB = LCW - pd.Timedelta(weeks=7)
FFW = pd.Timestamp("2026-08-09")
ANCHORS = (LB, LCW, FFW)
HORIZON = pd.date_range(FFW, periods=15, freq="W-SUN")


def _agg_by_group(rows):
    """Per-(SKU, group) SKU-week frame from ``(sku, group, week, pos, orders, proj)``."""
    return pd.DataFrame(
        rows, columns=["SKU", "Customer Grouping", "WeekDate", "POS", "Orders",
                       "Projection"],
    ).astype({"WeekDate": "datetime64[ns]"})


def _by_cust(rows):
    """Per-(SKU, group) summary from ``(sku, group, source, updated, price)``."""
    df = pd.DataFrame(
        rows, columns=["SKU", "Customer Grouping", "Data Source",
                       "Updated Projection Average", PRICE_COL],
    )
    # The columns the roll-up copies through or overwrites, in SUMMARY_COLUMNS order.
    df.insert(1, "Description", "Widget")
    df["Weeks with data"] = 8
    df["Current Projection Average"] = 0
    df["Projection Difference"] = 0
    df[RISK_COL] = np.nan
    return df


# --------------------------------------------------------------------------- #
# resolve_demand: each customer contributes ITS OWN signal                     #
# --------------------------------------------------------------------------- #
def test_resolve_demand_uses_each_groups_own_source():
    """An Orders-only customer's demand must survive into the total.

    This is the bug that made 90% of the old +19.9% gap: summing POS and Orders
    into separate columns and THEN choosing one source per SKU discards every
    customer that reports no sell-through.
    """
    agg = _agg_by_group([
        ("S1", "AMAZON", LCW, 100.0, np.nan, 10.0),
        ("S1", "DISTRIB", LCW, np.nan, 40.0, 5.0),
    ])
    by_cust = _by_cust([
        ("S1", "AMAZON", "POS", 100, 1.0),
        ("S1", "DISTRIB", "Orders", 40, 1.0),
    ])
    out = resolve_demand(agg, by_cust)
    assert list(out["demand"]) == [100.0, 40.0]

    rolled = roll_up_to_sku_week(out, ["POS", "Orders", "Projection", "demand"])
    # 140, not 100: the distributor is in the total.
    assert rolled["demand"].sum() == 140.0


def test_resolve_demand_falls_back_to_pos_without_a_source_map():
    """No usable summary -> POS, the pre-existing behaviour, not a crash."""
    agg = _agg_by_group([("S1", "AMAZON", LCW, 7.0, 99.0, 1.0)])
    out = resolve_demand(agg, pd.DataFrame())
    assert list(out["demand"]) == [7.0]


def test_historical_window_prefers_a_resolved_demand_column():
    """A frame that already carries ``demand`` is not re-resolved per SKU.

    Re-resolving after the roll-up is impossible — the customers have been added
    together — so honouring the precomputed column is what keeps the actuals line
    additive for a mixed-source SKU.
    """
    agg = _agg_by_group([("S1", "AMAZON", LCW, 100.0, np.nan, 1.0)])
    agg["demand"] = 140.0                     # deliberately != POS
    # A summary that would resolve "POS" and so give 100 if it were consulted.
    out = historical_window(agg, _by_cust([("S1", "AMAZON", "POS", 1, 1.0)]), ANCHORS)
    assert list(out["demand"]) == [140.0]


# --------------------------------------------------------------------------- #
# attach_current_projection: the existing plan must be summable                #
# --------------------------------------------------------------------------- #
def test_current_projection_is_additive_across_ragged_coverage():
    """Two customers whose plans span different numbers of weeks still sum.

    The models divide by however many horizon weeks a series HAS a plan for, so a
    customer planned for 5 of 15 weeks reported a 3x-inflated average and the rows
    could not be added. A fixed 15-week denominator sums exactly.
    """
    rows = []
    for wk in HORIZON:                        # AMAZON: all 15 weeks at 15/wk
        rows.append(("S1", "AMAZON", wk, np.nan, np.nan, 15.0))
    for wk in HORIZON[:5]:                    # DISTRIB: 5 weeks at 30/wk
        rows.append(("S1", "DISTRIB", wk, np.nan, np.nan, 30.0))
    agg = _agg_by_group(rows)
    by_cust = _by_cust([
        ("S1", "AMAZON", "POS", 0, 1.0),
        ("S1", "DISTRIB", "Orders", 0, 1.0),
    ])

    fixed = attach_current_projection(
        by_cust, agg, HORIZON, ["Customer Grouping", "SKU"])
    vals = dict(zip(fixed["Customer Grouping"], fixed["Current Projection Average"]))
    assert vals["AMAZON"] == pytest.approx(15.0)      # 225 / 15
    assert vals["DISTRIB"] == pytest.approx(10.0)     # 150 / 15, NOT 30

    # And the rows add up to the SKU's own total over the same denominator.
    agg_all = roll_up_to_sku_week(agg, ["POS", "Orders", "Projection"])
    sku = attach_current_projection(
        roll_up_summary(fixed, agg_all, ANCHORS, VIEW), agg_all, HORIZON, ["SKU"])
    assert sku["Current Projection Average"].iloc[0] == pytest.approx(25.0)
    assert fixed["Current Projection Average"].sum() == pytest.approx(25.0)


def test_current_projection_reads_a_missing_plan_as_zero():
    """No forward plan is 0 planned units, not an unknown — so it still sums."""
    agg = _agg_by_group([("S1", "AMAZON", HORIZON[0], np.nan, np.nan, np.nan)])
    out = attach_current_projection(
        _by_cust([("S1", "AMAZON", "POS", 5, 1.0)]), agg, HORIZON,
        ["Customer Grouping", "SKU"])
    assert out["Current Projection Average"].iloc[0] == 0.0
    assert out["Projection Difference"].iloc[0] == 5.0


def test_current_projection_is_left_unrounded():
    """Rounding ~4,000 rows then summing != rounding the sum, and this has to tie."""
    agg = _agg_by_group(
        [("S1", "AMAZON", wk, np.nan, np.nan, 1.0) for wk in HORIZON[:1]])
    out = attach_current_projection(
        _by_cust([("S1", "AMAZON", "POS", 0, 1.0)]), agg, HORIZON,
        ["Customer Grouping", "SKU"])
    assert out["Current Projection Average"].iloc[0] == pytest.approx(1 / 15)


# --------------------------------------------------------------------------- #
# roll_up_summary: the total IS the sum of its rows                            #
# --------------------------------------------------------------------------- #
def _two_sku_frames():
    rows = []
    for wk in pd.date_range(LB, LCW, freq="W-SUN"):
        rows += [
            ("S1", "AMAZON", wk, 100.0, np.nan, np.nan),
            ("S1", "DISTRIB", wk, np.nan, 40.0, np.nan),
            ("S2", "AMAZON", wk, 7.0, np.nan, np.nan),
        ]
    for wk in HORIZON:
        rows += [
            ("S1", "AMAZON", wk, np.nan, np.nan, 90.0),
            ("S1", "DISTRIB", wk, np.nan, np.nan, 30.0),
            ("S2", "AMAZON", wk, np.nan, np.nan, 6.0),
        ]
    agg = resolve_demand(_agg_by_group(rows), _by_cust([
        ("S1", "AMAZON", "POS", 100, 2.0),
        ("S1", "DISTRIB", "Orders", 40, 2.0),
        ("S2", "AMAZON", "POS", 7, 5.0),
    ]))
    by_cust = attach_current_projection(
        _by_cust([
            ("S1", "AMAZON", "POS", 100, 2.0),
            ("S1", "DISTRIB", "Orders", 40, 2.0),
            ("S2", "AMAZON", "POS", 7, 5.0),
        ]),
        agg, HORIZON, ["Customer Grouping", "SKU"],
    )
    agg_all = roll_up_to_sku_week(
        agg, ["POS", "Orders", "Projection", "demand"])
    return by_cust, agg_all


def test_roll_up_summary_ties_to_its_parts():
    """Every unit and money column totals to the sum of the customer rows."""
    by_cust, agg_all = _two_sku_frames()
    out = roll_up_summary(by_cust, agg_all, ANCHORS, VIEW)

    assert len(out) == 2, "one row per SKU"
    for col in ("Updated Projection Average", "Current Projection Average",
                "Projection Difference", RISK_COL):
        assert out[col].sum() == pytest.approx(by_cust[col].sum()), col

    s1 = out[out["SKU"] == "S1"].iloc[0]
    assert s1["Updated Projection Average"] == 140          # 100 + 40
    assert s1["Current Projection Average"] == pytest.approx(120.0)   # 90 + 30
    assert s1["Projection Difference"] == pytest.approx(20.0)
    assert s1[RISK_COL] == pytest.approx(40.0)              # 20 units x $2


def test_roll_up_summary_labels_a_mixed_source_sku():
    """POS at one customer and Orders at another is stated, not silently picked."""
    by_cust, agg_all = _two_sku_frames()
    out = roll_up_summary(by_cust, agg_all, ANCHORS, VIEW).set_index("SKU")
    assert out.loc["S1", "Data Source"] == MIXED_SOURCE
    assert out.loc["S2", "Data Source"] == "POS"
    # The view's label replaces the per-customer one.
    assert set(out["Customer Grouping"]) == {VIEW}


def test_roll_up_summary_counts_weeks_with_data_once_per_week():
    """Weeks two customers both sold in count once, not twice."""
    by_cust, agg_all = _two_sku_frames()
    out = roll_up_summary(by_cust, agg_all, ANCHORS, VIEW).set_index("SKU")
    assert out.loc["S1", "Weeks with data"] == 8       # not 16
    assert out.loc["S2", "Weeks with data"] == 8


def test_roll_up_summary_keeps_the_by_customer_column_order():
    """Column order drives the Excel exports, so it must not drift."""
    by_cust, agg_all = _two_sku_frames()
    out = roll_up_summary(by_cust, agg_all, ANCHORS, VIEW)
    shared = [c for c in by_cust.columns if c in out.columns]
    assert list(out.columns)[:len(shared)] == shared


# --------------------------------------------------------------------------- #
# The SKU-grain demand frame the descriptive averages are computed from        #
# --------------------------------------------------------------------------- #
def test_sku_grain_demand_frame_hands_demand_over_as_pos():
    """``_descriptive_averages`` picks its own source, so the choice is neutralised.

    Handing the already-resolved demand over as POS with Orders blank is what stops
    the SKU-grain average re-resolving one source per SKU and dropping the
    Orders-only customers again.
    """
    _, agg_all = _two_sku_frames()
    frame = sku_grain_demand_frame(agg_all, VIEW)
    assert frame["Orders"].isna().all()
    assert set(frame["Customer Grouping"]) == {VIEW}
    merged = frame.merge(agg_all, on=["SKU", "WeekDate"], suffixes=("", "_src"))
    pd.testing.assert_series_equal(
        merged["POS"], merged["demand"], check_names=False)


# --------------------------------------------------------------------------- #
# The shared roll-up both views use                                            #
# --------------------------------------------------------------------------- #
def test_roll_up_to_sku_week_sums_disjoint_customers():
    agg = _agg_by_group([
        ("S1", "AMAZON", LCW, 100.0, np.nan, 10.0),
        ("S1", "DISTRIB", LCW, 25.0, 40.0, 5.0),
    ])
    out = roll_up_to_sku_week(agg, ["POS", "Orders", "Projection"])
    assert len(out) == 1
    assert out["POS"].iloc[0] == 125.0
    assert out["Orders"].iloc[0] == 40.0
    assert out["Projection"].iloc[0] == 15.0
    assert "Customer Grouping" not in out.columns


def test_roll_up_to_sku_week_keeps_an_all_null_cell_null():
    """min_count=1: absent is not a real zero, which the span logic depends on."""
    agg = _agg_by_group([
        ("S1", "AMAZON", LCW, np.nan, np.nan, np.nan),
        ("S1", "DISTRIB", LCW, np.nan, np.nan, np.nan),
    ])
    out = roll_up_to_sku_week(agg, ["POS", "Orders", "Projection"])
    assert out["POS"].isna().all()
    # min_count=0 is the opt-out for a column that is never null.
    assert roll_up_to_sku_week(agg, ["POS"], min_count=0)["POS"].iloc[0] == 0.0


def test_roll_up_helpers_no_op_on_empty_input():
    """Guards, so an empty view renders its "nothing to forecast" message."""
    assert roll_up_summary(None, None, ANCHORS, VIEW) is None
    empty = pd.DataFrame()
    assert roll_up_summary(empty, None, ANCHORS, VIEW) is empty
    assert attach_current_projection(empty, empty, HORIZON, ["SKU"]) is empty


# --------------------------------------------------------------------------- #
# End to end on the real snapshot                                              #
# --------------------------------------------------------------------------- #
HAS_RAW = bool(glob.glob(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "raw_inputs", "demand_projections", "*.xlsx",
)))


@pytest.mark.skipif(not HAS_RAW, reason="no raw_inputs workbook")
@pytest.mark.slow
def test_combined_view_ties_out_on_the_live_snapshot():
    """The real thing: on the live snapshot, the view total IS the sum of its rows.

    Runs exactly the helper sequence ``dashboard.main()`` runs for the combined view
    and asserts every unit and money column ties. The synthetic tests above pin the
    arithmetic; this pins that the render path actually wires it up — 108 customer
    groups, 566 SKUs, mixed sources and ragged plan coverage all at once.

    Marked slow: it fits every group (~20s warm, minutes cold).
    """
    from dashboard_app.compute import (
        attach_descriptive_averages, attach_top_volume, compute_by_customer_frames,
    )
    from dashboard_app.datasources import discover_raw_files, load_raw_from_path
    from dashboard_app.pipeline import load_pipeline
    from dashboard_app.config import MODEL_OPTIONS

    mp = next(iter(MODEL_OPTIONS.values()))
    P = load_pipeline(mp)
    snap_date, raw = sorted(discover_raw_files())[-1]
    df = load_raw_from_path(raw, os.path.getmtime(raw), mp)
    today_ts = pd.Timestamp(snap_date)
    anchors = P.week_anchors(today_ts)

    by_cust, wbg, abg = compute_by_customer_frames(df, today_ts, mp)
    assert by_cust is not None and not by_cust.empty

    abg = resolve_demand(abg, by_cust)
    weekly = roll_up_to_sku_week(wbg, ["projected_pos"], min_count=0)
    agg = roll_up_to_sku_week(abg, ["POS", "Orders", "Projection", "demand"])
    by_cust = attach_current_projection(
        by_cust, abg, weekly["WeekDate"], ["Customer Grouping", "SKU"])
    summary = roll_up_summary(by_cust, agg, anchors, VIEW)
    summary = attach_descriptive_averages(
        summary, sku_grain_demand_frame(agg, VIEW), today_ts)
    summary = attach_top_volume(summary, P, df, today_ts)

    for col in ("Updated Projection Average", "Current Projection Average",
                "Projection Difference"):
        total = pd.to_numeric(summary[col], errors="coerce").sum()
        parts = pd.to_numeric(by_cust[col], errors="coerce").sum()
        assert total == pytest.approx(parts, rel=1e-9), (
            f"{col}: view total {total:,.2f} != sum of its rows {parts:,.2f}"
        )

    # One row per SKU, and every SKU in the parts is in the total.
    assert summary["SKU"].is_unique
    assert set(summary["SKU"]) == set(by_cust["SKU"].astype(str))

    # The actuals line is additive too — resolving the source per customer BEFORE
    # summing is what makes this hold for a mixed-source SKU.
    rolled = historical_window(agg, summary, anchors)
    per_group = historical_window(abg, by_cust, anchors)
    assert (rolled.groupby("WeekDate")["demand"].sum(min_count=1).sum()
            == pytest.approx(
                per_group.groupby("WeekDate")["demand"].sum(min_count=1).sum()))
