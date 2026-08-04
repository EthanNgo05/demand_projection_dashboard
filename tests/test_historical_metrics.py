"""Unit tests for dashboard_app.historical_metrics.

Pure pandas -- no Streamlit, no snapshot files, no AppTest. Every definition in
the module docstring that could reasonably be computed two ways is pinned here,
because the Historical Summary's whole value is that its numbers are trustworthy.
"""
import numpy as np
import pandas as pd
import pytest

from dashboard_app import config
from dashboard_app import historical_metrics as hm


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
# A Sunday, used as "last complete week" throughout so window maths is legible.
LCW = pd.Timestamp("2026-07-19")


class FakePipeline:
    """The two model-agnostic pipeline helpers historical_metrics calls.

    aggregate_to_sku_week mirrors models/regression.py exactly (including
    min_count=1, which keeps an all-NaN group NaN instead of collapsing to 0).
    """

    @staticmethod
    def aggregate_to_sku_week(df):
        grp = df.groupby(["SKU", "WeekDate"])
        agg = pd.concat(
            [grp["POS"].sum(min_count=1),
             grp["Orders"].sum(min_count=1),
             grp["Projection"].sum(min_count=1)],
            axis=1,
        ).reset_index()
        desc = df.dropna(subset=["Description"]).groupby("SKU")["Description"].first()
        agg["Description"] = agg["SKU"].map(desc)
        return agg

    @staticmethod
    def region_for_group(group):
        return {"US Retail": "US (LBC+NJ)", "EU Web": "EU (SH-CTS)"}.get(group, "Other")


def _weeks(end, n):
    """``n`` consecutive Sunday WeekDates ending at ``end``."""
    return [pd.Timestamp(end) - pd.Timedelta(weeks=i) for i in range(n - 1, -1, -1)]


def raw_rows(sku, customer, group, weeks, pos=None, orders=None, desc="Widget"):
    """Raw demand rows in the shape data_io._clean produces."""
    return pd.DataFrame({
        "SKU": sku,
        "Description": desc,
        "Customer": customer,
        "Customer Grouping": group,
        "WeekDate": weeks,
        "POS": pos if pos is not None else [np.nan] * len(weeks),
        "Orders": orders if orders is not None else [np.nan] * len(weeks),
        "Projection": [np.nan] * len(weeks),
    })


@pytest.fixture
def P():
    return FakePipeline()


@pytest.fixture
def three_year_frame(P):
    """~3 years of weekly history covering every case the module must handle."""
    weeks = _weeks(LCW, 160)
    raw = pd.concat([
        # Plain POS seller, 10 units every week, priced at $10.
        raw_rows("AAA", "Cust1", "US Retail", weeks, pos=[10.0] * len(weeks)),
        # Orders-only SKU (never any POS) -- must fall back to Orders.
        raw_rows("BBB", "Cust2", "EU Web", weeks, orders=[5.0] * len(weeks)),
        # Discontinued: sold 100/wk for the FIRST 80 weeks, nothing since. Carries
        # the trailing '*' on half its rows, as the real data does.
        raw_rows("CCC*", "Cust1", "US Retail", weeks[:40], pos=[100.0] * 40),
        raw_rows("CCC", "Cust1", "US Retail", weeks[40:80], pos=[100.0] * 40),
        # Unpriced SKU -- real units, but must never contribute revenue.
        raw_rows("DDD", "Cust1", "US Retail", weeks, pos=[7.0] * len(weeks)),
    ], ignore_index=True)
    prices = {"AAA": 10.0, "BBB": 20.0, "CCC": 50.0}
    return hm.build_frame(raw, P, prices, plytix_df=None), prices


# --------------------------------------------------------------------------- #
# Weekly frame: discontinued retention and '*' normalisation                   #
# --------------------------------------------------------------------------- #
def test_discontinued_skus_are_retained(three_year_frame):
    frame, _ = three_year_frame
    assert "CCC" in set(frame["SKU"]), "discontinued SKU must keep its history"


def test_star_suffix_collapses_into_one_series(three_year_frame):
    frame, _ = three_year_frame
    assert "CCC*" not in set(frame["SKU"]), "'*' must be stripped before aggregating"
    ccc = frame[frame["SKU"] == "CCC"]
    # 80 weeks of history, one row per week -- NOT 40 + 40 split across two codes.
    assert len(ccc) == 80
    assert ccc["WeekDate"].is_unique


def test_star_normalisation_does_not_double_count(P):
    """Same SKU-week under both codes must sum, not produce two rows."""
    week = [LCW]
    raw = pd.concat([
        raw_rows("EEE", "Cust1", "US Retail", week, pos=[3.0]),
        raw_rows("EEE*", "Cust2", "US Retail", week, pos=[4.0]),
    ], ignore_index=True)
    frame = hm.build_frame(raw, P, {"EEE": 1.0})
    assert len(frame) == 1
    assert frame["demand"].iloc[0] == 7.0


def test_empty_input_returns_typed_empty_frame(P):
    frame = hm.historical_weekly_frame(pd.DataFrame(), P)
    assert frame.empty
    assert "WeekDate" in frame.columns


# --------------------------------------------------------------------------- #
# Demand coalescing                                                            #
# --------------------------------------------------------------------------- #
def test_orders_only_sku_falls_back_to_orders(three_year_frame):
    frame, _ = three_year_frame
    bbb = frame[frame["SKU"] == "BBB"]
    assert (bbb["Data Source"] == "Orders").all()
    assert bbb["demand"].dropna().eq(5.0).all()


def test_pos_wins_when_both_signals_exist_in_a_week(P):
    """One signal per row -- never the sum of the two."""
    frame = hm.build_frame(
        raw_rows("BOTH", "C", "US Retail", [LCW], pos=[6.0], orders=[4.0]),
        P, {"BOTH": 1.0},
    )
    assert frame["demand"].iloc[0] == 6.0, "POS wins; 10.0 would be double-counting"


def test_off_label_weeks_are_still_counted(P):
    """The bug this module was rewritten to fix.

    A pair with POS early and Orders later must count BOTH stretches. Labelling
    the pair "POS" for its whole history and reading only the POS column dropped
    135,327 units across 6,156 SKU-weeks of the live snapshot.
    """
    weeks = _weeks(LCW, 60)
    raw = pd.concat([
        raw_rows("SWITCH", "Cust1", "US Retail", weeks[:20], pos=[9.0] * 20),
        raw_rows("SWITCH", "Cust1", "US Retail", weeks[20:], orders=[1.0] * 40),
    ], ignore_index=True)
    frame = hm.build_frame(raw, P, {"SWITCH": 1.0})
    assert frame["demand"].sum() == pytest.approx(20 * 9.0 + 40 * 1.0)
    assert (frame["Data Source"] == "Mixed").all(), \
        "a pair using both signals is Mixed, not POS"


def test_orders_survive_when_another_customer_reports_pos_same_week(P):
    """Row-level coalescing must happen BEFORE customers are summed.

    Coalescing after the customer aggregation keeps A's POS and silently loses
    B's Orders for the same SKU-week.
    """
    raw = pd.concat([
        raw_rows("SHARED", "CustA", "US Retail", [LCW], pos=[10.0]),
        raw_rows("SHARED", "CustB", "US Retail", [LCW], orders=[3.0]),
    ], ignore_index=True)
    frame = hm.build_frame(raw, P, {"SHARED": 1.0})
    assert len(frame) == 1
    assert frame["demand"].iloc[0] == 13.0, "B's Orders must not be discarded"


def test_no_signal_stays_nan_not_zero(P):
    frame = hm.build_frame(raw_rows("ZZZ", "C", "US Retail", [LCW]), P, {})
    assert pd.isna(frame["demand"].iloc[0]), "'no data' must not become 'sold nothing'"


# --------------------------------------------------------------------------- #
# Revenue                                                                      #
# --------------------------------------------------------------------------- #
def test_unpriced_sku_excluded_from_revenue_not_zeroed(three_year_frame):
    frame, _ = three_year_frame
    ddd = frame[frame["SKU"] == "DDD"]
    assert ddd["demand"].notna().all(), "units are real"
    assert ddd["revenue"].isna().all(), "revenue must be unknown, not 0"


def test_revenue_is_units_times_price(three_year_frame):
    frame, _ = three_year_frame
    aaa = frame[frame["SKU"] == "AAA"]
    assert aaa["revenue"].dropna().eq(100.0).all()   # 10 units x $10


def test_prices_may_be_a_series_not_only_a_dict(P):
    """The app passes prices as a pandas Series; `if not prices` raises on one.

    Regression guard: attach_revenue used to test the raw input for truthiness,
    which blew up with "truth value of a Series is ambiguous" against real data
    even though every dict-based unit test passed.
    """
    raw = raw_rows("AAA", "C", "US Retail", [LCW], pos=[4.0])
    series = pd.Series({"AAA": 2.5})
    frame = hm.build_frame(raw, P, series)
    assert frame["revenue"].iloc[0] == 10.0


def test_price_lookup_ignores_a_star_suffix(P):
    """Prices are keyed on the base code, so a discontinued SKU still prices."""
    raw = raw_rows("CCC*", "C", "US Retail", [LCW], pos=[2.0])
    frame = hm.build_frame(raw, P, pd.Series({"CCC*": 3.0}))
    assert frame["revenue"].iloc[0] == 6.0


@pytest.mark.parametrize("prices", [None, {}, pd.Series(dtype="float64")])
def test_no_prices_leaves_revenue_unknown(P, prices):
    frame = hm.build_frame(
        raw_rows("AAA", "C", "US Retail", [LCW], pos=[4.0]), P, prices
    )
    assert frame["revenue"].isna().all(), "no prices means unknown, not $0"
    assert frame["demand"].iloc[0] == 4.0, "units are unaffected"


def test_price_coverage_reports_the_gap(three_year_frame):
    frame, _ = three_year_frame
    start, end = hm.window_bounds(hm.WINDOW_13W, LCW)
    priced, total, pct = hm.price_coverage(frame, start, end)
    # AAA + BBB priced, DDD not (CCC stopped selling long before this window).
    assert (priced, total) == (2, 3)
    assert 0 < pct < 100


# --------------------------------------------------------------------------- #
# Windows and year-over-year                                                   #
# --------------------------------------------------------------------------- #
def test_ytd_starts_at_january_first():
    start, end = hm.window_bounds(hm.WINDOW_YTD, LCW)
    assert start == pd.Timestamp("2026-01-01")
    assert end == LCW


@pytest.mark.parametrize("kind,weeks", [
    (hm.WINDOW_13W, 13), (hm.WINDOW_52W, 52), (hm.WINDOW_3Y, 156),
])
def test_rolling_windows_span_whole_weeks_inclusive(kind, weeks):
    start, end = hm.window_bounds(kind, LCW)
    assert (end - start).days == (weeks - 1) * 7
    assert end == LCW, "no window may reach into the in-progress week"


def test_unknown_window_raises():
    with pytest.raises(ValueError):
        hm.window_bounds("Last fortnight", LCW)


def test_custom_window_sentinel_raises_rather_than_guessing():
    """WINDOW_CUSTOM has no fixed span; resolving it silently would be a wrong answer."""
    with pytest.raises(ValueError, match="sentinel"):
        hm.window_bounds(hm.WINDOW_CUSTOM, LCW)


def test_named_windows_excludes_only_the_custom_sentinel():
    assert set(hm.WINDOW_OPTIONS) - set(hm.NAMED_WINDOWS) == {hm.WINDOW_CUSTOM}


@pytest.mark.parametrize("kind,weeks", [
    (hm.WINDOW_4W, 4), (hm.WINDOW_26W, 26), (hm.WINDOW_2Y, 104),
])
def test_new_rolling_windows_span_whole_weeks(kind, weeks):
    start, end = hm.window_bounds(kind, LCW)
    assert (end - start).days == (weeks - 1) * 7
    assert end == LCW


def test_last_full_calendar_year_is_jan_to_dec_of_the_prior_year():
    start, end = hm.window_bounds(hm.WINDOW_LAST_YEAR, LCW)
    assert start == pd.Timestamp("2025-01-01")
    assert end == pd.Timestamp("2025-12-31")


@pytest.mark.parametrize("lcw", [
    pd.Timestamp("2026-01-04"),   # earliest plausible as-of week in a new year
    pd.Timestamp("2026-07-19"),
    pd.Timestamp("2026-12-27"),
])
def test_last_full_calendar_year_always_ends_before_the_as_of_week(lcw):
    """Which is why the branch needs no clamp against lcw."""
    start, end = hm.window_bounds(hm.WINDOW_LAST_YEAR, lcw)
    assert start == pd.Timestamp(year=lcw.year - 1, month=1, day=1)
    assert end == pd.Timestamp(year=lcw.year - 1, month=12, day=31)
    assert end < lcw


# --------------------------------------------------------------------------- #
# Custom range: snapping arbitrary dates to whole weeks                        #
# --------------------------------------------------------------------------- #
# NOTE on the end boundary: lcw is the SUNDAY that starts the last complete week,
# so a caller wanting that week included passes its Saturday (lcw + 6 days) — which
# is exactly what the view's date picker offers as its maximum.
LAST_DAY = LCW + pd.Timedelta(days=6)


def test_snap_window_includes_the_last_complete_week(three_year_frame):
    """Picking the maximum offered date must not silently drop the newest week."""
    start = LCW - pd.Timedelta(weeks=4)
    assert hm.snap_window(start, LAST_DAY, LCW) == (start, LCW)
    frame, _ = three_year_frame
    assert hm.window_totals(frame, *hm.snap_window(start, LAST_DAY, LCW))["weeks"] == 5


def test_snap_window_leaves_a_whole_week_pair_untouched():
    start = LCW - pd.Timedelta(weeks=4)
    assert hm.snap_window(start, LAST_DAY, LCW) == (start, LCW)


def test_snap_window_moves_a_midweek_start_forward():
    """A part-week at the front would otherwise be counted as a whole one."""
    wednesday = LCW - pd.Timedelta(weeks=4) + pd.Timedelta(days=3)
    snapped_start, _ = hm.snap_window(wednesday, LAST_DAY, LCW)
    assert snapped_start.day_name() == "Sunday"
    assert snapped_start > wednesday, "must move forward, never back"
    assert (snapped_start - wednesday).days == 4


def test_snap_window_excludes_a_partly_covered_trailing_week():
    """A week whose span runs past the chosen end is dropped, not counted whole."""
    start = LCW - pd.Timedelta(weeks=6)
    # Wednesday of the last complete week: that week runs to its Saturday, past the
    # requested end, so it must not be included.
    wednesday = LCW + pd.Timedelta(days=3)
    _, snapped_end = hm.snap_window(start, wednesday, LCW)
    assert snapped_end == LCW - pd.Timedelta(weeks=1)


def test_snap_window_clamps_to_the_last_complete_week():
    """A range reaching into the future stops at the last complete week."""
    start = LCW - pd.Timedelta(weeks=8)
    far_future = LCW + pd.Timedelta(weeks=20)
    assert hm.snap_window(start, far_future, LCW) == (start, LCW)


@pytest.mark.parametrize("days", [0, 1, 3, 6])
def test_snap_window_returns_none_for_a_sub_week_range(days):
    """Too narrow to hold one whole week -> the view warns instead of showing zeroes."""
    monday = LCW - pd.Timedelta(weeks=3) + pd.Timedelta(days=1)
    assert hm.snap_window(monday, monday + pd.Timedelta(days=days), LCW) is None


def test_snap_window_result_behaves_like_any_named_window(three_year_frame):
    frame, _ = three_year_frame
    # A Friday start snaps forward to LCW - 10 weeks; through LCW inclusive that is
    # 11 Sundays, not 10.
    bounds = hm.snap_window(LCW - pd.Timedelta(weeks=10, days=2), LAST_DAY, LCW)
    assert bounds == (LCW - pd.Timedelta(weeks=10), LCW)
    assert hm.window_totals(frame, *bounds)["weeks"] == 11
    # Tile/modal agreement must hold on a custom span too.
    assert hm.breadth(frame, *bounds)["dormant_skus"] == len(
        hm.dormant_skus_breakdown(frame, *bounds))


def test_prior_year_window_preserves_weekday():
    start, end = hm.window_bounds(hm.WINDOW_52W, LCW)
    p_start, p_end = hm.prior_year_window(start, end)
    assert p_end.weekday() == end.weekday(), "364 days keeps Sunday on Sunday"
    assert (end - p_end).days == 364
    assert (p_end - p_start).days == (end - start).days


def test_ytd_prior_year_anchors_to_prior_january_first():
    start, end = hm.window_bounds(hm.WINDOW_YTD, LCW)
    p_start, p_end = hm.prior_year_window(start, end, anchor_to_year_start=True)
    assert p_start == pd.Timestamp("2025-01-01")
    assert p_end == LCW - pd.Timedelta(days=364)


def test_yoy_comparison_is_like_for_like(three_year_frame):
    """A flat seller must show ~0% YoY, not an artefact of window length."""
    frame, _ = three_year_frame
    aaa = frame[frame["SKU"] == "AAA"]
    start, end = hm.window_bounds(hm.WINDOW_52W, LCW)
    p_start, p_end = hm.prior_year_window(start, end)
    cur = hm.window_totals(aaa, start, end)
    pri = hm.window_totals(aaa, p_start, p_end)
    assert cur["weeks"] == pri["weeks"] == 52
    assert hm.pct_change(cur["revenue"], pri["revenue"]) == pytest.approx(0.0)


@pytest.mark.parametrize("current,prior", [(5.0, 0.0), (5.0, None), (5.0, np.nan)])
def test_pct_change_returns_none_without_a_base(current, prior):
    assert hm.pct_change(current, prior) is None


def test_pct_change_uses_absolute_base():
    assert hm.pct_change(150.0, 100.0) == pytest.approx(50.0)
    assert hm.pct_change(50.0, 100.0) == pytest.approx(-50.0)


# --------------------------------------------------------------------------- #
# Month bucketing                                                              #
# --------------------------------------------------------------------------- #
def test_week_belongs_wholly_to_the_month_of_its_sunday(P):
    """2026-05-31 is a Sunday whose week runs into June; it counts as MAY."""
    straddler = pd.Timestamp("2026-05-31")
    assert straddler.day_name() == "Sunday"
    frame = hm.build_frame(
        raw_rows("AAA", "C", "US Retail", [straddler], pos=[10.0]), P, {"AAA": 1.0}
    )
    monthly = hm.monthly_totals(frame)
    assert len(monthly) == 1
    assert (monthly["Year"].iloc[0], monthly["MonthNum"].iloc[0]) == (2026, 5)
    assert monthly["revenue"].iloc[0] == 10.0


def test_monthly_totals_are_chronological(three_year_frame):
    frame, _ = three_year_frame
    monthly = hm.monthly_totals(frame)
    assert monthly["MonthStart"].is_monotonic_increasing


# --------------------------------------------------------------------------- #
# Breadth                                                                      #
# --------------------------------------------------------------------------- #
def test_active_counts_only_skus_that_sold(three_year_frame):
    frame, _ = three_year_frame
    start, end = hm.window_bounds(hm.WINDOW_13W, LCW)
    b = hm.breadth(frame, start, end)
    # CCC stopped 80 weeks ago, so only AAA/BBB/DDD are active now.
    assert b["active_skus"] == 3
    assert b["active_customers"] == 2


def test_new_sku_is_measured_against_all_history(P):
    weeks = _weeks(LCW, 60)
    raw = pd.concat([
        raw_rows("OLD", "C", "US Retail", weeks, pos=[1.0] * 60),
        raw_rows("NEW", "C", "US Retail", weeks[-4:], pos=[1.0] * 4),
    ], ignore_index=True)
    frame = hm.build_frame(raw, P, {})
    start, end = hm.window_bounds(hm.WINDOW_13W, LCW)
    assert hm.breadth(frame, start, end)["new_skus"] == 1


def test_dormant_sku_sold_before_the_window_but_not_inside_it(P):
    weeks = _weeks(LCW, 40)
    raw = pd.concat([
        raw_rows("STEADY", "C", "US Retail", weeks, pos=[1.0] * 40),
        # Sold only in the 52 weeks before the 13-week window opened.
        raw_rows("GONE", "C", "US Retail", weeks[:20], pos=[1.0] * 20),
    ], ignore_index=True)
    frame = hm.build_frame(raw, P, {})
    start, end = hm.window_bounds(hm.WINDOW_13W, LCW)
    assert hm.breadth(frame, start, end)["dormant_skus"] == 1


def test_breadth_on_empty_frame_is_all_zero(P):
    b = hm.breadth(hm.build_frame(pd.DataFrame(), P, {}), LCW, LCW)
    assert set(b.values()) == {0}


# --------------------------------------------------------------------------- #
# Concentration, Pareto, movers, top SKUs                                      #
# --------------------------------------------------------------------------- #
def test_concentration_is_a_percentage_of_priced_revenue(three_year_frame):
    frame, _ = three_year_frame
    start, end = hm.window_bounds(hm.WINDOW_13W, LCW)
    pct = hm.concentration(frame, start, end, n=10)
    # Only two priced SKUs sell in this window, so the top 10 are all of them.
    assert pct == pytest.approx(100.0)


def test_concentration_is_none_when_nothing_is_priced(P):
    frame = hm.build_frame(
        raw_rows("AAA", "C", "US Retail", _weeks(LCW, 5), pos=[1.0] * 5), P, {}
    )
    start, end = hm.window_bounds(hm.WINDOW_13W, LCW)
    assert hm.concentration(frame, start, end) is None


def test_pareto_share_is_monotonic_and_ends_at_100(three_year_frame):
    frame, _ = three_year_frame
    start, end = hm.window_bounds(hm.WINDOW_3Y, LCW)
    p = hm.pareto(frame, start, end)
    assert not p.empty
    assert p["cum_share"].is_monotonic_increasing
    assert p["cum_share"].iloc[-1] == pytest.approx(100.0)
    assert p["revenue"].is_monotonic_decreasing
    assert list(p["rank"]) == list(range(1, len(p) + 1))


def test_top_skus_ranked_by_revenue_not_units(P):
    """DEAR sells fewer units than CHEAP but earns more -- it must rank first."""
    weeks = _weeks(LCW, 10)
    raw = pd.concat([
        raw_rows("CHEAP", "C", "US Retail", weeks, pos=[100.0] * 10),
        raw_rows("DEAR", "C", "US Retail", weeks, pos=[5.0] * 10),
    ], ignore_index=True)
    frame = hm.build_frame(raw, P, {"CHEAP": 1.0, "DEAR": 100.0})
    start, end = hm.window_bounds(hm.WINDOW_13W, LCW)
    top = hm.top_skus(frame, start, end, n=5)
    assert list(top["SKU"]) == ["DEAR", "CHEAP"]
    assert top["units"].iloc[0] == 50.0


def test_yoy_movers_sorted_gainers_first(P):
    weeks = _weeks(LCW, 105)
    prior, current = weeks[:52], weeks[53:]
    raw = pd.concat([
        raw_rows("UP", "C", "US Retail", prior, pos=[1.0] * len(prior)),
        raw_rows("UP", "C", "US Retail", current, pos=[10.0] * len(current)),
        raw_rows("DOWN", "C", "US Retail", prior, pos=[10.0] * len(prior)),
        raw_rows("DOWN", "C", "US Retail", current, pos=[1.0] * len(current)),
    ], ignore_index=True)
    frame = hm.build_frame(raw, P, {"UP": 1.0, "DOWN": 1.0})
    movers = hm.yoy_movers(frame, LCW, n=5)
    assert movers["delta"].is_monotonic_decreasing
    assert movers["SKU"].iloc[0] == "UP" and movers["delta"].iloc[0] > 0
    assert movers["SKU"].iloc[-1] == "DOWN" and movers["delta"].iloc[-1] < 0


def test_yoy_movers_counts_a_launch_as_a_gain(P):
    """A SKU absent from the prior year is the biggest kind of mover."""
    weeks = _weeks(LCW, 20)
    frame = hm.build_frame(
        raw_rows("LAUNCH", "C", "US Retail", weeks, pos=[10.0] * 20), P,
        {"LAUNCH": 5.0},
    )
    movers = hm.yoy_movers(frame, LCW, n=5)
    assert movers["prior"].iloc[0] == 0.0
    assert movers["delta"].iloc[0] > 0


# --------------------------------------------------------------------------- #
# Dimensions and folding                                                       #
# --------------------------------------------------------------------------- #
def test_region_is_derived_from_customer_grouping(three_year_frame):
    frame, _ = three_year_frame
    assert set(frame["Region"]) == {"US (LBC+NJ)", "EU (SH-CTS)"}


def test_sku_type_defaults_to_unspecified_without_plytix(three_year_frame):
    frame, _ = three_year_frame
    assert (frame["SKU Type"] == hm.UNSPECIFIED_TYPE).all()


def test_sku_type_matches_a_discontinued_code(P):
    plytix = pd.DataFrame({"SKU": ["CCC"], "SKU Type": ["Sensor Can"]})
    raw = raw_rows("CCC*", "C", "US Retail", [LCW], pos=[1.0])
    frame = hm.build_frame(raw, P, {"CCC": 1.0}, plytix_df=plytix)
    assert frame["SKU Type"].iloc[0] == "Sensor Can"


def test_fold_to_top_n_collapses_the_tail():
    totals = pd.DataFrame({"Region": list("ABCDEFGHIJ"),
                           "revenue": [10.0] * 9 + [5.0]})
    folded = hm.fold_to_top_n(totals, n=8, label_col="Region")
    assert len(folded) == 9
    assert folded["Region"].iloc[-1] == hm.OTHER_LABEL
    assert folded["revenue"].sum() == pytest.approx(totals["revenue"].sum())


def test_weekly_by_dimension_membership_is_fixed_across_weeks(P):
    """A category must not flicker in and out of 'Other' week to week."""
    weeks = _weeks(LCW, 4)
    raw = pd.concat([
        raw_rows("A", "C1", "US Retail", weeks, pos=[100.0] * 4),
        raw_rows("B", "C2", "EU Web", weeks, pos=[1.0] * 4),
        # Spikes in one week only -- must stay in 'Other' every week.
        raw_rows("C", "C3", "Somewhere Else", weeks, pos=[0.0, 0.0, 0.0, 500.0]),
    ], ignore_index=True)
    frame = hm.build_frame(raw, P, {"A": 1.0, "B": 1.0, "C": 1.0})
    stacked = hm.weekly_by_dimension(frame, "Region", top_n=2)
    per_week = {frozenset(g) for _, g in stacked.groupby("WeekDate")["Region"]}
    assert len(per_week) == 1, "the same categories must appear every week"
    assert hm.OTHER_LABEL in next(iter(per_week)), "the spiky tail folds to Other"


def test_by_dimension_sorted_largest_first(three_year_frame):
    frame, _ = three_year_frame
    start, end = hm.window_bounds(hm.WINDOW_13W, LCW)
    out = hm.by_dimension(frame, "Region", start, end)
    assert out["revenue"].is_monotonic_decreasing


# --------------------------------------------------------------------------- #
# Chart-shaped helpers                                                         #
# --------------------------------------------------------------------------- #
def test_seasonality_week_of_year_starts_at_one():
    frame = pd.DataFrame({
        "WeekDate": [pd.Timestamp("2026-01-04"), pd.Timestamp("2026-01-11")],
        "revenue": [1.0, 2.0],
    })
    s = hm.seasonality_frame(frame)
    assert list(s["WeekOfYear"]) == [1, 2]
    assert set(s["Year"]) == {2026}


def test_month_year_matrix_has_twelve_rows(three_year_frame):
    frame, _ = three_year_frame
    m = hm.month_year_matrix(frame)
    assert list(m.index) == list(range(1, 13))


def test_weekly_totals_sorted_by_date(three_year_frame):
    frame, _ = three_year_frame
    w = hm.weekly_totals(frame)
    assert w["WeekDate"].is_monotonic_increasing


@pytest.mark.parametrize("fn,args", [
    (hm.weekly_totals, ()),
    (hm.monthly_totals, ()),
    (hm.seasonality_frame, ()),
])
def test_chart_helpers_tolerate_an_empty_frame(fn, args):
    assert fn(pd.DataFrame(), *args).empty


# --------------------------------------------------------------------------- #
# Breakdowns: the lists behind the clickable KPI tiles                         #
# --------------------------------------------------------------------------- #
# The tile numbers ARE len() of these frames, so these tests are what stops a tile
# reading 47 above a list of 45 rows.

_BREADTH_PAIRS = [
    ("active_skus", hm.active_skus_breakdown),
    ("active_customers", hm.active_customers_breakdown),
    ("new_skus", hm.new_skus_breakdown),
    ("dormant_skus", hm.dormant_skus_breakdown),
]


@pytest.mark.parametrize("kind", list(hm.NAMED_WINDOWS))
@pytest.mark.parametrize("key,fn", _BREADTH_PAIRS)
def test_tile_count_equals_its_breakdown_row_count(three_year_frame, key, fn, kind):
    """Every assortment tile must equal the length of the list it opens.

    Checked across every analysis window, because the window is what a planner
    changes right before they click.
    """
    frame, _ = three_year_frame
    start, end = hm.window_bounds(kind, LCW)
    assert hm.breadth(frame, start, end)[key] == len(fn(frame, start, end))


def test_concentration_equals_its_breakdown_share_total(three_year_frame):
    frame, _ = three_year_frame
    start, end = hm.window_bounds(hm.WINDOW_3Y, LCW)
    top = hm.top_share_breakdown(frame, start, end, n=10)
    assert hm.concentration(frame, start, end, n=10) == pytest.approx(
        top["Share %"].sum()
    )


@pytest.mark.parametrize("fn,expected", [
    (hm.active_skus_breakdown,
     ["SKU", "Description", "Units", "Revenue", "Weeks with sales"]),
    (hm.active_customers_breakdown,
     ["Customer Grouping", "Region", "SKUs", "Units", "Revenue"]),
    (hm.new_skus_breakdown,
     ["SKU", "Description", "First sale week", "Units", "Revenue"]),
    (hm.dormant_skus_breakdown,
     ["SKU", "Description", "Last sale week", "Weeks since last sale",
      "Units (prior 52 wks)", "Revenue (prior 52 wks)"]),
])
def test_breakdown_columns_are_exactly_as_declared(three_year_frame, fn, expected):
    frame, _ = three_year_frame
    start, end = hm.window_bounds(hm.WINDOW_52W, LCW)
    assert list(fn(frame, start, end).columns) == expected


@pytest.mark.parametrize("fn", [f for _, f in _BREADTH_PAIRS] +
                         [hm.top_share_breakdown, hm.monthly_breakdown,
                          hm.weekly_breakdown])
def test_breakdowns_tolerate_an_empty_frame(fn):
    """A filter combination selecting nothing must give an empty table, not raise."""
    out = fn(pd.DataFrame(columns=["SKU", "WeekDate", "demand", "revenue",
                                   "Description", "Customer Grouping"]),
             LCW - pd.Timedelta(weeks=4), LCW)
    assert out.empty and list(out.columns), "must keep its declared columns"


def test_active_skus_ranked_by_revenue_with_week_counts(P):
    weeks = _weeks(LCW, 6)
    raw = pd.concat([
        raw_rows("BIG", "C", "US Retail", weeks, pos=[10.0] * 6),
        raw_rows("SMALL", "C", "US Retail", weeks[:2], pos=[1.0] * 2),
    ], ignore_index=True)
    frame = hm.build_frame(raw, P, {"BIG": 5.0, "SMALL": 1.0})
    start, end = hm.window_bounds(hm.WINDOW_13W, LCW)
    out = hm.active_skus_breakdown(frame, start, end)
    assert list(out["SKU"]) == ["BIG", "SMALL"]
    assert list(out["Weeks with sales"]) == [6, 2]
    assert out["Revenue"].iloc[0] == 300.0     # 6 wks x 10 units x $5


def test_zero_demand_week_is_not_a_sale(P):
    """0 units is "stocked, sold nothing" -- it must not make a SKU active."""
    frame = hm.build_frame(
        raw_rows("ZERO", "C", "US Retail", _weeks(LCW, 3), pos=[0.0, 0.0, 0.0]),
        P, {"ZERO": 1.0},
    )
    start, end = hm.window_bounds(hm.WINDOW_13W, LCW)
    assert hm.active_skus_breakdown(frame, start, end).empty
    assert hm.breadth(frame, start, end)["active_skus"] == 0


def test_active_customers_carries_region_and_sku_count(three_year_frame):
    frame, _ = three_year_frame
    start, end = hm.window_bounds(hm.WINDOW_13W, LCW)
    out = hm.active_customers_breakdown(frame, start, end)
    assert set(out["Customer Grouping"]) == {"US Retail", "EU Web"}
    assert out["Region"].notna().all()
    us = out[out["Customer Grouping"] == "US Retail"].iloc[0]
    assert us["SKUs"] == 2          # AAA + DDD (CCC stopped long ago)


def test_new_skus_reports_its_first_sale_week(P):
    weeks = _weeks(LCW, 60)
    raw = pd.concat([
        raw_rows("OLD", "C", "US Retail", weeks, pos=[1.0] * 60),
        raw_rows("NEW", "C", "US Retail", weeks[-3:], pos=[2.0] * 3),
    ], ignore_index=True)
    frame = hm.build_frame(raw, P, {"NEW": 4.0})
    start, end = hm.window_bounds(hm.WINDOW_13W, LCW)
    out = hm.new_skus_breakdown(frame, start, end)
    assert list(out["SKU"]) == ["NEW"]
    assert out["First sale week"].iloc[0] == weeks[-3]
    assert out["Revenue"].iloc[0] == 24.0      # 3 wks x 2 units x $4


def test_dormant_reports_last_sale_and_weeks_since(P):
    weeks = _weeks(LCW, 40)
    raw = pd.concat([
        raw_rows("STEADY", "C", "US Retail", weeks, pos=[1.0] * 40),
        raw_rows("GONE", "C", "US Retail", weeks[:20], pos=[3.0] * 20),
    ], ignore_index=True)
    frame = hm.build_frame(raw, P, {"GONE": 2.0})
    start, end = hm.window_bounds(hm.WINDOW_13W, LCW)
    out = hm.dormant_skus_breakdown(frame, start, end)
    assert list(out["SKU"]) == ["GONE"]
    last = weeks[19]
    assert out["Last sale week"].iloc[0] == last
    assert out["Weeks since last sale"].iloc[0] == (end - last).days // 7
    assert out["Units (prior 52 wks)"].iloc[0] == 60.0


def test_top_share_has_units_and_dollars_and_a_running_total(three_year_frame):
    """The request was explicit: top 10 with units AND amount."""
    frame, _ = three_year_frame
    start, end = hm.window_bounds(hm.WINDOW_3Y, LCW)
    out = hm.top_share_breakdown(frame, start, end, n=10)
    assert list(out.columns) == ["SKU", "Description", "Units", "Revenue",
                                 "Share %", "Cumulative %"]
    assert out["Units"].notna().all() and (out["Units"] > 0).all()
    assert out["Revenue"].is_monotonic_decreasing
    assert out["Cumulative %"].is_monotonic_increasing
    assert out["Cumulative %"].iloc[-1] == pytest.approx(out["Share %"].sum())


def test_top_share_is_a_share_of_all_revenue_not_of_the_table(P):
    """With 12 priced SKUs, the top 10 must sum to LESS than 100%."""
    weeks = _weeks(LCW, 5)
    raw = pd.concat([
        raw_rows(f"S{i:02d}", "C", "US Retail", weeks, pos=[float(i + 1)] * 5)
        for i in range(12)
    ], ignore_index=True)
    frame = hm.build_frame(raw, P, {f"S{i:02d}": 1.0 for i in range(12)})
    start, end = hm.window_bounds(hm.WINDOW_13W, LCW)
    out = hm.top_share_breakdown(frame, start, end, n=10)
    assert len(out) == 10
    assert out["Share %"].sum() < 100.0


def test_monthly_breakdown_aligns_prior_year_by_calendar_month(P):
    """Jan must compare against Jan, not against 364 days earlier."""
    this_jan = [pd.Timestamp("2026-01-04"), pd.Timestamp("2026-01-11")]
    last_jan = [pd.Timestamp("2025-01-05"), pd.Timestamp("2025-01-12")]
    raw = pd.concat([
        raw_rows("AAA", "C", "US Retail", this_jan, pos=[10.0, 10.0]),
        raw_rows("AAA", "C", "US Retail", last_jan, pos=[5.0, 5.0]),
    ], ignore_index=True)
    frame = hm.build_frame(raw, P, {"AAA": 1.0})
    out = hm.monthly_breakdown(
        frame, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01"),
        pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01"),
    )
    assert list(out["Month"]) == ["Jan 2026"]
    assert out["Revenue"].iloc[0] == 20.0
    assert out["Revenue (prior year)"].iloc[0] == 10.0
    assert out["YoY %"].iloc[0] == pytest.approx(100.0)


def test_monthly_breakdown_leaves_missing_prior_year_blank(three_year_frame):
    frame, _ = three_year_frame
    start, end = hm.window_bounds(hm.WINDOW_13W, LCW)
    out = hm.monthly_breakdown(frame, start, end)   # no prior window supplied
    assert out["Revenue (prior year)"].isna().all()
    assert out["YoY %"].isna().all(), "no base means no percentage, not 0%"


def test_weekly_breakdown_is_most_recent_first(three_year_frame):
    frame, _ = three_year_frame
    start, end = hm.window_bounds(hm.WINDOW_13W, LCW)
    out = hm.weekly_breakdown(frame, start, end)
    assert len(out) == 13
    assert out["Week"].is_monotonic_decreasing


# --------------------------------------------------------------------------- #
# Compact tile formatting                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value,money,expected", [
    (0, False, "0"),
    (999, False, "999"),
    (1_000, False, "1.0K"),
    (1_234, False, "1.2K"),
    (999_999, False, "1,000.0K"),
    (1_000_000, False, "1.0M"),
    (4_082_263, False, "4.1M"),
    (210_022_936, True, "$210.0M"),
    (2_100_000_000, True, "$2.1B"),
    (1.5e12, True, "$1.5T"),
    (-5_000, True, "-$5.0K"),
    (500, True, "$500"),
])
def test_fmt_compact(value, money, expected):
    assert config.fmt_compact(value, money=money) == expected


@pytest.mark.parametrize("blank", [None, float("nan")])
def test_fmt_compact_blanks_match_fmt_dollar(blank):
    assert config.fmt_compact(blank) == "—" == config.fmt_dollar(blank)


def _metrics_for(frame, lcw, window):
    """The metrics dict historical_summary._render_kpis consumes.

    Mirrors historical_summary._metrics without Streamlit's session cache, so the
    tile specs can be exercised as pure functions.
    """
    start, end = hm.window_bounds(window, lcw)
    ytd = hm.window_bounds(hm.WINDOW_YTD, lcw)
    ytd_prior = hm.prior_year_window(*ytd, anchor_to_year_start=True)
    w13 = hm.window_bounds(hm.WINDOW_13W, lcw)
    w52 = hm.window_bounds(hm.WINDOW_52W, lcw)
    w4 = (lcw - pd.Timedelta(weeks=3), lcw)
    return {
        "bounds": (start, end), "window_kind": window,
        "window": hm.window_totals(frame, start, end),
        "ytd": hm.window_totals(frame, *ytd),
        "ytd_prior": hm.window_totals(frame, *ytd_prior),
        "w4": hm.window_totals(frame, *w4),
        "w4_prev": hm.window_totals(frame, w4[0] - pd.Timedelta(weeks=4),
                                    w4[0] - pd.Timedelta(days=1)),
        "w13": hm.window_totals(frame, *w13),
        "w13_prev": hm.window_totals(frame, w13[0] - pd.Timedelta(weeks=13),
                                     w13[0] - pd.Timedelta(days=1)),
        "w52": hm.window_totals(frame, *w52),
        "breadth": hm.breadth(frame, start, end),
        "concentration": hm.concentration(frame, start, end),
        "coverage": hm.price_coverage(frame, start, end),
    }


@pytest.mark.parametrize("window", list(hm.NAMED_WINDOWS))
def test_no_two_tiles_show_the_same_thing(three_year_frame, window):
    """Two tiles must never be the same number at any window setting.

    A window-scoped "Revenue (window)" tile used to duplicate "YTD Revenue" exactly
    whenever the window was Year to date — the default — so the default view showed
    one figure twice and both tiles opened the same breakdown.
    """
    from dashboard_app import historical_summary as hs

    frame, _ = three_year_frame
    m = _metrics_for(frame, LCW, window)
    labels = [t for _, ids in hs._SECTIONS for t in ids]

    seen = {}
    for tile_id in labels:
        key = (hs._tile_label(tile_id, window),
               hs._tile_value(tile_id, m, compact=False))
        seen.setdefault(key, []).append(tile_id)
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    assert not dupes, f"tiles duplicate each other at {window!r}: {dupes}"


def test_every_tile_has_a_breakdown(three_year_frame):
    """No tile may be a dead end — the whole point of making them clickable."""
    from dashboard_app import historical_summary as hs

    frame, _ = three_year_frame
    m = _metrics_for(frame, LCW, hm.WINDOW_3Y)
    for tile_id in hs._TILES:
        table, note = hs._breakdown_frame(tile_id, frame, m, LCW)
        assert table is not None, f"{tile_id} has no breakdown"
        assert list(table.columns), f"{tile_id} breakdown has no columns"
        assert note, f"{tile_id} breakdown has no explanatory note"


def test_percent_tile_tooltip_is_more_precise_than_the_tile(three_year_frame):
    """The exact form must actually add precision, not repeat the rounded tile."""
    from dashboard_app import historical_summary as hs

    frame, _ = three_year_frame
    m = _metrics_for(frame, LCW, hm.WINDOW_3Y)
    assert hs._tile_value("top10_share", m).endswith("%")
    assert "." in hs._tile_value("top10_share", m, compact=False)


def test_fmt_compact_keeps_a_near_constant_width_across_magnitudes():
    """The whole point: twelve tiles side by side need one rhythm.

    Asserts the SPREAD rather than exact widths — "$1.2K" (5) through "$210.0M" (7)
    is the range that matters. The exact figures these replace span "$500" to
    "$210,022,936", a 12-character spread, which is what made the grid unreadable.
    """
    compact = [config.fmt_compact(v, money=True)
               for v in (1.2e3, 45.6e3, 1.2e6, 210e6, 2.1e9)]
    widths = [len(s) for s in compact]
    assert max(widths) - min(widths) <= 2, f"widths vary too much: {compact}"

    exact = [config.fmt_dollar(v) for v in (1.2e3, 45.6e3, 1.2e6, 210e6, 2.1e9)]
    exact_widths = [len(s) for s in exact]
    assert max(exact_widths) - min(exact_widths) > max(widths) - min(widths), \
        "compact form must be more uniform than the exact form it replaces"
