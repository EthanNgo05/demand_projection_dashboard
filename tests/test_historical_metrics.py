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
    """WINDOW_CUSTOM has no fixed span; resolving it silently would be a wrong answer.

    The message must name the resolver, so the traceback tells a caller what to do.
    """
    with pytest.raises(ValueError, match="snap_window"):
        hm.window_bounds(hm.WINDOW_CUSTOM, LCW)


def test_named_windows_excludes_exactly_the_data_dependent_ones():
    assert (set(hm.WINDOW_OPTIONS) - set(hm.NAMED_WINDOWS)
            == set(hm.DATA_DEPENDENT_WINDOWS)
            == {hm.WINDOW_CUSTOM, hm.WINDOW_ALL})


def test_all_history_window_raises_from_window_bounds():
    """Its span comes from the DATA, so lcw alone cannot resolve it."""
    with pytest.raises(ValueError, match="all_history_bounds"):
        hm.window_bounds(hm.WINDOW_ALL, LCW)


def test_all_history_bounds_spans_the_whole_snapshot(three_year_frame):
    frame, _ = three_year_frame
    start, end = hm.all_history_bounds(frame, LCW)
    assert start == pd.Timestamp(frame["WeekDate"].min())
    assert end == LCW
    # Every week on record, so it must be the widest window available.
    widest = max(hm.window_totals(frame, *hm.window_bounds(w, LCW))["weeks"]
                 for w in hm.NAMED_WINDOWS)
    assert hm.window_totals(frame, start, end)["weeks"] >= widest


def test_all_history_bounds_ignores_forward_projection_weeks(P):
    """The frame carries 15 weeks of forward projections; they are not history."""
    weeks = _weeks(LCW, 5) + [LCW + pd.Timedelta(weeks=n) for n in (1, 2, 3)]
    frame = hm.build_frame(
        raw_rows("AAA", "C", "US Retail", weeks, pos=[1.0] * len(weeks)),
        P, {"AAA": 1.0},
    )
    start, end = hm.all_history_bounds(frame, LCW)
    assert end == LCW, "must not reach past the last complete week"
    assert hm.window_totals(frame, start, end)["weeks"] == 5


def test_all_history_bounds_is_none_when_there_is_nothing(P):
    assert hm.all_history_bounds(pd.DataFrame(), LCW) is None
    # A frame holding only future weeks has no history to span.
    future = hm.build_frame(
        raw_rows("AAA", "C", "US Retail", [LCW + pd.Timedelta(weeks=2)], pos=[1.0]),
        P, {},
    )
    assert hm.all_history_bounds(future, LCW) is None


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


@pytest.mark.parametrize("window", list(hm.NAMED_WINDOWS))
def test_prior_period_abuts_the_window_without_gap_or_overlap(window):
    """The momentum comparison: the equal-length stretch immediately before.

    Lengths are compared in WEEKS TOTALLED, not in days between endpoints. A rolling
    window's endpoints are both Sundays and so sit (weeks - 1) * 7 days apart, and a
    calendar window's edges fall mid-week — measuring either in raw days hands the
    calendar year a 53-week comparable for its 52 weeks.
    """
    start, end = hm.window_bounds(window, LCW)
    p_start, p_end = hm.prior_period_window(start, end)
    assert p_end == start - pd.Timedelta(days=1), "must end the day before the window"
    assert p_start.weekday() == start.weekday(), "shifted by whole weeks"
    assert hm._anchor_weeks(p_start, p_end) == hm._anchor_weeks(start, end)


def test_prior_period_of_a_four_week_window_is_the_four_weeks_before(three_year_frame):
    frame, _ = three_year_frame
    start, end = hm.window_bounds(hm.WINDOW_4W, LCW)
    p_start, p_end = hm.prior_period_window(start, end)
    assert (p_start, p_end) == (LCW - pd.Timedelta(weeks=7),
                                LCW - pd.Timedelta(weeks=3, days=1))
    assert hm.window_totals(frame, p_start, p_end)["weeks"] == 4


def test_prior_period_and_prior_year_are_different_questions():
    """One is momentum, one is season; nothing may quietly conflate them."""
    start, end = hm.window_bounds(hm.WINDOW_13W, LCW)
    assert hm.prior_period_window(start, end) != hm.prior_year_window(start, end)


@pytest.fixture
def step_up_frame(P):
    """One SKU at 1 unit/wk, except 5 a year ago and 10 in the last 4 weeks.

    ``three_year_frame`` is flat, so a 4-week window there compares 10 units against
    10 units a year earlier and yoy_movers correctly reports nothing moved. This
    fixture gives the short window real movement AND makes each candidate comparison
    period a distinct level, so a test can tell which one was actually measured:

    * weeks[156:160] -- the current 4 weeks           -> 10/wk
    * weeks[104:108] -- the same 4 weeks a year back  ->  5/wk
    * everything else (incl. two years back)          ->  1/wk
    """
    weeks = _weeks(LCW, 160)
    pos = [1.0] * 160
    pos[104:108] = [5.0] * 4
    pos[156:160] = [10.0] * 4
    raw = raw_rows("STEP", "C", "US Retail", weeks, pos=pos)
    return hm.build_frame(raw, P, {"STEP": 1.0})


def test_yoy_movers_follows_the_window_it_is_given(step_up_frame):
    """It used to hardcode the trailing 52 weeks and ignore its caller entirely."""
    short = hm.yoy_movers(step_up_frame, *hm.window_bounds(hm.WINDOW_4W, LCW))
    long = hm.yoy_movers(step_up_frame, *hm.window_bounds(hm.WINDOW_52W, LCW))
    assert not short.empty and not long.empty
    # The step is 4 weeks of +9 units either way, but the 52-week window carries 48
    # flat weeks alongside it, so the two windows cannot report the same current.
    assert short["current"].sum() == pytest.approx(40.0)
    assert long["current"].sum() == pytest.approx(88.0)


def test_yoy_movers_accepts_an_explicit_prior_window(step_up_frame):
    """So the chart can reuse the tiles' comparison instead of deriving a second.

    The Year-to-date window anchors its comparable to the prior January 1 rather than
    364 days back, and only the caller knows that rule applies — hence the parameter.
    """
    start, end = hm.window_bounds(hm.WINDOW_4W, LCW)
    two_years_back = tuple(d - pd.Timedelta(days=364)
                           for d in hm.prior_year_window(start, end))
    explicit = hm.yoy_movers(step_up_frame, start, end, prior=two_years_back)
    default = hm.yoy_movers(step_up_frame, start, end)
    assert not explicit.empty and not default.empty
    assert hm.window_totals(step_up_frame, *two_years_back)["weeks"] == 4
    # One year back the SKU ran at 5/wk, two years back at 1/wk -- so the figure
    # names which window was measured.
    assert default["prior"].sum() == pytest.approx(20.0)
    assert explicit["prior"].sum() == pytest.approx(4.0)


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
    movers = hm.yoy_movers(frame, *hm.window_bounds(hm.WINDOW_52W, LCW), n=5)
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
    movers = hm.yoy_movers(frame, *hm.window_bounds(hm.WINDOW_52W, LCW), n=5)
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
def test_week_of_year_starts_at_one():
    frame = pd.DataFrame({
        "WeekDate": [pd.Timestamp("2026-01-04"), pd.Timestamp("2026-01-11")],
        "revenue": [1.0, 2.0],
    })
    s = hm.weekly_by_year(frame)
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
    (hm.weekly_by_year, ()),
])
def test_chart_helpers_tolerate_an_empty_frame(fn, args):
    assert fn(pd.DataFrame(), *args).empty


# --------------------------------------------------------------------------- #
# Year selection: what the chart tabs pick their periods from                  #
# --------------------------------------------------------------------------- #
def test_weekly_by_year_puts_every_year_on_one_axis(three_year_frame):
    """The overlay's premise: the same point of the calendar lands on one x position.

    Alignment is by DAY OFFSET from each year's own January 1st, not by week number.
    Week 1's Sunday therefore differs between years by up to six days -- January 1st
    falls on a different weekday each year -- and that is correct: it keeps a week
    sitting where it actually falls in the calendar rather than snapping it to a
    shared grid it does not occupy.
    """
    frame, _ = three_year_frame
    out = hm.weekly_by_year(frame)
    assert not out.empty

    # Nothing escapes the reference year, bar a Dec-31 week spilling into January.
    years = set(out["SeasonDate"].dt.year)
    assert years <= {hm._ALIGN_YEAR, hm._ALIGN_YEAR + 1}

    # Every year's week N lands within one week of every other year's week N, which
    # is what makes the curves comparable.
    spread = out.groupby("WeekOfYear")["SeasonDate"].agg(lambda s: s.max() - s.min())
    assert spread.max() <= pd.Timedelta(days=6), (
        f"week numbers drifted apart by {spread.max()} across years"
    )

    # The offset is preserved exactly, which is the actual contract.
    align_start = pd.Timestamp(year=hm._ALIGN_YEAR, month=1, day=1)
    assert ((out["SeasonDate"] - align_start).dt.days >= 0).all()

    # The reference year is never a leap year: Feb 29 is a position no 365-day year
    # could reach, so re-dating into one would misalign every year after February.
    assert not pd.Timestamp(year=hm._ALIGN_YEAR, month=1, day=1).is_leap_year


def test_weekly_by_year_totals_match_weekly_totals(three_year_frame):
    """Re-dating is an axis operation and must not move a single dollar."""
    frame, _ = three_year_frame
    assert hm.weekly_by_year(frame)["revenue"].sum() == pytest.approx(
        hm.weekly_totals(frame)["revenue"].sum()
    )


def test_available_years_is_bounded_by_lcw(three_year_frame):
    """The frame carries forward projection weeks; a year picker must not offer
    a year that exists only in them."""
    frame, _ = three_year_frame
    years = hm.available_years(frame, LCW)
    assert years == sorted(years), "years must come back oldest first"
    assert all(y <= LCW.year for y in years)
    assert hm.available_years(pd.DataFrame()) == []


def test_a_364_day_shift_is_wrong_for_a_calendar_year():
    """Why the movers tab passes an explicit calendar prior instead of the default.

    prior_year_window shifts by 364 days so whole weeks meet whole weeks — correct
    for a rolling window, and wrong for a calendar year: the shifted span runs one
    day INTO the year being measured, so a SKU's January revenue would be counted on
    both sides of its own comparison.
    """
    start = pd.Timestamp("2025-01-01")
    end = pd.Timestamp("2025-12-31")
    shifted_start, shifted_end = hm.prior_year_window(start, end)
    assert shifted_end >= start, (
        "if this ever stops overlapping, the calendar shift in _tab_movers can go"
    )

    # What the tab uses instead: both endpoints back one calendar year, which for a
    # full year is exactly the previous full year and cannot overlap.
    prior = (start - pd.DateOffset(years=1), end - pd.DateOffset(years=1))
    assert prior == (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))
    assert prior[1] < start

    # And for a part-elapsed year it stays like-for-like: same calendar span, one
    # year back, rather than a whole year against a fraction of one.
    partial_end = pd.Timestamp("2026-07-19")
    partial = (pd.Timestamp("2026-01-01") - pd.DateOffset(years=1),
               partial_end - pd.DateOffset(years=1))
    assert partial == (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-07-19"))


def test_calendar_year_bounds_clamps_both_ends(three_year_frame):
    """The current year stops at the last complete week, not at December."""
    frame, _ = three_year_frame
    start, end = hm.calendar_year_bounds(LCW.year, frame, LCW)
    assert start == pd.Timestamp(year=LCW.year, month=1, day=1)
    assert end <= LCW, "the in-progress year must not claim weeks it has no data for"

    # A year with nothing in it resolves to None rather than an empty span.
    assert hm.calendar_year_bounds(1999, frame, LCW) is None

    # A completed year keeps its December, clamped only by the data itself.
    earlier = LCW.year - 1
    bounds = hm.calendar_year_bounds(earlier, frame, LCW)
    assert bounds is not None
    assert bounds[1] <= pd.Timestamp(year=earlier, month=12, day=31)


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
    tile specs can be exercised as pure functions. Three spans, all derived from the
    selected window — see test_metrics_hold_no_span_the_window_did_not_set.
    """
    start, end = hm.window_bounds(window, lcw)
    prior_period = hm.prior_period_window(start, end)
    prior_year = hm.prior_year_window(
        start, end, anchor_to_year_start=(window == hm.WINDOW_YTD))
    return {
        "bounds": (start, end), "window_kind": window,
        "prior_period_bounds": prior_period,
        "prior_year_bounds": prior_year,
        "window": hm.window_totals(frame, start, end),
        "prior_period": hm.window_totals(frame, *prior_period),
        "prior_year": hm.window_totals(frame, *prior_year),
        "breadth": hm.breadth(frame, start, end),
        "concentration": hm.concentration(frame, start, end),
        "coverage": hm.price_coverage(frame, start, end),
    }


def test_metrics_hold_no_span_the_window_did_not_set():
    """The structural guard against the defect this grid was rebuilt to fix.

    Seven of the twelve original tiles read spans derived from ``lcw`` alone — a
    fixed YTD, 13 weeks, 52 weeks — so choosing "Last 4 weeks" moved five tiles and
    left the rest insisting on 52 weeks. If no such key can exist in the metrics
    dict, no tile can read one. Compares the real ``_metrics`` source against this
    module's clone so the two cannot drift apart either.
    """
    import inspect
    from dashboard_app import historical_summary as hs

    source = inspect.getsource(hs._metrics)
    for banned in ("WINDOW_52W", "WINDOW_13W", '"ytd"', '"w52"', '"w13"', '"w4"'):
        assert banned not in source, (
            f"_metrics computes {banned}, which is a span the analysis window did "
            f"not set — the tile reading it will ignore the selector"
        )
    # window_bounds may be called exactly once: to resolve the SELECTED window.
    assert source.count("window_bounds") == 1


@pytest.mark.parametrize("window", list(hm.NAMED_WINDOWS))
def test_no_two_tiles_show_the_same_thing(three_year_frame, window):
    """Two tiles must never be the same number at any window setting.

    Now held structurally rather than by pinning each tile to a fixed span: the
    eight tiles are eight different QUESTIONS about one window (a total, an average,
    a share, four counts), so none can collapse onto another whatever the window.
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


@pytest.mark.parametrize("tile_id", ["revenue", "units", "revenue_per_week",
                                     "active_skus"])
def test_the_analysis_window_actually_changes_the_tiles(three_year_frame, tile_id):
    """The reported complaint, as a test: pick a different window, get a different
    number. A trailing-52-week figure shown while four weeks are selected is the
    exact symptom this must fail on.

    ``top10_share`` is deliberately absent: the fixture has fewer than ten priced
    SKUs, so its share is 100% over any window. It is covered by
    test_metrics_hold_no_span_the_window_did_not_set instead.
    """
    from dashboard_app import historical_summary as hs

    frame, _ = three_year_frame
    short = _metrics_for(frame, LCW, hm.WINDOW_4W)
    # 3 years rather than 52 weeks: the discontinued SKU last sold 80 weeks ago, so
    # only a window that reaches past it changes the assortment counts too.
    long = _metrics_for(frame, LCW, hm.WINDOW_3Y)
    assert hs._tile_value(tile_id, short) != hs._tile_value(tile_id, long), (
        f"{tile_id} reads the same over 4 weeks as over 52 — it is not measured "
        f"over the analysis window"
    )


def test_every_tile_has_a_breakdown(three_year_frame):
    """No tile may be a dead end — the whole point of making them clickable."""
    from dashboard_app import historical_summary as hs

    frame, _ = three_year_frame
    m = _metrics_for(frame, LCW, hm.WINDOW_3Y)
    for tile_id in hs._TILES:
        table, note = hs._breakdown_frame(tile_id, frame, m)
        assert table is not None, f"{tile_id} has no breakdown"
        assert list(table.columns), f"{tile_id} breakdown has no columns"
        assert note, f"{tile_id} breakdown has no explanatory note"


def test_every_breakdown_covers_the_windows_own_weeks(three_year_frame):
    """A modal must describe the period of the tile that opened it.

    The date-bearing breakdowns used to re-derive fixed YTD / 13-week / 52-week
    spans from ``lcw``, so clicking a tile could open a table about a different
    quarter entirely.
    """
    from dashboard_app import historical_summary as hs

    frame, _ = three_year_frame
    m = _metrics_for(frame, LCW, hm.WINDOW_13W)
    start, end = m["bounds"]
    for tile_id in ("revenue_per_week",):
        table, _ = hs._breakdown_frame(tile_id, frame, m)
        weeks = pd.to_datetime(table["Week"])
        assert weeks.min() >= start and weeks.max() <= end, (
            f"{tile_id} breakdown reaches outside the window "
            f"({weeks.min()}–{weeks.max()} vs {start}–{end})"
        )


def test_revenue_per_week_is_the_windows_average(three_year_frame):
    from dashboard_app import historical_summary as hs

    frame, _ = three_year_frame
    m = _metrics_for(frame, LCW, hm.WINDOW_13W)
    assert m["window"]["weeks"] == 13
    expected = m["window"]["revenue"] / 13
    assert hs._TILES["revenue_per_week"]["value"](m) == pytest.approx(expected)


def test_revenue_per_week_is_blank_for_an_empty_window(three_year_frame):
    """No complete week means no average — "—", not a confident $0/week."""
    from dashboard_app import historical_summary as hs

    frame, _ = three_year_frame
    m = _metrics_for(frame.iloc[:0], LCW, hm.WINDOW_13W)
    assert hs._TILES["revenue_per_week"]["value"](m) is None
    assert hs._tile_value("revenue_per_week", m) == "—"


@pytest.mark.parametrize("window", list(hm.NAMED_WINDOWS))
def test_headline_deltas_compare_against_a_stated_window(three_year_frame, window):
    """Every delta's base must be one of the two spans the header names."""
    from dashboard_app import historical_summary as hs

    frame, _ = three_year_frame
    m = _metrics_for(frame, LCW, window)
    for tile_id in ("revenue", "units", "revenue_per_week"):
        # Not asserting a value -- asserting the delta is computable from the two
        # comparison windows in `m` and nothing else, by removing them.
        blank = {"revenue": None, "units": None, "weeks": 0}
        stripped = dict(m, prior_period=blank, prior_year=blank)
        assert hs._TILES[tile_id]["delta"](stripped) is None, (
            f"{tile_id}'s delta survives with both comparison windows blanked, so "
            f"it is reading a period the header never states"
        )


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
