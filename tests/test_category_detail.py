"""The Category detail section and the container-demand tiles.

Category detail is SKU detail drilled one level out: the same seven KPI tiles and
the same chart over a curated product group instead of one SKU. The things worth
pinning here are the three places a rollup one level up can silently go wrong —
averaging containers, summing a SKU-level constant across customer rows, and
averaging a ratio instead of taking the ratio of the totals — plus the membership
list itself, which is hand-curated and therefore the thing most likely to rot.
"""
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from dashboard_app.config import (
    PRICE_COL, RISK_COL, EIGHT_WK_AVG_COL, ONHAND_COL, WOS_COL,
    CONTAINER_HIST_COL, CONTAINER_FC_COL, PRODUCT_CATEGORIES, KPI_ORDER, KPI_HELP,
)
from dashboard_app.kpis import (
    _weekly_containers, _container_breakdown, _category_members,
    _sheet_slug, _file_slug, _range_label, _history_window_from_range,
)

# The historical container TILE is labelled dynamically after the Date range preset,
# so it is not keyed by CONTAINER_HIST_COL on screen — that constant stays the stable
# header for the listing column and the Excel export. The picker defaults to
# "6 Months" (charts.RANGE_PRESET_DEFAULT).
HIST_TILE = "Container Demand (6-Month avg/wk)"
HIST_TILE_1M = "Container Demand (1-Month avg/wk)"

# Three real members of the curated Liners set, so these tests exercise the
# shipped list rather than a monkeypatched stand-in.
LINER_A, LINER_B, LINER_C = "CW0160", "CW0161", "CW0162"
# A step can WITH a liner pocket, and a hinge-block part that names one in its
# description: the two shapes the curation rule exists to exclude.
NOT_LINER_CAN, NOT_LINER_PART = "CW2023", "PD6274"


# --------------------------------------------------------------------------- #
# _weekly_containers                                                          #
# --------------------------------------------------------------------------- #
def _two_sku_frame():
    """A sells every week; B sells in week 1 only — the short-span case."""
    wk = pd.date_range("2026-01-05", periods=3, freq="W-MON")
    return pd.DataFrame(
        [{"SKU": "A", "WeekDate": w, "demand": 100.0} for w in wk]
        + [{"SKU": "B", "WeekDate": wk[0], "demand": 60.0}]
    )


def test_containers_average_weekly_totals_not_per_sku_averages():
    """Per-week sum THEN mean — not the sum of each SKU's own average.

    The distinction is the whole reason the helper exists. B sells 60 units (3
    containers) in one week of three. Averaging the weekly totals spreads that over
    the window: (13 + 10 + 10) / 3 = 11. Summing per-SKU averages instead divides B
    by its own single week and then adds it as though it recurred weekly: 10 + 3 =
    13, an 18% overstatement of the containers a planner would book.

    This mirrors the note in ``_render_kpis`` about the Total Weekly Demand tile —
    the container tile has to use the same convention or the two tiles beside each
    other would describe different windows.
    """
    value, covered, total = _weekly_containers(
        _two_sku_frame(), "demand", pd.Series({"A": 10.0, "B": 20.0})
    )
    assert value == pytest.approx(11.0)
    assert (covered, total) == (2, 2)
    # The wrong answer, spelled out so a future refactor can't quietly adopt it.
    assert value != pytest.approx(100 / 10 + 60 / 20)


def test_containers_divide_per_sku_before_summing():
    """A container holds a different number of every SKU, so there is no
    category-level load to divide a category total by.

    Same units, different loads: dividing the summed units by either load (or by
    their mean) gives a different answer than dividing each SKU by its own.
    """
    wk = pd.Timestamp("2026-01-05")
    frame = pd.DataFrame([
        {"SKU": "A", "WeekDate": wk, "demand": 100.0},
        {"SKU": "B", "WeekDate": wk, "demand": 100.0},
    ])
    value, _, _ = _weekly_containers(
        frame, "demand", pd.Series({"A": 10.0, "B": 100.0}))
    assert value == pytest.approx(10.0 + 1.0)
    # Total units ÷ mean load would be 200 / 55 ≈ 3.6 — not this.
    assert value != pytest.approx(200 / 55)


def test_containers_skip_skus_with_no_load_and_report_coverage():
    """A SKU missing from the Plytix export drops out, exactly as an unpriced SKU
    drops out of Revenue Risk — and the caller is told how many, so a partial
    figure is never presented as a complete one."""
    value, covered, total = _weekly_containers(
        _two_sku_frame(), "demand", pd.Series({"A": 10.0})
    )
    assert value == pytest.approx(10.0), "A only: 10 containers in each of 3 weeks"
    assert (covered, total) == (1, 2)


def test_containers_return_none_rather_than_zero_when_unavailable():
    """"No Container Load on file" must not render as "zero containers"."""
    frame = _two_sku_frame()
    assert _weekly_containers(frame, "demand", None) == (None, 0, 0)
    assert _weekly_containers(frame, "demand", pd.Series(dtype=float)) == (None, 0, 0)
    assert _weekly_containers(None, "demand", pd.Series({"A": 10.0})) == (None, 0, 0)
    # A load of zero would divide to inf and swamp the weekly sum.
    value, covered, total = _weekly_containers(
        frame, "demand", pd.Series({"A": 10.0, "B": 0.0}))
    assert value == pytest.approx(10.0)
    assert (covered, total) == (1, 2)


def test_containers_accept_a_plain_dict():
    """container_load_from_plytix returns a Series, but the helper is also handed
    dicts by tests and by callers that build a map inline."""
    value, _, _ = _weekly_containers(_two_sku_frame(), "demand", {"A": 10.0, "B": 20.0})
    assert value == pytest.approx(11.0)


# --------------------------------------------------------------------------- #
# The curated membership list                                                 #
# --------------------------------------------------------------------------- #
def test_liners_category_excludes_cans_and_parts():
    """The curation rule's whole job: "liners" the product, not every SKU whose
    description happens to contain the word.

    ``CW2023`` is a step can WITH a liner pocket and ``PD6274`` a hinge block for a
    liner-rim recycler. Both match a naive substring search on the description and
    neither is a liner.
    """
    liners = PRODUCT_CATEGORIES["Liners"]
    assert len(liners) == 76
    assert {LINER_A, LINER_B, LINER_C} <= liners
    assert NOT_LINER_CAN not in liners
    assert NOT_LINER_PART not in liners


def test_category_members_intersect_with_what_the_view_carries():
    """The curated list is snapshot-independent; a view is not. Offering members
    the loaded snapshot has no rows for would put empty SKUs in the sub-list and a
    misleading count in the dropdown label."""
    summary = pd.DataFrame({"SKU": [LINER_A, LINER_B, NOT_LINER_CAN, "SKU-Z"]})
    assert _category_members(summary, "Liners") == [LINER_A, LINER_B]
    assert _category_members(summary, "Nonexistent") == []


def test_container_columns_are_registered_for_ordering_and_tooltips():
    """Every KPI field carries one canonical position and one tooltip; a name that
    misses ``KPI_ORDER`` sorts to the end of every card it appears on."""
    for col in (CONTAINER_HIST_COL, CONTAINER_FC_COL):
        assert col in KPI_ORDER
        assert col in KPI_HELP


# --------------------------------------------------------------------------- #
# The rendered section                                                        #
# --------------------------------------------------------------------------- #
def _frames(onhand=True):
    """A SKU x customer frame in the shape Optimized passes: three liners plus a
    non-liner, each sold to two customer groups."""
    rows = []
    for sku, eight in ((LINER_A, 10.0), (LINER_B, 20.0), (LINER_C, 30.0),
                       (NOT_LINER_CAN, 99.0)):
        for cust in ("AMAZON-DC", "ACR"):
            rows.append({
                "SKU": sku, "Description": f"{sku} desc", "Customer Grouping": cust,
                "Data Source": "POS", "Weeks with data": 12,
                EIGHT_WK_AVG_COL: eight,
                "Current Projection Average": eight,
                "Updated Projection Average": eight + 1.0,
            })
    df = pd.DataFrame(rows)
    df["Projection Difference"] = (
        df["Updated Projection Average"] - df["Current Projection Average"])
    df[PRICE_COL] = 5.0
    df[RISK_COL] = df["Projection Difference"] * df[PRICE_COL]
    if onhand:
        # A SKU-level constant, repeated on every one of that SKU's customer rows —
        # which is exactly what makes a naive sum over-count it.
        # Deliberately uneven: these make the ratio of the totals (10.0) differ
        # from the mean of the per-SKU ratios (8.3), so the WOS test below can
        # tell the two formulas apart.
        df[ONHAND_COL] = df["SKU"].map(
            {LINER_A: 100.0, LINER_B: 200.0, LINER_C: 900.0, NOT_LINER_CAN: 999.0})

    hist = pd.date_range("2026-01-05", periods=12, freq="W-MON")
    fcst = pd.date_range("2026-04-06", periods=4, freq="W-MON")
    skus = [LINER_A, LINER_B, LINER_C, NOT_LINER_CAN]
    agg = pd.DataFrame([
        {"SKU": s, "WeekDate": w, "POS": 10.0, "Orders": None, "Projection": 9.0,
         "demand": 10.0}
        for s in skus for w in hist
    ])
    weekly = pd.DataFrame([
        {"SKU": s, "WeekDate": w, "projected_pos": 11.0}
        for s in skus for w in fcst
    ])
    anchors = (pd.Timestamp("2026-01-05"), pd.Timestamp("2026-03-30"),
               pd.Timestamp("2026-04-06"))
    return df, agg, weekly, anchors


def _category_app():
    # AppTest.from_function ships only this function's source to a temp
    # script, so module-level names have to be re-imported here.
    from dashboard_app.kpis import render_category_detail_section
    from test_category_detail import _frames, LINER_A, LINER_B, LINER_C
    from test_category_detail import NOT_LINER_CAN
    import pandas as pd
    df, agg, weekly, anchors = _frames()
    render_category_detail_section(
        df, agg, weekly, df, anchors, None,
        container_load=pd.Series({LINER_A: 10.0, LINER_B: 20.0, LINER_C: 40.0,
                                  NOT_LINER_CAN: 1.0}),
        key="best",
    )


def _run(fn):
    at = AppTest.from_function(fn, default_timeout=60).run()
    assert not at.exception, at.exception
    return at


def test_category_tiles_total_only_the_categorys_skus():
    """The tiles are the sum over the category's members and nothing else.

    ``NOT_LINER_CAN`` carries a deliberately huge 99.0 so that if the slice ever
    leaked it would be impossible to miss.
    """
    at = _run(_category_app)
    assert at.selectbox(key="best_category").value == "Liners"

    tiles = {m.label: m.value for m in at.metric}
    # Each liner has TWO customer rows, and _render_kpis sums them — which is the
    # roll-up this section exists to show. Updated (11 + 21 + 31) x 2 = 126;
    # current (10 + 20 + 30) x 2 = 120.
    assert tiles["Updated Forecast (avg/wk)"] == "126"
    assert tiles["Current Forecast (avg/wk)"] == "120"
    assert tiles["Projection Difference (avg/wk)"] == "+6"
    assert tiles["SKUs Forecasted"] == "3", "the can is not a liner"
    # Each of the 3 liners contributes 10 units/wk of demand -> 30 a week.
    assert tiles["Total Weekly Demand (8-Week avg)"] == "30"
    # Revenue risk = (1 + 1 + 1) x 2 customers x $5.
    assert tiles["Revenue Risk (avg/wk)"] == "+$30"


def test_category_on_hand_counts_each_sku_once():
    """On Hand is a SKU-level constant repeated across a SKU's customer rows.

    Summing the rows straight would count each SKU once per customer group — here
    2x, and on the live data up to 91x. The section de-duplicates on SKU first, the
    same guard ``exceptions._sum_distinct_skus`` exists for.
    """
    at = _run(_category_app)
    tiles = {m.label: m.value for m in at.metric}
    assert tiles[ONHAND_COL] == "1,200", "100 + 200 + 900, not doubled to 2,400"


def test_category_wos_is_the_ratio_of_totals_not_the_mean_of_ratios():
    """A mean of ratios is not the ratio of the totals, and it is the latter that
    answers "how long does this category's stock last"."""
    at = _run(_category_app)
    tiles = {m.label: m.value for m in at.metric}
    # Ratio of the totals: 1,200 on hand / 120 current weekly = 10.0 weeks.
    assert tiles[WOS_COL] == "10.0"
    # The mean of the per-SKU ratios is a different number — 100/20, 200/40, 900/60
    # = 5, 5, 15, averaging 8.3. Asserting it is NOT that is the point of the test.
    assert tiles[WOS_COL] != "8.3"


def test_category_container_tiles():
    """Both tenses, each per-SKU divided then summed per week.

    Historical: 10 units/wk each ÷ loads 10, 20, 40 = 1 + 0.5 + 0.25 = 1.75.
    Forecast:   11 units/wk each ÷ the same loads = 1.1 + 0.55 + 0.275 = 1.925.
    """
    at = _run(_category_app)
    tiles = {m.label: m.value for m in at.metric}
    assert tiles[HIST_TILE] == "1.75"
    assert tiles[CONTAINER_FC_COL] == "1.93"


def _no_load_app():
    # AppTest.from_function ships only this function's source to a temp
    # script, so module-level names have to be re-imported here.
    from dashboard_app.kpis import render_category_detail_section
    from test_category_detail import _frames
    df, agg, weekly, anchors = _frames()
    render_category_detail_section(df, agg, weekly, df, anchors, None,
                                   container_load=None, key="best")


def test_container_tiles_show_an_em_dash_without_plytix():
    """Unknown must not read as zero — the same contract Revenue Risk keeps when
    no list prices are loaded."""
    at = _run(_no_load_app)
    tiles = {m.label: m.value for m in at.metric}
    assert tiles[HIST_TILE] == "—"
    assert tiles[CONTAINER_FC_COL] == "—"


def _no_members_app():
    # AppTest.from_function ships only this function's source to a temp
    # script, so module-level names have to be re-imported here.
    from dashboard_app.kpis import render_category_detail_section
    from test_category_detail import _frames, NOT_LINER_CAN
    df, agg, weekly, anchors = _frames()
    # A view carrying no liners at all.
    df = df[df["SKU"] == NOT_LINER_CAN]
    agg = agg[agg["SKU"] == NOT_LINER_CAN]
    weekly = weekly[weekly["SKU"] == NOT_LINER_CAN]
    render_category_detail_section(df, agg, weekly, df, anchors, None, key="best")


def test_no_category_in_view_says_so_instead_of_raising():
    """A regional view can legitimately carry none of a category's SKUs."""
    at = _run(_no_members_app)
    assert not at.selectbox, "no dropdown when there is nothing to pick"
    assert any("No product category" in c.value for c in at.caption)


def _no_onhand_app():
    # AppTest.from_function ships only this function's source to a temp
    # script, so module-level names have to be re-imported here.
    from dashboard_app.kpis import render_category_detail_section
    from test_category_detail import _frames
    df, agg, weekly, anchors = _frames(onhand=False)
    render_category_detail_section(df, agg, weekly, df, anchors, None, key="best")


def test_missing_warehouse_snapshot_omits_supply_tiles():
    """Absent, not zero-filled: "unknown stock" must never render as 0 on hand."""
    at = _run(_no_onhand_app)
    labels = {m.label for m in at.metric}
    assert ONHAND_COL not in labels
    assert WOS_COL not in labels
    assert "Updated Forecast (avg/wk)" in labels, "the rest of the section still renders"


# --------------------------------------------------------------------------- #
# SKU detail keeps its container tiles too                                    #
# --------------------------------------------------------------------------- #
def _sku_app():
    # AppTest.from_function ships only this function's source to a temp
    # script, so module-level names have to be re-imported here.
    from dashboard_app.kpis import render_sku_detail_section
    from test_category_detail import _frames, LINER_A
    import pandas as pd
    df, agg, weekly, anchors = _frames()
    render_sku_detail_section(
        df, agg, weekly, df, anchors, None,
        container_load=pd.Series({LINER_A: 10.0}), key="quick",
    )


def test_sku_detail_gains_container_tiles():
    """The same two tiles, same helper, scoped to one SKU: 10 units/wk ÷ a load of
    10 = 1.0 container, and 11 forecast units = 1.1."""
    at = _run(_sku_app)
    tiles = {m.label: m.value for m in at.metric}
    assert tiles[HIST_TILE] == "1.00"
    assert tiles[CONTAINER_FC_COL] == "1.10"


# --------------------------------------------------------------------------- #
# _container_breakdown: the parts must add up to the whole                    #
# --------------------------------------------------------------------------- #
def test_per_sku_containers_sum_to_the_total():
    """The per-SKU column has to add up to the tile above it, or the table reads
    as broken.

    B sells 60 units (3 containers) in one week of three. The obvious per-SKU
    figure divides B by its own single week and reports 3.00 — but the total
    divides every SKU by all three weeks, so a column of those would sum to 13
    beside a tile reading 11. Sharing one denominator makes B contribute 1.00 and
    the column sum the tile exactly.
    """
    total, per_sku, covered, n = _container_breakdown(
        _two_sku_frame(), "demand", pd.Series({"A": 10.0, "B": 20.0})
    )
    assert total == pytest.approx(11.0)
    assert per_sku.sum() == pytest.approx(total), "the parts ARE the whole"
    assert per_sku["A"] == pytest.approx(10.0)
    assert per_sku["B"] == pytest.approx(1.0), "spread over 3 weeks, not 1"
    assert per_sku["B"] != pytest.approx(60 / 20), "the naive own-span figure"
    assert (covered, n) == (2, 2)


def test_weekly_containers_is_the_total_half_of_the_breakdown():
    """The wrapper must not become a second path to the number."""
    frame, cl = _two_sku_frame(), pd.Series({"A": 10.0, "B": 20.0})
    total, _, covered, n = _container_breakdown(frame, "demand", cl)
    assert _weekly_containers(frame, "demand", cl) == (total, covered, n)


def test_breakdown_returns_an_empty_series_not_zeros_when_unavailable():
    """A SKU with no Container Load must be blank in the column, never 0.00 —
    "sits out the tile" and "demands no containers" are different claims."""
    frame = _two_sku_frame()
    total, per_sku, _, _ = _container_breakdown(frame, "demand", pd.Series({"A": 10.0}))
    assert total == pytest.approx(10.0)
    assert "B" not in per_sku.index, "B drops out rather than contributing 0"
    assert per_sku.sum() == pytest.approx(total)

    total, per_sku, covered, n = _container_breakdown(frame, "demand", None)
    assert total is None and per_sku.empty and (covered, n) == (0, 0)


# --------------------------------------------------------------------------- #
# The listing table: container columns + export                               #
# --------------------------------------------------------------------------- #
def test_listing_container_columns_sum_to_the_tiles():
    """What the user actually reads: open the expander, add the column up, and get
    the number on the tile beside the chart."""
    at = _run(_category_app)
    tiles = {m.label: m.value for m in at.metric}

    assert len(at.dataframe) == 1, "one listing table in the section"
    listing = at.dataframe[0].value
    for col in (CONTAINER_HIST_COL, CONTAINER_FC_COL):
        assert col in listing.columns
    # Same fixture as test_category_container_tiles: 1.75 hist, 1.93 forecast.
    assert listing[CONTAINER_HIST_COL].sum() == pytest.approx(1.75)
    assert listing[CONTAINER_FC_COL].sum() == pytest.approx(1.925)
    # The COLUMN header is the stable constant; the TILE names the selected range.
    assert tiles[HIST_TILE] == "1.75"
    assert tiles[CONTAINER_FC_COL] == "1.93"
    # Only the category's SKUs, and each once.
    assert sorted(listing["SKU"]) == sorted([LINER_A, LINER_B, LINER_C])


def test_listing_omits_container_columns_without_plytix():
    """No Container Load map -> no container columns at all, rather than a column
    of em dashes pretending to be data."""
    at = _run(_no_load_app)
    listing = at.dataframe[0].value
    assert CONTAINER_HIST_COL not in listing.columns
    assert CONTAINER_FC_COL not in listing.columns
    assert "Container Load" not in listing.columns
    assert "SKU" in listing.columns, "the listing itself still renders"


def test_listing_is_downloadable_through_the_shared_export_path():
    """Every export in the app goes through with_export_flags -> summary_to_excel.

    AppTest does not surface download buttons as elements, so this is checked at
    the source — the same way test_phase5_dashboard checks its exports.
    """
    import inspect

    from dashboard_app.kpis import render_category_detail_section

    src = inspect.getsource(render_category_detail_section)
    assert "st.download_button(" in src
    assert "summary_to_excel(with_export_flags(listing)" in src
    assert 'key=f"dl_category_skus_{key}"' in src, (
        "namespaced by key — the section renders on both Quick and Optimized"
    )


def test_export_slugs_are_legal():
    """Excel caps sheet names at 31 chars and rejects []:*?/\\ ."""
    assert _sheet_slug("Liners") == "Liners"
    assert _file_slug("Liners") == "liners"
    long_bad = _sheet_slug("A/B: a very very very long category name")
    assert len(long_bad) <= 31
    assert not set(long_bad) & set('[]:*?/\\')
    assert " " not in _file_slug("Trash Cans / Bins")


# --------------------------------------------------------------------------- #
# Section order on the Quick page                                             #
# --------------------------------------------------------------------------- #
def test_by_sku_table_sits_above_category_detail():
    """`Summary table by SKU (view total)` is the table form of what SKU detail
    just charted — one row per SKU at the view total — so the two stay adjacent.

    Category detail is the step out to the next grain up, which puts it after that
    table and immediately before the main by-customer table. Optimized Projections
    reads the same way: Category detail is the last section before its summary
    table.
    """
    import inspect

    import dashboard

    src = inspect.getsource(dashboard.main)
    sku_detail = src.index("render_sku_detail_section(")
    by_sku = src.index('st.expander("Summary table by SKU (view total)")')
    category = src.index("render_category_detail_section(")
    by_customer = src.index('st.markdown("### Summary table by SKU and customer")')
    assert sku_detail < by_sku < category < by_customer


# --------------------------------------------------------------------------- #
# The Date range selector drives the historical container tile — and ONLY it  #
# --------------------------------------------------------------------------- #
def _ranged_frames():
    """Like ``_frames`` but with demand that CHANGES over the window.

    The stock fixture sells a flat 10 units every week, so narrowing the window
    leaves every average identical and a range test would pass without the feature
    existing. Here the last four weeks run 10x the first eight, so a 1-Month window
    and a 6-Month window cannot agree.
    """
    df, agg, weekly, anchors = _frames()
    cutoff = pd.Timestamp("2026-03-02")
    agg = agg.copy()
    late = pd.to_datetime(agg["WeekDate"]) >= cutoff
    agg.loc[late, ["POS", "demand"]] = 100.0
    return df, agg, weekly, anchors


def _ranged_app():
    # AppTest.from_function ships only this function's source to a temp
    # script, so module-level names have to be re-imported here.
    from dashboard_app.kpis import render_category_detail_section
    from test_category_detail import _ranged_frames, LINER_A, LINER_B, LINER_C
    from test_category_detail import NOT_LINER_CAN
    import pandas as pd
    df, agg, weekly, anchors = _ranged_frames()
    render_category_detail_section(
        df, agg, weekly, df, anchors, None,
        container_load=pd.Series({LINER_A: 10.0, LINER_B: 20.0, LINER_C: 40.0,
                                  NOT_LINER_CAN: 1.0}),
        key="best",
    )


def test_narrowing_the_date_range_moves_only_the_historical_container_tile():
    """The whole agreed rule, in one test.

    The Date range selector drives historical KPIs — but not those whose label
    already names a fixed window. So exactly one tile may move: the historical
    container demand. "Total Weekly Demand (8-Week avg)" says 8-Week and therefore
    stays an 8-week average; the forecast tiles have no historical input at all; On
    Hand and WOS are point-in-time.
    """
    at = _run(_ranged_app)
    before = {m.label: m.value for m in at.metric}
    assert HIST_TILE in before, "labelled with the default preset"

    at = at.selectbox(key="range_cat_best_preset").set_value("1 Month").run()
    assert not at.exception, at.exception
    after = {m.label: m.value for m in at.metric}

    # The tile moved, and its label followed the preset.
    assert HIST_TILE_1M in after
    assert HIST_TILE not in after
    assert after[HIST_TILE_1M] != before[HIST_TILE], (
        "a shorter window over rising demand must give a different average"
    )

    # Everything else is untouched. This is the half of the rule that is easy to
    # break by accident.
    for label in ("Total Weekly Demand (8-Week avg)", "Current Forecast (avg/wk)",
                  "Updated Forecast (avg/wk)", "Projection Difference (avg/wk)",
                  "Revenue Risk (avg/wk)", "Projected Revenue (avg/wk)",
                  "SKUs Forecasted", ONHAND_COL, WOS_COL, CONTAINER_FC_COL):
        assert label in before, f"{label} missing from the fixture"
        assert after[label] == before[label], f"{label} must not follow the range"


def test_the_per_sku_column_still_ties_after_a_range_change():
    """The tie has to hold at EVERY window, not just the default one — the listing
    and the tile must be sliced from the same frame."""
    at = _run(_ranged_app)
    at = at.selectbox(key="range_cat_best_preset").set_value("1 Month").run()
    assert not at.exception, at.exception

    tiles = {m.label: m.value for m in at.metric}
    listing = at.dataframe[0].value
    assert listing[CONTAINER_HIST_COL].sum() == pytest.approx(
        float(tiles[HIST_TILE_1M]), abs=0.005
    ), "column sum must still equal the tile after narrowing the range"


def test_history_window_clamps_the_pickers_forecast_end():
    """``chart_range_control`` returns the FORECAST horizon as its end in every
    branch — presets trim history only. Passing that through unclamped would pull
    forecast weeks into a historical average."""
    anchors = (pd.Timestamp("2026-01-05"), pd.Timestamp("2026-08-30"),
               pd.Timestamp("2026-09-06"))
    picker = (pd.Timestamp("2026-03-01"), pd.Timestamp("2026-12-13"))  # end = horizon
    start, end, ffw = _history_window_from_range(picker, anchors)
    assert start == pd.Timestamp("2026-03-01")
    assert end == anchors[1], "clamped back to the last completed week"
    assert ffw == anchors[2]
    # No range at all -> the model's window, unchanged.
    assert _history_window_from_range(None, anchors) == anchors


def test_range_labels_read_as_windows():
    assert _range_label("6 Months") == "6-Month"
    assert _range_label("1 Year") == "1-Year"
    assert _range_label("All") == "All-Time"
    assert _range_label("Custom\u2026") == "custom range"


def test_export_header_is_stable_across_range_changes():
    """The tile label moves with the picker; the column header must not, or a
    downloaded workbook would have a different schema depending on what the user
    happened to have selected."""
    at = _run(_ranged_app)
    assert CONTAINER_HIST_COL in at.dataframe[0].value.columns
    at = at.selectbox(key="range_cat_best_preset").set_value("1 Month").run()
    assert not at.exception, at.exception
    assert CONTAINER_HIST_COL in at.dataframe[0].value.columns
    assert HIST_TILE_1M not in at.dataframe[0].value.columns

