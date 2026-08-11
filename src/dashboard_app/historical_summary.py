"""The Historical Summary view: what sell-through has already done.

The one view in this app with no forecast in it. Every other view exists to
produce or interrogate the 15-week projection; this one answers the descriptive
questions a planner asks before trusting any projection -- how much revenue is
booked this year, whether the quarter is up or down on last year, which SKUs carry
the business, and when the season lands.

Three deliberate differences from the projection views, each captioned on screen
so a planner never has to wonder why two tabs disagree:

1. It reads ``ExclusionResult.df_with_discontinued``, so a SKU that sold for years
   before being retired still counts for those years. Its totals therefore run
   HIGHER than the projection views', which correctly drop those SKUs.
2. Revenue is units x Plytix list price -- a retail-value proxy, never invoiced
   revenue -- and unpriced SKUs are left out of revenue rather than counted as $0.
3. POS-vs-Orders is decided over the whole history rather than a trailing window.

All arithmetic lives in ``historical_metrics`` (streamlit-free, unit-tested) and
all figures in ``historical_charts``; this module is layout, caching and captions.
"""
import pandas as pd
import streamlit as st

from dashboard_app import historical_charts as hc
from dashboard_app import historical_metrics as hm
from dashboard_app.charts import _as_price_map
from dashboard_app.config import bounded_put, fmt_compact, fmt_dollar
from dashboard_app.compute import with_export_flags
from dashboard_app.keyskus import mark_key_sku, sku_chip_column_config
from dashboard_app.tables import render_filtered_table

# Cap on the per-filter derived-metric cache (same bounded_put semantics the
# forecast/spike caches use, so one session cannot grow it without limit).
_CACHE_CAP = 12

def _measure_word(value_col):
    """"revenue" / "units" for chart titles, from the displayed measure column."""
    return "revenue" if value_col == "revenue" else "units"


def _signature(df, price_map, plytix_df, today_str):
    """Structural key for the enriched frame -- cheap len()-markers in the style of
    the exclusion/bestmix signatures elsewhere in the app."""
    return (
        today_str,
        0 if df is None else int(len(df)),
        0 if not price_map else int(len(price_map)),
        0 if plytix_df is None else int(len(plytix_df)),
    )


def _base_frame(df, P, price_map, plytix_df, today_str):
    """The enriched weekly frame, memoised in session state.

    This is the heaviest aggregation in the app -- every customer group, every SKU,
    three years, discontinued SKUs included -- so it is built once per snapshot and
    reused across filter changes, tab switches and view switches.
    """
    sig = _signature(df, price_map, plytix_df, today_str)
    if st.session_state.get("hist_base_sig") != sig:
        with st.spinner("Summarising sell-through history…"):
            st.session_state["hist_base"] = hm.build_frame(
                df, P, price_map, plytix_df=plytix_df
            )
        st.session_state["hist_base_sig"] = sig
    return st.session_state["hist_base"]


def _filter_bar(base, lcw):
    """Region / Customer Grouping multi-selects + the analysis window.

    Region narrows the Customer Grouping options, mirroring the Quick Projections
    Region -> group cascade, so the second control never offers a group the first
    has already excluded. An empty multi-select means "no filter" rather than
    "nothing", which is how these read to a planner.

    There is deliberately no SKU-type filter: planners don't slice by it. The
    ``SKU Type`` column itself stays on the frame -- the Mix & Breakdown tab's
    revenue-share donut reads it.

    Returns ``(regions, groups, window, bounds)`` where ``bounds`` is None for a
    named window, or the snapped ``(start, end)`` pair for a custom range.
    """
    c1, c2, c3 = st.columns([1, 1.4, 1.1])

    regions = sorted(base["Region"].dropna().astype(str).unique())
    picked_regions = c1.multiselect(
        "Region", regions, key="hist_region",
        help="Fulfilment region, derived from the customer group. "
             "Leave empty for all regions.",
    )

    scoped = base[base["Region"].astype(str).isin(picked_regions)] \
        if picked_regions else base
    groups = sorted(scoped["Customer Grouping"].dropna().astype(str).unique())
    picked_groups = c2.multiselect(
        "Customer group", groups, key="hist_group",
        help="Options follow the selected region(s). Leave empty for all groups.",
    )

    window = c3.selectbox(
        "Analysis window", list(hm.WINDOW_OPTIONS), index=0, key="hist_window",
        help="The period EVERY tile is measured over. It does not move the charts: "
             "the chart tabs each carry their own year selection, so you can read a "
             "chart over a different period without repointing the tiles.",
    )

    # Two windows have no span derivable from lcw alone, so they resolve here and
    # travel to _metrics as an explicit bounds override.
    bounds = None
    if window == hm.WINDOW_CUSTOM:
        bounds = _custom_range(base, lcw)
    elif window == hm.WINDOW_ALL:
        bounds = hm.all_history_bounds(base, lcw)
    return picked_regions, picked_groups, window, bounds


def _custom_range(base, lcw):
    """Date picker for a custom analysis window, snapped to whole weeks.

    The only free date picker left in this view, and it belongs to the TILES. The
    chart tabs used to carry one each; they now pick calendar years instead, because
    that is the unit their figures compare. Note the guard below: ``st.date_input``
    returns a SINGLE date mid-selection, so a range must only be applied once both
    ends have been chosen.

    Returns the snapped ``(start, end)``, or None when the pick is incomplete or too
    narrow to hold a complete week (the caller then falls back and warns).
    """
    data_min = pd.to_datetime(base["WeekDate"]).min()
    # The pickable maximum is the SATURDAY that closes the last complete week, not
    # lcw itself (which is that week's Sunday). snap_window only counts a week whose
    # whole span falls inside the range, so offering lcw as the maximum would make
    # picking the maximum silently drop the most recent week.
    last_day = pd.Timestamp(lcw) + pd.Timedelta(days=6)
    default_start = max(data_min, pd.Timestamp(lcw) - pd.DateOffset(months=3))
    picked = st.date_input(
        "Custom range",
        value=(default_start.date(), last_day.date()),
        min_value=data_min.date(), max_value=last_day.date(),
        key="hist_custom_range",
        help="Click the calendar or type dates. Snapped to whole weeks — sell-through "
             "is recorded weekly, so a part-week would distort the totals.",
    )
    if not (isinstance(picked, (tuple, list)) and len(picked) == 2):
        return None
    return hm.snap_window(picked[0], picked[1], lcw)


def _apply_filters(base, regions, groups):
    out = base
    if regions:
        out = out[out["Region"].astype(str).isin(regions)]
    if groups:
        out = out[out["Customer Grouping"].astype(str).isin(groups)]
    return out


def _metrics(frame, window, lcw, cache_key, bounds=None, floor=None):
    """Every figure the tiles need, all measured over the SELECTED window.

    ``bounds`` overrides the named window's own span — that is how a custom range
    (already snapped to whole weeks) gets in.

    Exactly three spans exist here: the window, the equal-length period before it
    (momentum) and the same weeks a year earlier (season). There are deliberately
    NO spans derived from ``lcw`` alone -- tiles used to read fixed YTD / 13-week /
    52-week windows, which is why changing the selector left most of the grid
    sitting still. Those periods are still reachable: they are window OPTIONS, and
    Year to date is the default.

    ``floor`` is the earliest week the SNAPSHOT holds, read off the unfiltered
    frame. Every span carries its own coverage against it (see
    ``hm.window_totals``) so a tile can tell "there is no data back there" apart
    from "nothing sold back there" and withhold a delta for the first. It must come
    from the unfiltered frame: measured on the filtered one, narrowing to a customer
    who started trading last year would look like a snapshot that starts last year,
    and their genuine growth would be suppressed as uncomparable.
    """
    store = st.session_state.setdefault("hist_metrics", {})
    if cache_key in store:
        return store[cache_key]

    start, end = bounds if bounds is not None else hm.window_bounds(window, lcw)
    prior_start, prior_end = hm.prior_period_window(start, end)
    # The one place the window's KIND still matters: Year to date must compare
    # against the prior year FROM JANUARY 1, not against the 364 days before this
    # January 1. Every other window is a like-for-like 364-day shift.
    ly_start, ly_end = hm.prior_year_window(
        start, end, anchor_to_year_start=(window == hm.WINDOW_YTD))

    result = {
        "bounds": (start, end),
        # Carried so _render_window_dates can name the window and the breakdowns can
        # reuse the tiles' own comparison rather than re-deriving it.
        "window_kind": window,
        "prior_period_bounds": (prior_start, prior_end),
        "prior_year_bounds": (ly_start, ly_end),
        "window": hm.window_totals(frame, start, end, floor, lcw),
        # Both comparison spans can reach past the floor of the snapshot, and when
        # they do they hold FEWER weeks than the window they are compared against --
        # so a total-against-total delta reads the snapshot's depth as growth. "All
        # history" was the worst case (Revenue +36.7% against an overlapping subset
        # of itself); "Last 3 years" was the most believable, and so the more
        # dangerous, at Units +11.0% where the per-week truth was -4.0%. Each dict
        # carries `fully_covered`; `_total_delta` and `_per_week_delta` refuse to
        # divide without it.
        "prior_period": hm.window_totals(frame, prior_start, prior_end, floor, lcw),
        "prior_year": hm.window_totals(frame, ly_start, ly_end, floor, lcw),
        "breadth": hm.breadth(frame, start, end, floor),
        "concentration": hm.concentration(frame, start, end, n=10),
        "coverage": hm.price_coverage(frame, start, end),
        # Named by _render_window_dates when a comparison span falls short of it, so
        # a withheld delta reads as a known limit of the snapshot rather than as a
        # broken tile.
        "floor": floor,
    }
    bounded_put(store, cache_key, result, _CACHE_CAP)
    return result


# --------------------------------------------------------------------------- #
# KPI tiles                                                                    #
# --------------------------------------------------------------------------- #
def _delta(pct):
    """Streamlit delta string, or None so the tile renders without an arrow.

    None (not "0%") when there is no prior-year base: an absent comparison must
    look absent, not flat.
    """
    return None if pct is None else f"{pct:+.1f}%"


# --------------------------------------------------------------------------- #
# The tile grid                                                                #
# --------------------------------------------------------------------------- #
# One spec table drives BOTH the grid and the modals, so a tile can never open the
# wrong breakdown and a new KPI cannot be added to one without the other.
#
# Two sections of exactly FOUR. The uniform width is the point: at 4/4/5 the last
# row's tiles were narrower than the ones above and nothing lined up vertically,
# which is most of why perfectly good numbers read as a scattered list.
#
# EVERY tile is measured over the selected analysis window. The grid used to hold
# twelve, seven of which were pinned to fixed spans (YTD / 13 wks / 52 wks) read off
# lcw -- so picking "Last 4 weeks" moved five tiles and left seven insisting on 52
# weeks, which reads as a broken control. Those periods were not deleted, they were
# demoted to what they always were: window options. Year to date is still the
# default, so the default grid still opens on YTD figures.
#
# The two headline totals carry the SEASONAL comparison (same weeks last year),
# because that is what a planner reads a total against. Revenue / Week carries the
# MOMENTUM one (the equal-length period just before) and is also the figure that
# makes windows of different lengths comparable at all -- the job the fixed tiles
# used to do. No two tiles can collide on a value, which is the property the
# fixed-window design was protecting, now held structurally instead.
#
# Fields:
#   id     -- stable slug; keys the container, the button and the modal dispatch
#   label  -- tile label ({window} is substituted)
#   kind   -- "money" | "units" | "count" | "percent"; picks the formatter
#   value  -- metrics dict -> raw number (exact; the tile compacts it)
#   delta  -- metrics dict -> percent or None
#   help   -- tooltip AND the definition line inside the modal, so one sentence
#             defines each KPI in exactly one place
_SECTIONS = [
    ("Sell-through", ["revenue", "units", "revenue_per_week", "top10_share"]),
    ("Assortment", ["active_skus", "active_customers", "new_skus", "dormant_skus"]),
]


def _rate(span, key):
    """A span's ``key`` per week, or None when there is no week to divide by.

    Divides by ``covered_weeks`` -- the weeks the span NAMES that the snapshot holds
    -- not by the weeks that happen to carry rows. The two agree on the unfiltered
    frame, which is why this read correctly for so long, and diverge hard under a
    filter: 21 of the customer groups have sales in fewer than 52 of the last 52
    weeks, and dividing by weeks-with-rows made MAKRO's $1,218/week read as
    $63,360/week -- a 52x overstatement, on the one tile whose whole job is to stay
    comparable across windows.

    None (not 0) so the tile renders "—": an empty span has no average, and a
    confident "$0 / week" would be a different claim.
    """
    weeks = span["covered_weeks"]
    total = span[key]
    if not weeks or total is None or pd.isna(total):
        return None
    return total / weeks


def _per_week(span):
    """Revenue per week -- the Revenue / Week tile's own value."""
    return _rate(span, "revenue")


def _tile_delta(tile_id, m):
    """A tile's delta, or None when its comparison base is not comparable.

    Reads ``base`` and ``total`` off the tile's own spec, so the span a delta
    divides by is declared in exactly one place -- the same rule the grid and the
    modals already share.

    Refuses to divide by a span the snapshot does not fully hold. A truncated base
    understates itself, so the delta reports the extract's history anchor as growth:
    "Last 3 years" showed Units +11.0% against a prior year missing 21 of its 156
    weeks, when the like-for-like per-week movement was -4.0% -- wrong by 15 points
    and by its sign. ``_coverage_note`` then states that per-week movement under the
    tile, so the trend is not lost with the bad percentage.
    """
    spec = _TILES[tile_id]
    base = spec.get("base")
    if not base or not m[base]["fully_covered"]:
        return None
    return hm.pct_change(spec["total"](m["window"]), spec["total"](m[base]))


def _coverage_note(tile_id, m):
    """The per-week comparison a withheld delta leaves on the table, or None.

    A truncated base cannot carry a total-against-total delta, but the weeks it DOES
    hold are whole, real weeks -- so per-week against per-week over exactly those
    weeks is a fact, and the only honest trend available for the long windows. It
    goes under the tile as text rather than into the delta arrow because it measures
    something different from the total above it, and says so: the label names the
    per-week basis and how much of the base period is actually on record.

    On the live snapshot this is what "Last 3 years" has instead of its old
    "Units +11.0%": "per week: -4.0% vs the 135 of 156 weeks on record".
    """
    spec = _TILES[tile_id]
    base = spec.get("base")
    if not base:
        return None
    span = m[base]
    # Nothing to add when the delta rendered normally, and nothing to compute when
    # the base holds no weeks at all (All history's momentum span).
    if span["fully_covered"] or not span["covered_weeks"]:
        return None
    pct = hm.pct_change(spec["rate"](m["window"]), spec["rate"](span))
    if pct is None:
        return None
    return (f"per week: {pct:+.1f}% vs the {span['covered_weeks']} of "
            f"{span['span_weeks']} weeks on record")


_TILES = {
    # `base` names the comparison span, `total` the figure compared when the snapshot
    # holds that span in full, and `rate` its per-week form -- the fallback stated
    # under the tile when it does not. Declared here rather than inside the delta
    # lambda so _tile_delta and _coverage_note cannot disagree about which period a
    # tile is measured against.
    "revenue": dict(
        label="Revenue", kind="money",
        value=lambda m: m["window"]["revenue"],
        base="prior_year",
        total=lambda s: s["revenue"],
        rate=lambda s: _rate(s, "revenue"),
        delta=lambda m: _tile_delta("revenue", m),
        help="Units × Plytix list price over the analysis window — a retail-value "
             "proxy, not invoiced revenue. Compared against the same weeks a year "
             "earlier (a 364-day shift, so whole weeks meet whole weeks; a "
             "Year-to-date window anchors to the prior January 1 instead). When that "
             "year-earlier span reaches back past the start of the snapshot it holds "
             "fewer weeks than the window, so the arrow is withheld and a per-week "
             "comparison over the weeks that ARE on record is shown instead.",
    ),
    "units": dict(
        label="Units", kind="units",
        value=lambda m: m["window"]["units"],
        base="prior_year",
        total=lambda s: s["units"],
        rate=lambda s: _rate(s, "units"),
        delta=lambda m: _tile_delta("units", m),
        help="Actual sell-through units over the analysis window (POS, falling back "
             "to Orders for customers who report no POS), against the same weeks a "
             "year earlier. When that span reaches past the start of the snapshot "
             "the arrow gives way to a per-week comparison over the weeks on record.",
    ),
    # Delta is per-week against per-week, not total against total. The two agree
    # whenever both periods are dense, and when they aren't, an average tile whose
    # delta came from totals would be comparing a different pair of numbers than the
    # one it displays.
    # Already a rate, so `total` and `rate` are the same function -- its delta was
    # always per-week against per-week. The coverage gate still applies: the prior
    # period for "Last 3 years" holds 31 real weeks in a 156-week slot, and a
    # seven-month sliver of early 2023 is not "the equal-length period immediately
    # before". The note under the tile says how many weeks it really is.
    "revenue_per_week": dict(
        label="Revenue / Week", kind="money",
        value=lambda m: _per_week(m["window"]),
        base="prior_period",
        total=_per_week,
        rate=_per_week,
        delta=lambda m: _tile_delta("revenue_per_week", m),
        help="Average revenue per complete week in the window — the figure that "
             "stays comparable when you change the window's length. Divided by every "
             "week the window covers, including weeks with no sales, so filtering to "
             "a quiet customer group lowers the average instead of raising it. Its "
             "delta is MOMENTUM: this window against the equal-length period "
             "immediately before it, not against last year, and only when the "
             "snapshot covers that period in full.",
    ),
    "top10_share": dict(
        label="Top-10 Revenue Share", kind="percent",
        value=lambda m: m["concentration"], delta=lambda m: None,
        help="Share of window revenue earned by the ten biggest SKUs. Shows '—' "
             "when nothing in the window carries a list price.",
    ),
    "active_skus": dict(
        label="Active SKUs", kind="count",
        value=lambda m: m["breadth"]["active_skus"], delta=lambda m: None,
        help="SKUs with any sell-through inside the analysis window.",
    ),
    "active_customers": dict(
        label="Active Customers", kind="count",
        value=lambda m: m["breadth"]["active_customers"], delta=lambda m: None,
        help="Customer groups with any sell-through inside the window.",
    ),
    # One of only two figures on this screen that reach outside the window, and both
    # do so because their DEFINITION requires it, not because a span was left
    # hardcoded. Said plainly in the tooltip so it can't read as the old bug.
    "new_skus": dict(
        label="New SKUs", kind="count",
        value=lambda m: m["breadth"]["new_skus"], delta=lambda m: None,
        help="SKUs whose FIRST EVER week of sell-through in this snapshot falls "
             "inside the window. \"First ever\" is measured across all history — "
             "that is what makes a SKU new. A returns-only week does not date a "
             "launch. Shows '—' for a window that opens at the start of the "
             "snapshot, where every SKU would count as new.",
    ),
    "dormant_skus": dict(
        label="Dormant SKUs", kind="count",
        value=lambda m: m["breadth"]["dormant_skus"], delta=lambda m: None,
        help="Sold in the 52 weeks before the window opened, but nothing inside it "
             "— assortment drifting away. The 52-week lookback is fixed whatever "
             "the window: over four weeks, \"dormant\" would mean nothing. Shows "
             "'—' for a window that opens at the start of the snapshot, where that "
             "lookback lands on weeks the data does not reach.",
    ),
}


def _tile_label(tile_id, window):
    return _TILES[tile_id]["label"].format(window=window)


def _tile_value(tile_id, m, compact=True):
    """The tile's display string. ``compact=False`` gives the exact figure.

    Tiles compact (twelve quarter-width columns need one rhythm); tooltips and
    modals show the exact number, so precision is always one hover away.
    """
    spec = _TILES[tile_id]
    v = spec["value"](m)
    kind = spec["kind"]
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    if kind == "percent":
        # The tile rounds to a whole percent; the exact form keeps a decimal, so the
        # tooltip actually adds something instead of repeating the tile.
        return f"{v:.0f}%" if compact else f"{v:.1f}%"
    if kind == "count":
        return f"{v:,}"          # counts are small; abbreviating them loses info
    if kind == "money":
        return fmt_compact(v, money=True) if compact else fmt_dollar(v)
    return fmt_compact(v) if compact else f"{v:,.0f}"


def _span(start, end):
    """A date range as one string, not repeating a year both ends share."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start.year == end.year:
        return f"{start:%b %d} – {end:%b %d, %Y}"
    return f"{start:%b %d, %Y} – {end:%b %d, %Y}"


def _render_window_dates(m):
    """The analysis window's ACTUAL dates, stated above the tiles.

    A named window ("Last 13 weeks") never tells you which weeks it covers, so a
    planner reading a number had no way to know what period it spanned. Sits above
    the tiles rather than in the caption below them, because you need the period
    before you read the figures, not after.

    The two comparison spans are named on the same line, because the tiles' deltas
    are percentages against them and a percentage whose base is unstated is not a
    fact. A span the snapshot does not fully hold is named and marked as such rather
    than dropped: those are exactly the cases where the tiles now render without a
    delta, and an unexplained missing delta reads as a broken tile. (This used to
    claim both spans came back empty for All history. Only the momentum one does --
    the year-earlier span overlaps the window, and the tiles were quietly reporting
    Revenue +36.7% against a subset of themselves.)

    Every span is read straight off ``m`` — the same pairs every tile and breakdown is
    computed from — so the dates displayed are by construction the dates used, not a
    re-derivation that could drift out of step. The charts are NOT among them: each
    tab picks its own years and states its own dates through ``_range_note``, which is
    why this line says "Analysis window" rather than naming a period for the whole
    view.
    """
    start, end = m["bounds"]
    # The weeks the window COVERS, which is what every per-week figure divides by.
    # `["weeks"]` counts weeks carrying rows and would under-report a filtered
    # selection's span -- the mismatch that used to inflate Revenue / Week.
    weeks = m["window"]["covered_weeks"]

    label = m["window_kind"]
    note = ""
    if label == hm.WINDOW_CUSTOM:
        # The picker snaps to whole weeks, so what was typed and what was measured
        # can differ by up to six days at each end — exactly when seeing the real
        # dates matters most.
        label = "Custom range"
        note = " · snapped to whole weeks"

    line = (f"**Analysis window — {label}:** {_span(start, end)} &nbsp;·&nbsp; "
            f"{weeks} complete week{'' if weeks == 1 else 's'}{note}")
    st.markdown(line, help="Every figure below is measured over exactly these "
                           "weeks. Weeks start on Sunday, and the in-progress week "
                           "is always excluded.")

    comparisons = []
    for key, bounds_key, name in (
        ("prior_period", "prior_period_bounds", "the period before"),
        ("prior_year", "prior_year_bounds", "the same weeks last year"),
    ):
        span = m[key]
        if not span["weeks"] and not span["fully_covered"]:
            continue                      # nothing there at all; say nothing
        text = f"vs {name} ({_span(*m[bounds_key])})"
        if not span["fully_covered"]:
            # Name the span AND why no delta came from it, with the date that decided
            # it. Silence here is what makes a withheld delta look like a bug instead
            # of a limit of the data.
            short = span["span_weeks"] - span["covered_weeks"]
            since = (f" (history starts {pd.Timestamp(m['floor']):%b %d, %Y})"
                     if m.get("floor") is not None
                     and not pd.isna(pd.Timestamp(m["floor"])) else "")
            text += (f" — **no delta**: {short} of its {span['span_weeks']} weeks "
                     f"are before the snapshot starts{since}")
        comparisons.append(text)
    if comparisons:
        st.caption("Deltas: " + " &nbsp;·&nbsp; ".join(comparisons))


def _render_kpis(m, window, frame):
    """Eight clickable tiles in two labelled sections of four.

    Plain ``st.metric`` calls rather than tables._render_kpi_tiles -- those are for
    per-row detail cards driven by config.KPI_ORDER, and these are view-level
    aggregates that belong to no row.

    Each tile is a metric PLUS a keyed button in the same keyed container.
    ``st.metric`` has no click event, so the button is the click target; the
    stylesheet in dashboard.py positions it over the whole card. Rendering it as an
    ordinary button and letting CSS move it means that if a Streamlit release ever
    changes the DOM under that selector, the tile degrades to a plainly visible
    working button rather than a dead card.
    """
    _render_window_dates(m)

    for section, tile_ids in _SECTIONS:
        st.caption(f"**{section}**")
        with st.container(key=f"histkpi-row-{section.lower()}"):
            cols = st.columns(len(tile_ids))
            for col, tile_id in zip(cols, tile_ids):
                with col, st.container(key=f"histkpi-tile-{tile_id}"):
                    spec = _TILES[tile_id]
                    # NO help= on either element, deliberately:
                    #  * on the button, a tooltip inserts three auto-height wrappers
                    #    and makes Streamlit emit two <button>s, which is what
                    #    collapsed the click target to the top ~2.5rem of the card;
                    #  * on the metric, the (?) icon would sit UNDER the invisible
                    #    overlay -- unhoverable and unclickable, a dead affordance.
                    # The exact figure and this KPI's definition both lead the modal
                    # body instead, one click away from anywhere on the tile.
                    st.metric(_tile_label(tile_id, window),
                              _tile_value(tile_id, m),
                              delta=_delta(spec["delta"](m)))
                    # The trend a withheld delta would otherwise take with it, as
                    # text rather than an arrow because it measures per-week movement
                    # while the figure above it is a total. Sits above the button so
                    # the click target stays the last element in the card.
                    note = _coverage_note(tile_id, m)
                    if note:
                        st.caption(note)
                    if st.button("Breakdown", key=f"histkpi-go-{tile_id}",
                                 width="stretch"):
                        _breakdown_dialog(tile_id, frame, m, window)


# --------------------------------------------------------------------------- #
# The click-through breakdown modal                                            #
# --------------------------------------------------------------------------- #
# ONE dialog function, dispatching on the tile id. Streamlit allows only one dialog
# function per script run and fixes the title at decoration time, so twelve dialogs
# is not an option and a dynamically-decorated closure would collide on the fragment
# hash (which derives from the function's name). Hence a static title plus the KPI
# name as the modal's own heading.
#
# Opening a dialog from inside a fragment is already proven in this app --
# watchlist_view._list_controls is a fragment that opens three, and _remove_dialog
# opens from inside render_selectable_table's fragment.

# Column formatting per breakdown column name. column_config formatting SILENTLY
# OVERRIDES a pandas Styler on the same column, so these tables are formatted here
# only and never handed a Styler.
_COL_CONFIG = {
    "Units": st.column_config.NumberColumn(format="%,.0f"),
    "Revenue": st.column_config.NumberColumn(format="$%,.0f"),
    "Revenue (prior year)": st.column_config.NumberColumn(format="$%,.0f"),
    "YoY %": st.column_config.NumberColumn(format="%+.1f%%"),
    "Share %": st.column_config.NumberColumn(format="%.1f%%"),
    "Cumulative %": st.column_config.NumberColumn(format="%.1f%%"),
    "Units (prior 52 wks)": st.column_config.NumberColumn(format="%,.0f"),
    "Revenue (prior 52 wks)": st.column_config.NumberColumn(format="$%,.0f"),
    "SKUs": st.column_config.NumberColumn(format="%,.0f"),
    "Weeks with sales": st.column_config.NumberColumn(format="%,.0f"),
    "Weeks since last sale": st.column_config.NumberColumn(format="%,.0f"),
    "First sale week": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
    "Last sale week": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
    "Week": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
}


def _breakdown_frame(tile_id, frame, m):
    """(table, caption) for a tile's modal.

    Each entry answers the question the tile provokes -- "which ones?" for a count,
    "made up of what?" for a total.

    Every table is built from ``m["bounds"]`` and ``m["prior_year_bounds"]``, so a
    modal cannot describe a different period from the tile that opened it. This used
    to re-derive fixed YTD / 13-week / 52-week spans from ``lcw``, which is why the
    parameter is gone.
    """
    start, end = m["bounds"]

    if tile_id == "revenue":
        return (hm.monthly_breakdown(frame, start, end, *m["prior_year_bounds"]),
                "Month by month across the window, against the same months last "
                "year. A part-month at either end is a part-month: the window is "
                "whole WEEKS, which don't align to month boundaries.")
    if tile_id == "units":
        return (hm.monthly_breakdown(frame, start, end, *m["prior_year_bounds"]),
                "Month by month across the window. Units are the left column; "
                "revenue is shown alongside for context.")
    if tile_id == "revenue_per_week":
        return (hm.weekly_breakdown(frame, start, end),
                "Every week in the window, most recent first — the weeks the "
                "average is taken over, so an outlier week is visible rather than "
                "buried in the mean.")
    if tile_id == "top10_share":
        return (hm.top_share_breakdown(frame, start, end, n=10),
                "The ten biggest SKUs by revenue, with units and each SKU's share "
                "of ALL window revenue (not just of this table).")
    if tile_id == "active_skus":
        return (hm.active_skus_breakdown(frame, start, end),
                "Every SKU that sold inside the window, biggest revenue first.")
    if tile_id == "active_customers":
        return (hm.active_customers_breakdown(frame, start, end),
                "Every customer group that sold inside the window.")
    # Both of these are defined by what happened BEFORE the window, so when the
    # window opens at the start of the snapshot the tile reads "—" (see hm.breadth).
    # The modal has to agree with it: showing the raw list under a blank tile is the
    # discrepancy the shared-implementation rule exists to prevent. `breadth` is the
    # single place that decision is made, so read it rather than re-testing the floor.
    if tile_id == "new_skus":
        if m["breadth"]["new_skus"] is None:
            return (pd.DataFrame(), "This window opens at the earliest week on "
                    "record, so every SKU's first sale falls inside it by "
                    "definition. Pick a shorter window to see genuine launches.")
        return (hm.new_skus_breakdown(frame, start, end),
                "SKUs whose first ever recorded sale falls inside the window, "
                "oldest launch first.")
    if tile_id == "dormant_skus":
        if m["breadth"]["dormant_skus"] is None:
            return (pd.DataFrame(), "Dormancy looks back 52 weeks before the window "
                    "opened, and this window opens at the earliest week on record — "
                    "there is no earlier stretch to have sold in.")
        return (hm.dormant_skus_breakdown(frame, start, end),
                "SKUs that sold in the 52 weeks before the window but nothing "
                "inside it. Units and revenue describe that earlier stretch.")
    return pd.DataFrame(), ""


@st.dialog("KPI breakdown", width="large")
def _breakdown_dialog(tile_id, frame, m, window):
    """The list behind one tile: heading, exact figure, definition, table, CSV."""
    spec = _TILES[tile_id]
    st.subheader(_tile_label(tile_id, window))

    exact = _tile_value(tile_id, m, compact=False)
    compact = _tile_value(tile_id, m)
    # Both forms together, so a planner can tie the tile they clicked to the
    # precise figure without doing the arithmetic themselves.
    st.markdown(f"### {exact}" + (f"  &nbsp;<small>({compact} on the tile)"
                                  f"</small>" if compact != exact else ""),
                unsafe_allow_html=True)
    st.caption(spec["help"])

    table, note = _breakdown_frame(tile_id, frame, m)
    if table is None or table.empty:
        st.info("Nothing to break down for this selection.")
        return
    if note:
        st.caption(note)

    # The SKU-grained breakdowns get the blue "Key" chip beside the SKU like every
    # other table; the month/week ones have no SKU column and mark_key_sku no-ops.
    display, sku_values = mark_key_sku(table)
    st.dataframe(
        display, width="stretch", hide_index=True,
        column_config={**{c: cfg for c, cfg in _COL_CONFIG.items()
                          if c in display.columns},
                       **(sku_chip_column_config(sku_values) if sku_values else {})},
    )
    st.download_button(
        "⬇️ Download this breakdown (CSV)",
        with_export_flags(table).to_csv(index=False).encode("utf-8"),
        file_name=f"historical_{tile_id}.csv",
        mime="text/csv",
        key=f"histkpi-dl-{tile_id}",
    )


def _render_caption(m, frame, lcw, base_rows, filtered_rows):
    """The honesty line: as-of week, price coverage, and the discontinued note."""
    priced, total, pct = m["coverage"]
    bits = [f"Data through the week of **{pd.Timestamp(lcw):%b %d, %Y}** "
            f"(the last complete week; the in-progress week is excluded)."]
    if total:
        bits.append(
            f"Revenue covers **{priced} of {total}** SKUs that sold in this window "
            f"(**{pct:.0f}%** of units) — SKUs with no Plytix list price are left "
            f"out of revenue rather than counted as $0."
        )
    else:
        bits.append("No list prices loaded, so revenue figures are unavailable.")
    bits.append(
        "Discontinued SKUs are **included** here for the years they were active, "
        "so these totals run higher than the projection views, which correctly "
        "drop them."
    )
    # The window's DATES and its two comparison spans are stated above the tiles by
    # _render_window_dates -- once, where they're needed before reading the numbers.
    # What's left to say here is the window's own totals and the two figures that
    # reach outside it by definition, so neither reads as a span left hardcoded.
    win = m["window"]
    bits.append(
        f"That window ({win['units']:,.0f} units / {fmt_dollar(win['revenue'])}) "
        f"is the period for **every tile**; the charts below do not follow it — "
        f"each tab picks its own years and states the dates it plots. Two tile "
        f"definitions reach past the window on purpose: **New SKUs** tests a "
        f"first-ever sale across all history, and **Dormant SKUs** looks back a "
        f"fixed 52 weeks before the window opened."
    )
    if filtered_rows != base_rows:
        bits.append(f"Filtered to **{filtered_rows:,}** of {base_rows:,} SKU-weeks.")
    st.caption(" ".join(bits))


# --------------------------------------------------------------------------- #
# Chart tabs                                                                   #
# --------------------------------------------------------------------------- #
# The "show everything" entry in the single-year pickers. A sentinel string rather
# than None so it can sit in a selectbox option list beside the real years.
ALL_YEARS = "All years"


def _year_multiselect(years, key):
    """Year picker for the trend tab: any combination, all of them by default.

    A multiselect rather than a date range because the trend charts overlay years on
    one Jan-Dec axis -- the unit a planner compares here IS the year, and an arbitrary
    span (say Mar 2024 to Aug 2025) would cut two seasons in half and overlay the
    halves. Returns a list of ints; an empty list is a legitimate state and the
    figures render their empty panel for it.
    """
    picked = st.multiselect(
        "Years", years, default=years, key=key,
        format_func=str,
        help="Each selected year is drawn as its own line/bar. Colours follow the "
             "year, so removing one never recolours the others.",
    )
    return [int(y) for y in picked]


def _year_select(years, key):
    """Year picker for the mix and movers tabs: exactly one year, or all of them.

    Single-select because these tabs' charts each collapse a period into one ranking
    -- a share donut or a Pareto curve over several years merges them rather than
    comparing them, so offering a multi-select would promise a comparison the figures
    cannot draw. Defaults to the latest year, which is the one a planner opens for.

    Returns an int, or ``ALL_YEARS``.
    """
    options = [ALL_YEARS] + [int(y) for y in reversed(years)]
    return st.selectbox(
        "Year", options, index=1 if years else 0, key=key, format_func=str,
        help="One calendar year at a time. The current year runs to the last "
             "complete week, not to December.",
    )


def _year_bounds(frame, choice, lcw):
    """(start, end) for a ``_year_select`` choice, or None when it holds no data."""
    if choice == ALL_YEARS:
        weeks = pd.to_datetime(frame["WeekDate"])
        return (weeks.min(), weeks.max()) if not weeks.empty else None
    return hm.calendar_year_bounds(choice, frame, lcw)


def _range_note(frame, start, end):
    """State the dates the charts below are ACTUALLY plotting.

    Sits directly under the tab label and above the first figure. A named selection
    ("2026", "All years") never tells you which weeks it covers, and the answer is
    not guessable: the current year stops at the last complete week rather than
    December, and the earliest year on record starts wherever the extract's history
    anchor happens to sit rather than at January.

    The week count is weeks that carry DATA, counted off the plotted frame, and is
    labelled as such -- calling it "complete weeks" would overclaim for a sparse
    selection where a filter has emptied some of the span.
    """
    if start is None or end is None:
        return
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    weeks = int(pd.to_datetime(frame["WeekDate"]).nunique()) if not frame.empty else 0
    bits = [f"**Showing:** {_span(start, end)}",
            f"{weeks} week{'' if weeks == 1 else 's'} of data"]
    # Only for a year that has not finished -- a completed year ending in December
    # needs no caveat, and printing one would imply the data were incomplete.
    if end < pd.Timestamp(year=end.year, month=12, day=31):
        bits.append(f"{end.year} is still in progress")
    st.markdown(" &nbsp;·&nbsp; ".join(bits))


def _tab_trend(frame, value_col, years, colors):
    """All history, every year overlaid on one Jan-Dec axis.

    Two charts, not the three this tab used to carry. The third was a seasonality
    line chart against a bare week-of-year number, which plotted exactly the curves
    the weekly overlay now does -- the only difference was the axis tick labels.
    """
    st.caption(
        "All history, independent of the analysis window above. Years are overlaid "
        "on a shared January–December axis, so the same week of each year lines up."
    )
    picked = _year_multiselect(years, "hist_trend_years")
    if picked:
        scoped = frame[frame["WeekDate"].dt.year.isin(picked)]
        weeks = pd.to_datetime(scoped["WeekDate"])
        _range_note(scoped, weeks.min(), weeks.max())

    # The builders get the WHOLE frame and filter through `years`, rather than being
    # handed a pre-filtered one. Two reasons: an empty selection then reaches them as
    # "no years chosen" instead of "no data", which is a far more useful panel to read;
    # and `colors` is keyed on every year, so the filtering has to happen where the
    # colour lookup does or the two could disagree about which years exist.
    #
    # The explicit keys are load-bearing, not decoration. Streamlit derives a
    # chart's element ID from its parameters, and with NO years selected both
    # builders return the same "No years selected" placeholder figure -- byte for
    # byte identical, so the two auto-generated IDs collided and the whole tab
    # raised DuplicateElementId (surfacing as FragmentHandledException, since this
    # renders inside _render_body's fragment). Keep the keys.
    st.plotly_chart(
        hc.weekly_year_overlay(hm.weekly_by_year(frame, value_col), value_col,
                               colors=colors, years=picked), width="stretch",
        key="hist_trend_weekly_overlay")
    st.plotly_chart(
        hc.monthly_year_overlay(hm.monthly_totals(frame, value_col), value_col,
                                colors=colors, years=picked), width="stretch",
        key="hist_trend_monthly_overlay")


def _tab_mix(frame, value_col, years, lcw):
    # Three charts. There was a fourth -- a revenue-share-by-SKU-type donut -- which
    # planners didn't want; the `SKU Type` column stays on the frame (see
    # historical_metrics.attach_sku_type) so that analysis can come back cheaply, but
    # nothing reads it today.
    st.caption(
        "One year at a time, independent of the analysis window above."
    )
    choice = _year_select(years, "hist_mix_year")
    bounds = _year_bounds(frame, choice, lcw)
    if bounds is None:
        st.info("No sell-through history in that year for this selection.")
        return
    start, end = bounds
    _range_note(hm.clip(frame, start, end), start, end)

    # Bounds, not a clipped frame: by_dimension already clips internally, so
    # pre-clipping here would filter the same rows twice.
    st.plotly_chart(
        hc.share_donut(hm.by_dimension(frame, "Region", start, end, value_col),
                       "Region", value_col), width="stretch",
        key="hist_mix_region_donut")
    st.plotly_chart(
        hc.ranked_bars(
            hm.by_dimension(frame, "Customer Grouping", start, end, value_col,
                            top_n=10),
            "Customer Grouping", value_col,
            title=f"Top customer groups by {_measure_word(value_col)}"),
        width="stretch", key="hist_mix_customer_bars")
    # Clipped explicitly, because weekly_by_dimension takes a frame rather than
    # bounds. Handing it the whole frame put one tab's three charts on two different
    # periods -- the donut and the bars on the year, the lines on all history.
    st.plotly_chart(
        hc.dimension_lines(hm.weekly_by_dimension(hm.clip(frame, start, end),
                                                  "Region", value_col),
                           "Region", value_col), width="stretch",
        key="hist_mix_region_lines")


def _tab_movers(frame, value_col, years, lcw):
    st.caption(
        "One year at a time, independent of the analysis window above."
    )
    choice = _year_select(years, "hist_movers_year")
    bounds = _year_bounds(frame, choice, lcw)
    if bounds is None:
        st.info("No sell-through history in that year for this selection.")
        return
    start, end = bounds
    _range_note(hm.clip(frame, start, end), start, end)

    # top_skus always returns BOTH measures; pick the one being displayed rather
    # than renaming a column (renaming revenue -> demand would plot dollars under
    # a units label).
    top = hm.top_skus(frame, start, end, n=15)
    measure = "revenue" if value_col == "revenue" else "units"
    chart_frame = (top[["SKU", measure]].rename(columns={measure: value_col})
                   if not top.empty else top)
    st.plotly_chart(
        hc.ranked_bars(chart_frame, "SKU", value_col,
                       title=f"Top 15 SKUs by {_measure_word(value_col)}"),
        width="stretch", key="hist_movers_top_skus")
    if not top.empty:
        render_filtered_table(
            top.rename(columns={"units": "Units", "revenue": "Revenue"}),
            key="hist_top_skus", style=False,
            column_config={
                "Units": st.column_config.NumberColumn(format="%,.0f"),
                "Revenue": st.column_config.NumberColumn(format="$%,.0f"),
            },
        )

    # Movers and Pareto are revenue-only by definition: "biggest mover" and
    # "concentration" are commercial questions, and ranking them by units would
    # let a high-volume, low-value SKU outrank the business's actual earners.
    st.caption("Movers and concentration are measured in revenue regardless of "
               "the measure toggle — ranking them by units would let cheap, "
               "high-volume SKUs outrank the real earners.")

    if choice == ALL_YEARS:
        # No honest comparable: "all history against the year before all history"
        # is not a period that exists. Saying so beats drawing a chart whose prior
        # side is empty and whose every SKU therefore reads as a record gainer.
        st.info(
            "Pick a single year to see year-over-year movers — with every year "
            "selected there is no earlier period to compare against."
        )
    else:
        # The same span one CALENDAR year earlier, passed explicitly. yoy_movers
        # defaults to a 364-day shift, which is right for a rolling window (it keeps
        # whole weeks aligned) but wrong for a calendar year: shifting Jan 1–Dec 31
        # 2025 back 364 days gives Jan 2 2024 – Jan 1 2025, a span overlapping the
        # very year being measured. A calendar shift is what "versus last year" means
        # once the selector is a year.
        prior = (start - pd.DateOffset(years=1), end - pd.DateOffset(years=1))
        st.caption(f"Compared against {_span(*prior)}.")
        st.plotly_chart(
            hc.movers_chart(hm.yoy_movers(frame, start, end, prior=prior, n=10)),
            width="stretch", key="hist_movers_yoy")

    st.plotly_chart(hc.pareto_chart(hm.pareto(frame, start, end)),
                    width="stretch", key="hist_movers_pareto")
    with st.expander("How to read the concentration chart"):
        st.markdown(
            "Each point is one SKU, ranked highest-revenue first, and the line is "
            "the running share of total revenue those SKUs account for. Where it "
            "crosses the dotted 80% line tells you how many SKUs carry four-fifths "
            "of the business.\n\n"
            "- **A steep early climb** means revenue is concentrated in a handful of "
            "SKUs. Those are where a forecast miss or a stockout costs the most, and "
            "where extra forecasting attention pays for itself.\n"
            "- **A shallow, straight line** means revenue is spread thin across many "
            "SKUs. No single miss hurts much, but the long tail is expensive to plan "
            "in aggregate and is usually where safety stock quietly accumulates.\n\n"
            "Switching the year selector above and watching the crossing point move "
            "tells you whether the business is consolidating onto fewer winners or "
            "diversifying — which changes where forecasting effort is worth spending."
        )


def _tab_heatmap(frame, value_col):
    # The only tab with no control at all, and deliberately so: a month x year grid
    # already HAS a year axis, so a year picker would just delete columns from a chart
    # whose entire point is comparing them side by side.
    st.caption(
        "Every year in the snapshot, independent of the analysis window above — "
        "this grid needs several seasons side by side to say anything."
    )
    weeks = pd.to_datetime(frame["WeekDate"])
    _range_note(frame, weeks.min(), weeks.max())
    st.plotly_chart(hc.month_year_heatmap(hm.month_year_matrix(frame, value_col),
                                          value_col), width="stretch",
                    key="hist_heatmap_grid")


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #
@st.fragment
def _render_body(base, lcw):
    """Filter bar + tiles + charts, isolated in a fragment.

    The fragment keeps a filter or tab change from rerunning the whole dashboard
    (which would re-read the snapshot and re-run every cached forecast lookup) --
    the same isolation tables.render_filtered_table already uses for filter chips.
    """
    regions, groups, window, bounds = _filter_bar(base, lcw)
    # The earliest week the SNAPSHOT holds, off `base` -- before any filter. This is
    # the line between "no data" and "sold nothing", and every withheld delta is
    # decided against it. Read from the filtered frame it would move with the
    # selection, and a customer who started trading last year would look like a
    # snapshot that starts last year, suppressing their real growth.
    floor = pd.to_datetime(base["WeekDate"]).min()
    frame = _apply_filters(base, regions, groups)
    # Drop the snapshot's forward projection weeks ONCE, here, so nothing downstream
    # can see them: not a chart, and not either range control's data maximum (which
    # is read off the frame and was reaching ~15 weeks past the last complete week).
    # all_history_bounds already clips for exactly this reason; this generalises it.
    # _filter_bar keeps reading `base`, so the custom-range picker is unaffected.
    frame = frame[frame["WeekDate"] <= pd.Timestamp(lcw)]

    if frame.empty:
        st.info("No sell-through history matches these filters. Clear one to "
                "widen the selection.")
        return

    if window == hm.WINDOW_CUSTOM and bounds is None:
        # Either only one end of the range has been picked yet, or the range is too
        # narrow to hold a single complete week. Say so and stop rather than falling
        # back to a default window, which would show numbers for a period the user
        # did not ask for.
        st.warning(
            "Pick a start and an end at least one full week apart. Sell-through is "
            "recorded weekly (weeks start on Sunday), so a range shorter than one "
            "complete week has nothing to total."
        )
        return

    # The snapped dates are part of the cache key, or two different custom ranges
    # would collide on the same entry. The snapshot signature is in it too: without
    # one, a refreshed workbook that happens to yield the same row count for the same
    # filters would be served last load's numbers.
    cache_key = (st.session_state.get("hist_base_sig"),
                 tuple(regions), tuple(groups), window, bounds, len(frame))
    m = _metrics(frame, window, lcw, cache_key, bounds=bounds, floor=floor)

    _render_kpis(m, window, frame)
    _render_caption(m, frame, lcw, len(base), len(frame))

    # Revenue is the primary lens for this view; units stay available in every
    # hover and in the top-SKU table. The toggle exists because a planner
    # occasionally needs to see volume where price mix would distort the picture.
    value_col = "revenue" if st.radio(
        "Measure", ["Revenue", "Units"], horizontal=True, key="hist_measure",
        help="Revenue is units x list price; Units is raw sell-through.",
    ) == "Revenue" else "demand"

    t1, t2, t3, t4 = st.tabs(["Trend & Seasonality", "Mix & Breakdown",
                              "Movers & Concentration", "Seasonal Heatmap"])
    # No spans passed down: every tab resolves its own from its year picker. `m` stays
    # the single source for the TILES' three spans, which is what it was always for.
    #
    # `years` and `colors` are built ONCE here, off the whole filtered frame, and
    # handed to the tabs. That is what makes a year's colour stable: rebuilt inside a
    # tab from its own selection, deselecting 2023 would shift 2024 into slot 1 and
    # repaint a line the planner never touched.
    years = hm.available_years(frame, lcw)
    colors = hc.year_color_map(years)
    with t1:
        _tab_trend(frame, value_col, years, colors)
    with t2:
        _tab_mix(frame, value_col, years, lcw)
    with t3:
        _tab_movers(frame, value_col, years, lcw)
    with t4:
        _tab_heatmap(frame, value_col)


def render_historical_summary(df_hist, today_ts, today_str, prices,
                              n_excluded_rows, anchors, P, plytix_df=None,
                              data_sig=None, onhand_by_sku=None):
    """Render the Historical Summary view.

    Signature mirrors render_exceptions / render_watchlist / _render_best_model_
    combined so main()'s dispatch call is uniform, plus ``plytix_df`` for SKU Type.
    ``df_hist`` is ExclusionResult.df_with_discontinued -- the frame that still
    holds retired SKUs' active years -- NOT the forecast frame. ``data_sig`` and
    ``onhand_by_sku`` are accepted for signature parity and unused: nothing here
    forecasts, and on-hand is a forward-looking supply figure with no place in a
    backward-looking view.
    """
    _, lcw, _ = anchors

    if df_hist is None or df_hist.empty:
        st.info("No demand history loaded.")
        return

    price_map = _as_price_map(prices)
    base = _base_frame(df_hist, P, price_map, plytix_df, today_str)
    if base.empty:
        st.info("No sell-through history in this snapshot.")
        return

    _render_body(base, lcw)
