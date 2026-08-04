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
        help="Scopes the Assortment tiles, the top-10 share, and the Mix / Movers "
             "charts. The Revenue and Units tiles cover their own named windows, "
             "and the trend charts keep their own date range.",
    )

    bounds = None
    if window == hm.WINDOW_CUSTOM:
        bounds = _custom_range(base, lcw)
    return picked_regions, picked_groups, window, bounds


def _custom_range(base, lcw):
    """Date picker for a custom analysis window, snapped to whole weeks.

    Mirrors the "Custom…" branch of historical_charts.history_range_control,
    including its guard that ``st.date_input`` returns a SINGLE date mid-selection
    and must only be applied once both ends have been chosen.

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


def _metrics(frame, window, lcw, cache_key, bounds=None):
    """Every window-scoped figure the tiles need, cached per filter+window combo.

    ``bounds`` overrides the named window's own span — that is how a custom range
    (already snapped to whole weeks) gets in.
    """
    store = st.session_state.setdefault("hist_metrics", {})
    if cache_key in store:
        return store[cache_key]

    start, end = bounds if bounds is not None else hm.window_bounds(window, lcw)
    ytd_start, ytd_end = hm.window_bounds(hm.WINDOW_YTD, lcw)
    ytd_prior = hm.prior_year_window(ytd_start, ytd_end, anchor_to_year_start=True)
    w4_start, w4_end = lcw - pd.Timedelta(weeks=3), lcw
    w13_start, w13_end = hm.window_bounds(hm.WINDOW_13W, lcw)
    w52_start, w52_end = hm.window_bounds(hm.WINDOW_52W, lcw)

    result = {
        "bounds": (start, end),
        # Kept so the modal can re-derive the prior-year window with the same
        # YTD anchoring rule the tile used.
        "window_kind": window,
        "window": hm.window_totals(frame, start, end),
        # No prior-year total for the SELECTED window: every tile compares against a
        # fixed window of its own (see _TILES), so a "prior" here went unread. Its
        # removal is also what keeps a custom or calendar-year window from needing a
        # year-over-year alignment rule of its own.
        "ytd": hm.window_totals(frame, ytd_start, ytd_end),
        "ytd_prior": hm.window_totals(frame, *ytd_prior),
        "w4": hm.window_totals(frame, w4_start, w4_end),
        # The four weeks immediately before the last four -- a sequential
        # comparison, NOT year-over-year, so a planner reads momentum not season.
        "w4_prev": hm.window_totals(frame, w4_start - pd.Timedelta(weeks=4),
                                    w4_start - pd.Timedelta(days=1)),
        "w13": hm.window_totals(frame, w13_start, w13_end),
        "w13_prev": hm.window_totals(frame, w13_start - pd.Timedelta(weeks=13),
                                     w13_start - pd.Timedelta(days=1)),
        "w52": hm.window_totals(frame, w52_start, w52_end),
        "breadth": hm.breadth(frame, start, end),
        "concentration": hm.concentration(frame, start, end, n=10),
        "coverage": hm.price_coverage(frame, start, end),
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
# Three sections of exactly FOUR. The uniform width is the point: at 4/4/5 the last
# row's tiles were narrower than the ones above and nothing lined up vertically,
# which is most of why twelve perfectly good numbers read as a scattered list.
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
    ("Revenue", ["ytd_revenue", "rev_13w", "rev_52w", "top10_share"]),
    ("Units", ["ytd_units", "units_4w", "units_13w", "units_52w"]),
    ("Assortment", ["active_skus", "active_customers", "new_skus", "dormant_skus"]),
]

_TILES = {
    "ytd_revenue": dict(
        label="YTD Revenue", kind="money",
        value=lambda m: m["ytd"]["revenue"],
        delta=lambda m: hm.pct_change(m["ytd"]["revenue"], m["ytd_prior"]["revenue"]),
        help="Units × Plytix list price for every completed week since Jan 1 — a "
             "retail-value proxy, not invoiced revenue. Compared against the same "
             "stretch of last year (Jan 1 to 364 days before the last complete "
             "week, so whole weeks meet whole weeks).",
    ),
    # Fixed 13 weeks, NOT the selected analysis window. A window-scoped revenue
    # tile duplicated YTD Revenue exactly whenever the window was Year to date --
    # which is the default -- so the default view showed one number twice and both
    # tiles opened the same modal. Fixed windows here mirror the Units row below and
    # can never collide; the analysis window still scopes the Assortment tiles,
    # the concentration share, and every Mix / Movers / Heatmap chart.
    "rev_13w": dict(
        label="Last 13 Wks Revenue", kind="money",
        value=lambda m: m["w13"]["revenue"],
        delta=lambda m: hm.pct_change(m["w13"]["revenue"],
                                      m["w13_prev"]["revenue"]),
        help="Revenue over the last 13 complete weeks — a quarter — against the "
             "quarter before it.",
    ),
    "rev_52w": dict(
        label="Trailing 52-Wk Revenue", kind="money",
        value=lambda m: m["w52"]["revenue"], delta=lambda m: None,
        help="The last 52 complete weeks — a full season, independent of where we "
             "are in the calendar year.",
    ),
    "top10_share": dict(
        label="Top-10 Revenue Share", kind="percent",
        value=lambda m: m["concentration"], delta=lambda m: None,
        help="Share of window revenue earned by the ten biggest SKUs. Shows '—' "
             "when nothing in the window carries a list price.",
    ),
    "ytd_units": dict(
        label="YTD Units", kind="units",
        value=lambda m: m["ytd"]["units"],
        delta=lambda m: hm.pct_change(m["ytd"]["units"], m["ytd_prior"]["units"]),
        help="Actual sell-through units since Jan 1 (POS, falling back to Orders "
             "for customers who report no POS), vs the same stretch last year.",
    ),
    "units_4w": dict(
        label="Last 4 Wks Units", kind="units",
        value=lambda m: m["w4"]["units"],
        delta=lambda m: hm.pct_change(m["w4"]["units"], m["w4_prev"]["units"]),
        help="The last 4 complete weeks against the 4 before them — momentum, not a "
             "year-over-year comparison, so a seasonal peak reads as a rise.",
    ),
    "units_13w": dict(
        label="Last 13 Wks Units", kind="units",
        value=lambda m: m["w13"]["units"],
        delta=lambda m: hm.pct_change(m["w13"]["units"], m["w13_prev"]["units"]),
        help="A quarter of complete weeks, against the quarter before it.",
    ),
    "units_52w": dict(
        label="Trailing 52 Wks Units", kind="units",
        value=lambda m: m["w52"]["units"], delta=lambda m: None,
        help="Units over the last 52 complete weeks.",
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
    "new_skus": dict(
        label="New SKUs", kind="count",
        value=lambda m: m["breadth"]["new_skus"], delta=lambda m: None,
        help="SKUs whose FIRST EVER week of sell-through in this snapshot falls "
             "inside the window.",
    ),
    "dormant_skus": dict(
        label="Dormant SKUs", kind="count",
        value=lambda m: m["breadth"]["dormant_skus"], delta=lambda m: None,
        help="Sold in the 52 weeks before the window opened, but nothing inside it "
             "— assortment drifting away.",
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


def _render_kpis(m, window, frame, lcw):
    """Twelve clickable tiles in three labelled sections of four.

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
                    if st.button("Breakdown", key=f"histkpi-go-{tile_id}",
                                 width="stretch"):
                        _breakdown_dialog(tile_id, frame, m, lcw, window)


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


def _breakdown_frame(tile_id, frame, m, lcw):
    """(table, caption) for a tile's modal.

    Each entry answers the question the tile provokes -- "which ones?" for a count,
    "made up of what?" for a total.
    """
    start, end = m["bounds"]
    ytd_start, ytd_end = hm.window_bounds(hm.WINDOW_YTD, lcw)
    ytd_prior = hm.prior_year_window(ytd_start, ytd_end, anchor_to_year_start=True)
    w52 = hm.window_bounds(hm.WINDOW_52W, lcw)

    if tile_id == "ytd_revenue":
        return (hm.monthly_breakdown(frame, ytd_start, ytd_end, *ytd_prior),
                "Month by month since Jan 1, against the same months last year.")
    if tile_id == "rev_13w":
        w13 = hm.window_bounds(hm.WINDOW_13W, lcw)
        return (hm.monthly_breakdown(frame, *w13),
                "Month by month across the trailing quarter. (The Units tile below "
                "breaks the same quarter down week by week.)")
    if tile_id == "rev_52w":
        return (hm.monthly_breakdown(frame, *w52),
                "Month by month across the last 52 complete weeks.")
    if tile_id == "top10_share":
        return (hm.top_share_breakdown(frame, start, end, n=10),
                "The ten biggest SKUs by revenue, with units and each SKU's share "
                "of ALL window revenue (not just of this table).")
    if tile_id == "ytd_units":
        return (hm.monthly_breakdown(frame, ytd_start, ytd_end, *ytd_prior),
                "Month by month since Jan 1. Units are the left column; revenue is "
                "shown alongside for context.")
    if tile_id == "units_4w":
        return (hm.weekly_breakdown(frame, start=lcw - pd.Timedelta(weeks=7),
                                    end=lcw),
                "The last 4 complete weeks and the 4 before them, so the comparison "
                "the tile makes is visible week by week.")
    if tile_id == "units_13w":
        return (hm.weekly_breakdown(frame, *hm.window_bounds(hm.WINDOW_13W, lcw)),
                "Every week of the trailing quarter, most recent first.")
    if tile_id == "units_52w":
        return (hm.monthly_breakdown(frame, *w52),
                "Month by month across the last 52 complete weeks.")
    if tile_id == "active_skus":
        return (hm.active_skus_breakdown(frame, start, end),
                "Every SKU that sold inside the window, biggest revenue first.")
    if tile_id == "active_customers":
        return (hm.active_customers_breakdown(frame, start, end),
                "Every customer group that sold inside the window.")
    if tile_id == "new_skus":
        return (hm.new_skus_breakdown(frame, start, end),
                "SKUs whose first ever recorded sale falls inside the window, "
                "oldest launch first.")
    if tile_id == "dormant_skus":
        return (hm.dormant_skus_breakdown(frame, start, end),
                "SKUs that sold in the 52 weeks before the window but nothing "
                "inside it. Units and revenue describe that earlier stretch.")
    return pd.DataFrame(), ""


@st.dialog("KPI breakdown", width="large")
def _breakdown_dialog(tile_id, frame, m, lcw, window):
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

    table, note = _breakdown_frame(tile_id, frame, m, lcw)
    if table is None or table.empty:
        st.info("Nothing to break down for this selection.")
        return
    if note:
        st.caption(note)

    st.dataframe(
        table, width="stretch", hide_index=True,
        column_config={c: cfg for c, cfg in _COL_CONFIG.items()
                       if c in table.columns},
    )
    st.download_button(
        "⬇️ Download this breakdown (CSV)",
        table.to_csv(index=False).encode("utf-8"),
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
    # The Revenue and Units tiles all cover FIXED windows (YTD / 13 / 52 weeks), so
    # say plainly what the Analysis window selector does move -- otherwise changing
    # it and seeing the top two rows sit still reads as a broken control.
    win = m["window"]
    start, end = m["bounds"]
    # A custom range is named by the WEEKS IT ACTUALLY COVERS, not by what was typed:
    # the picker snaps to whole weeks, so the two can differ by up to six days at each
    # end and a planner must be able to see what was really measured.
    if m["window_kind"] == hm.WINDOW_CUSTOM:
        span = (f"**Custom range**, snapped to whole weeks: "
                f"**{pd.Timestamp(start):%b %d %Y} – {pd.Timestamp(end):%b %d %Y}**")
    else:
        span = f"The **{m['window_kind']}** analysis window"
    bits.append(
        f"{span} ({win['weeks']} complete weeks, {win['units']:,.0f} units / "
        f"{fmt_dollar(win['revenue'])}) scopes the **Assortment** tiles, the "
        f"top-10 share, and the Mix / Movers / Heatmap charts. The Revenue and "
        f"Units tiles above cover their own named windows."
    )
    if filtered_rows != base_rows:
        bits.append(f"Filtered to **{filtered_rows:,}** of {base_rows:,} SKU-weeks.")
    st.caption(" ".join(bits))


# --------------------------------------------------------------------------- #
# Chart tabs                                                                   #
# --------------------------------------------------------------------------- #
def _tab_trend(frame, value_col):
    st.caption(
        "Full history, independent of the analysis window above — these charts "
        "have their own date range."
    )
    rng = hc.history_range_control(frame, key="hist_trend")
    scoped = hm.clip(frame, *rng) if rng else frame

    st.plotly_chart(hc.weekly_trend_chart(hm.weekly_totals(scoped, value_col),
                                          value_col), width="stretch")
    monthly = hm.monthly_totals(scoped, value_col)
    latest_year = int(pd.Timestamp(frame["WeekDate"].max()).year) \
        if not frame.empty else None
    st.plotly_chart(hc.monthly_yoy_chart(monthly, value_col, latest_year),
                    width="stretch")
    st.plotly_chart(hc.seasonality_chart(hm.seasonality_frame(scoped, value_col),
                                         value_col), width="stretch")


def _tab_mix(frame, value_col, start, end):
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            hc.share_donut(hm.by_dimension(frame, "Region", start, end, value_col),
                           "Region", value_col), width="stretch")
    with right:
        st.plotly_chart(
            hc.share_donut(
                hm.by_dimension(frame, "SKU Type", start, end, value_col, top_n=8),
                "SKU Type", value_col), width="stretch")
    st.plotly_chart(
        hc.ranked_bars(
            hm.by_dimension(frame, "Customer Grouping", start, end, value_col,
                            top_n=10),
            "Customer Grouping", value_col,
            title=f"Top customer groups by {_measure_word(value_col)}"),
        width="stretch")
    st.plotly_chart(
        hc.stacked_area(hm.weekly_by_dimension(frame, "Region", value_col),
                        "Region", value_col), width="stretch")


def _tab_movers(frame, value_col, start, end, lcw):
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
        width="stretch")
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
    st.plotly_chart(hc.movers_chart(hm.yoy_movers(frame, lcw, n=10)),
                    width="stretch")
    st.plotly_chart(hc.pareto_chart(hm.pareto(frame, start, end)), width="stretch")


def _tab_heatmap(frame, value_col):
    st.plotly_chart(hc.month_year_heatmap(hm.month_year_matrix(frame, value_col),
                                          value_col), width="stretch")


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
    frame = _apply_filters(base, regions, groups)

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
    # would collide on the same entry.
    cache_key = (tuple(regions), tuple(groups), window, bounds, len(frame))
    m = _metrics(frame, window, lcw, cache_key, bounds=bounds)
    start, end = m["bounds"]

    _render_kpis(m, window, frame, lcw)
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
    with t1:
        _tab_trend(frame, value_col)
    with t2:
        _tab_mix(frame, value_col, start, end)
    with t3:
        _tab_movers(frame, value_col, start, end, lcw)
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
