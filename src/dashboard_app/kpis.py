"""KPI row and the Optimal Projections (best-model-per-group) combined view."""
import functools
import re

import pandas as pd
import streamlit as st

from dashboard_app.config import (
    PRICE_COL, RISK_COL, fmt_dollar, MODEL_USED_COL, BEST_MODEL_COMBINED_VIEW,
    ALL_TIME_AVG_COL, EIGHT_WK_AVG_COL, TREND_COL, ONHAND_COL, WOS_COL, KPI_HELP,
    MIXED_SOURCE, CONTAINER_HIST_COL, CONTAINER_FC_COL, PRODUCT_CATEGORIES,
)
from dashboard_app.summaries import (
    resolve_avg_col, avg_window_phrase, historical_window,
    historical_window_label, _format_generated_at, price_map_from_summary,
)
from dashboard_app.compute import (
    compute_by_customer_best, _agent_summaries_mtime, _agent_summaries_oldest_at,
    attach_supply_columns, summary_to_excel, with_export_flags,
)
from dashboard_app.refresh import batch_in_progress
from dashboard_app.charts import (
    chart_range_control, chart_range_preset, aggregate_chart, sku_chart,
    customer_share_donut,
)
from dashboard_app.tables import FIXED_FILTER_LABELS, render_selectable_table

# The summary table's condensed row: the five columns a planner scans. Every
# other field is one click away in the detail card, and the Excel download still
# ships the full frame.
BEST_MIX_CONDENSED_COLS = ["SKU", "Customer Grouping", EIGHT_WK_AVG_COL,
                           "Current Projection Average", RISK_COL]
# The KPI tiles on each detail card. A SET, not a sequence — config.kpi_sort orders
# them, so a field sits in the same place here as on every other view's card.
# Identifies which group's card this is (one SKU can have several open at once, one
# per customer group), then everything the condensed row has no space for.
#
# Forecast/money fields used to render as st.metric in a column beside the chart
# inside render_sku_detail_card; they are tiles here now, so the card has ONE KPI
# zone instead of two. Projected Revenue is derived rather than a column and comes
# from projection_kpi_extras.
BEST_MIX_CARD_COLS = [
    "Customer Grouping", MODEL_USED_COL, "Data Source",
    "Weeks with data", ALL_TIME_AVG_COL, EIGHT_WK_AVG_COL, TREND_COL,
    "Current Projection Average", "Updated Projection Average",
    "Projection Difference",
    PRICE_COL, RISK_COL,
    ONHAND_COL, WOS_COL,
]


def _render_kpis(summary, agg, anchors, stacked=False, avg_col=None,
                 show_sku_count=True):
    """Render the 7-metric KPI row shared by every view.

    Uses only ``summary`` + the SKU-week ``agg`` + the week ``anchors``. SKU
    counts use ``nunique`` (not row count) so the Optimal Projections combined
    view — which carries one row per (SKU, Customer Grouping) — reports distinct
    SKUs; for single-model views SKU is unique per row, so this is unchanged.

    ``stacked`` lays the seven metrics out vertically (one per line) instead of
    across a 7-column row, so they fit a narrow side column like the SKU/Customer
    detail charts. The trailing informational captions are shown only in the wide
    row layout.

    ``show_sku_count`` drops the leading "SKUs Forecasted" tile, for a caller whose
    ``summary`` is a SINGLE SKU — the count can only ever read 1 there, and a tile
    with one possible value is noise. Meant for ``stacked=True`` callers: in the
    wide seven-column layout it would leave an empty column.

    ``avg_col`` names the descriptive-average column whose window label describes
    ``anchors``, for the total-weekly-demand metric's label and help text. It has to
    be passed explicitly by any caller whose ``summary`` carries BOTH averages
    (``attach_descriptive_averages`` puts All-Time first, so the ``resolve_avg_col``
    fallback would claim an all-time window even when ``anchors`` spans 8 weeks).
    """
    lb, lcw, ffw = anchors
    # Avg. weekly demand = the mean of the TOTAL weekly demand actually plotted
    # on the chart's "Actual demand" line (POS/Orders summed across SKUs per
    # week, then averaged over the weeks in the window). Do NOT sum the per-SKU
    # "N-Week POS/Orders Average" column here: that per-SKU average divides each
    # SKU by its own weeks-with-data, so summing it counts a SKU that sold in
    # only a few weeks as if it sold every week and overstates the total.
    n_skus = int(summary["SKU"].nunique())
    avg_col = avg_col or resolve_avg_col(summary)
    hist_demand = historical_window(agg, summary, (lb, lcw, ffw))
    weekly_totals = hist_demand.groupby("WeekDate")["demand"].sum(min_count=1)
    total_avg = float(weekly_totals.mean()) if not weekly_totals.empty else 0.0
    total_updated = summary["Updated Projection Average"].sum()
    total_initial = summary["Current Projection Average"].sum()
    diff = total_updated - total_initial
    # Total Projection Value = Σ (list price × updated weekly-avg forecast) over
    # priced SKUs. Unpriced SKUs map to NaN and are skipped, so this covers the
    # same population as Revenue Risk. Per-week basis (Updated Projection Average
    # is already a weekly mean).
    has_price = PRICE_COL in summary.columns and summary[PRICE_COL].notna().any()
    proj_value = (
        (summary[PRICE_COL] * summary["Updated Projection Average"]).sum()
        if has_price else None
    )
    # Count DISTINCT Orders SKUs, not rows: the Optimal Projections combined
    # table carries one row per (SKU, Customer Grouping), so a row-sum would
    # count an Orders SKU once per group and blow past n_skus (distinct SKUs).
    n_orders = int(summary.loc[summary["Data Source"] == "Orders", "SKU"].nunique()) \
        if "Data Source" in summary.columns else 0

    # Wide layout: seven side-by-side columns. Stacked: render straight into the
    # current container (st) so each metric sits on its own line.
    # The keyed container tags the wide KPI row with a `.st-key-kpi_bubble_row`
    # CSS class so the stylesheet can make just these bubbles equal-height (the
    # stacked per-SKU/side metrics are intentionally left untouched).
    if stacked:
        k1, k2, k3, k4, k5, k6, k7 = [st] * 7
    else:
        with st.container(key="kpi_bubble_row"):
            k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    if show_sku_count:
        k1.metric(
            "SKUs Forecasted", f"{n_skus:,}",
            help=f"{n_orders} forecast from Orders (no POS)" if n_orders else None,
        )
    # "Total Weekly Demand", not "Historical Demand": this is a VIEW TOTAL and is
    # deliberately not the sum of the per-SKU average column (see the note above on
    # why summing that column would overstate the total). Naming it after the total
    # keeps it from being read as the same metric a table row shows.
    #
    # The window is in the LABEL, not just the help: it follows the selected model
    # (8 weeks vs all history), so two runs of this row can show very different
    # numbers under an otherwise identical caption.
    window = historical_window_label(avg_col)
    k2.metric(
        f"Total Weekly Demand ({window} avg)", f"{total_avg:,.0f}",
        help=f"Mean of TOTAL weekly actual demand (POS/Orders) summed across every "
             f"SKU, over the {avg_window_phrase(avg_col).lower()} window — the "
             f"average of the chart's actual-demand line, and the window the "
             f"selected model fits on. Not the sum of the per-SKU "
             f"'{window} POS/Orders Average' column below: that column divides each "
             f"SKU by its own span, so summing it would count a SKU that only sold "
             f"for part of the window as if it sold throughout.",
    )
    k3.metric(
        "Current Forecast (avg/wk)", f"{total_initial:,.0f}",
        help="Mean of the existing system projection over the forecast horizon "
             "(the 15 future weeks) — the average of the chart's original-"
             "projection line over the forecast window.",
    )
    k4.metric(
        "Updated Forecast (avg/wk)", f"{total_updated:,.0f}",
        help="Mean of this model's updated forecast over the 15 future weeks — "
             "the average of the chart's updated-forecast line.",
    )
    k5.metric(
        "Projection Difference (avg/wk)", f"{diff:+,.0f}",
        delta=f"{(diff / total_initial * 100):+.1f}%" if total_initial else None,
    )
    has_risk = RISK_COL in summary.columns and summary[RISK_COL].notna().any()
    if has_risk:
        net_risk = summary[RISK_COL].sum()
        k6.metric(
            "Revenue Risk (avg/wk)", fmt_dollar(net_risk, signed=True),
            help="Σ (projection difference × list price) over priced SKUs. "
                 "Negative = forecast fell below the original projection.",
        )
    else:
        k6.metric(
            "Revenue Risk (avg/wk)", "—",
            help="Load a list_prices_*.xlsx (sidebar) to enable revenue risk.",
        )
    if proj_value is not None:
        k7.metric(
            "Projected Revenue (avg/wk)", fmt_dollar(proj_value),
            help="Σ (list price × updated weekly-avg forecast) over priced SKUs "
                 "— the gross value at list price of the forecasted weekly demand.",
        )
    else:
        k7.metric(
            "Projected Revenue (avg/wk)", "—",
            help="Load a list_prices_*.xlsx (sidebar) to enable projection value.",
        )
    if not stacked and n_orders:
        st.caption(
            f"⚑ {n_orders} of {n_skus} SKUs had no POS in the window and "
            "were forecast from Orders."
        )
    if not stacked and PRICE_COL in summary.columns:
        n_noprice = int(summary.drop_duplicates("SKU")[PRICE_COL].isna().sum())
        if n_noprice:
            st.caption(
                f"💲 {n_noprice} of {n_skus} SKUs have no list price; "
                "their revenue risk is left blank."
            )


# --------------------------------------------------------------------------- #
# Container demand: weekly units restated in the unit planners actually order   #
# --------------------------------------------------------------------------- #

def _as_container_load(container_load):
    """``container_load`` as a str-indexed Series, or None when unusable.

    Accepts the Series ``container_load_from_plytix`` returns or a plain dict.
    Never mutates the input: the map is derived once in ``dashboard.main`` and
    handed to every section, so reindexing it in place would edit shared state
    from inside a render function.
    """
    if container_load is None:
        return None
    load = (container_load if isinstance(container_load, pd.Series)
            else pd.Series(container_load, dtype="float64"))
    if load.empty:
        return None
    return load.set_axis(load.index.astype(str))


def _container_breakdown(frame, value_col, container_load):
    """Weekly container demand for ``frame``, as a total AND its per-SKU parts.

    Returns ``(total, per_sku, n_covered, n_total)`` where ``per_sku`` is a
    SKU -> containers/week Series that **sums exactly to** ``total``.

    That tie is the reason this returns both together instead of offering two
    functions. The total is computed PER WEEK and then averaged: Σ over SKUs of
    (units ÷ that SKU's load) for each week, meaned over the weeks present. The
    obvious way to write the per-SKU column — each SKU's own mean ÷ its load —
    divides a SKU that sold in 3 of 8 weeks by 3 while the total divides it by 8,
    so the column would not add up to the tile beside it and the table would read
    as broken. Both figures here are ``containers ÷ n_weeks`` over the SAME
    ``n_weeks``, so they agree by construction rather than by coincidence.

    The same reasoning as the Total Weekly Demand tile in ``_render_kpis``: a
    per-SKU average divides each SKU by its own weeks-with-data, so summing those
    would count a SKU that sold in a few weeks as if it sold in all of them.

    The division has to happen PER SKU before the weekly sum: Container Load is a
    per-SKU constant (a container holds 46,430 of CW0166 but 393 of ST2030), so
    there is no such thing as a category-level load to divide a category total by.

    ``container_load`` is the SKU -> units-per-container map from
    ``agent.data_io.container_load_from_plytix`` (a Series; a plain dict also
    works). SKUs missing from it map to NaN and drop out, exactly the way unpriced
    SKUs drop out of Revenue Risk — the figure then covers a subset, and the caller
    says which.

    ``total`` is None when no SKU in the frame has a usable load, so the caller can
    render an em dash rather than a zero: "no Container Load on file" must not read
    as "zero containers". ``per_sku`` is then an empty Series, never zeros.
    """
    empty = pd.Series(dtype="float64")
    load = _as_container_load(container_load)
    if load is None or frame is None or getattr(frame, "empty", True):
        return None, empty, 0, 0
    if value_col not in frame.columns or "SKU" not in frame.columns:
        return None, empty, 0, 0

    skus = frame["SKU"].astype(str)
    units = pd.to_numeric(frame[value_col], errors="coerce")
    # .where(> 0): a zero or negative load would divide to inf, which would then
    # dominate the weekly sum. container_load_from_plytix already filters those
    # out; this keeps the helper honest against a dict passed in by a caller.
    per_sku_load = pd.to_numeric(skus.map(load), errors="coerce")
    containers = units / per_sku_load.where(per_sku_load > 0)

    has_val = units.notna()
    keep = containers.notna()
    n_total = int(skus[has_val].nunique())
    n_covered = int(skus[has_val & keep].nunique())
    if not keep.any():
        return None, empty, 0, n_total

    kept = containers[keep]
    weeks = pd.to_datetime(frame.loc[keep, "WeekDate"])
    n_weeks = int(weeks.nunique())
    if not n_weeks:
        return None, empty, 0, n_total

    total = float(kept.sum()) / n_weeks
    per_sku = kept.groupby(skus[keep]).sum() / n_weeks
    return total, per_sku, n_covered, n_total


def _weekly_containers(frame, value_col, container_load):
    """Mean weekly container count for ``frame`` — the total half of
    ``_container_breakdown``, for callers that need no per-SKU split.

    Returns ``(containers, n_covered, n_total)``.
    """
    total, _, n_covered, n_total = _container_breakdown(
        frame, value_col, container_load)
    return total, n_covered, n_total


def _container_help(col, hist_frame):
    """``KPI_HELP`` for a container tile, plus the exact weeks behind it.

    Naming the real first and last week matters for the historical tile because a
    preset can be CLIPPED by short history: "1 Year" on a SKU with four months of
    data is a four-month average, and only the dates reveal that.
    """
    base = KPI_HELP.get(col)
    if col != CONTAINER_HIST_COL or hist_frame is None or hist_frame.empty:
        return base
    weeks = pd.to_datetime(hist_frame["WeekDate"])
    if weeks.empty:
        return base
    span = (f"Weeks used: {weeks.min().date()} → {weeks.max().date()} "
            f"({weeks.nunique()} weeks).")
    return f"{base} {span}" if base else span


# How a Date-range preset reads inside a tile label. Only the wording differs from
# charts.RANGE_PRESETS' keys; anything not listed falls through unchanged.
_RANGE_LABELS = {
    "1 Month": "1-Month", "3 Months": "3-Month", "6 Months": "6-Month",
    "9 Months": "9-Month", "1 Year": "1-Year", "2 Years": "2-Year",
    "3 Years": "3-Year", "All": "All-Time", "Custom…": "custom range",
}


def _range_label(preset):
    """A Date-range preset name as it should read inside a tile label."""
    return _RANGE_LABELS.get(preset, str(preset))


def _history_window_from_range(date_range, anchors):
    """``anchors`` re-based on the chart's selected date range, for history math.

    ``chart_range_control`` returns ``end`` as the FORECAST horizon in every branch —
    presets trim history only, so the forecast always stays visible. Handing that
    tuple straight to ``summaries.historical_window`` would therefore drag forecast
    weeks into a historical slice. Clamping ``end`` to ``lcw`` (the last completed
    week) is what keeps this a history window.

    ``start`` needs no clamp: the picker already floors it at the frame's first week.
    Falls back to ``anchors`` unchanged when no range is supplied, so a caller
    without a picker keeps the model's window.
    """
    if not date_range:
        return anchors
    _, lcw, ffw = anchors
    start, end = date_range
    return pd.Timestamp(start), min(pd.Timestamp(end), lcw), ffw


def _render_container_tiles(hist_frame, weekly_frame, container_load,
                            range_label=None):
    """The two container-demand tiles, plus a coverage caption when SKUs dropped out.

    Two tiles rather than one because the question has two tenses: how many
    containers a week is the category moving NOW (historical window, the container
    twin of Total Weekly Demand), and how many the updated forecast implies going
    forward (the twin of Updated Forecast). A planner books against both, and the
    gap between them is the whole point.

    Both come from ``_weekly_containers``, so the two tiles cannot drift apart in
    method — only in input frame.

    ``range_label`` names the window ``hist_frame`` was sliced to, e.g. ``"6-Month"``,
    and makes the historical tile read "Container Demand (6-Month avg/wk)". That tile
    FOLLOWS the chart's Date range selector, so it has to say which window it is on —
    a bare "hist" would leave two differently-windowed averages sitting in one column
    with nothing to tell them apart. The forecast tile takes no such label: the picker
    only ever trims history, so its window is the same at every preset.

    The DISPLAYED label is dynamic but ``KPI_HELP`` is keyed by the stable constant,
    which is also the column header in the SKU listing and its Excel export — a
    downloaded workbook keeps a fixed schema whatever the picker was set to.
    """
    hist, _, _ = _weekly_containers(hist_frame, "demand", container_load)
    fc, fc_covered, fc_total = _weekly_containers(
        weekly_frame, "projected_pos", container_load)
    hist_label = (f"Container Demand ({range_label} avg/wk)"
                  if range_label else CONTAINER_HIST_COL)
    for col, label, val in ((CONTAINER_HIST_COL, hist_label, hist),
                            (CONTAINER_FC_COL, CONTAINER_FC_COL, fc)):
        if val is None:
            st.metric(
                label, "—",
                help="Load a list_prices_*.xlsx or refresh the Plytix feed "
                     "(sidebar) to enable container demand — it needs each SKU's "
                     "Container Load.",
            )
        else:
            # Two decimals: a single SKU's weekly demand is often a fraction of a
            # container, and rounding that to 0.0 would hide the number entirely.
            st.metric(label, f"{val:,.2f}", help=_container_help(col, hist_frame))
    missing = fc_total - fc_covered
    if fc is not None and missing:
        st.caption(
            f"📦 {missing} of {fc_total} SKUs have no Container Load in "
            "the Plytix export; they are left out of the two container tiles."
        )


def render_sku_detail_card(agg_by_group, weekly_by_group, anchors, chart_anchors,
                           pm, row, key_base, model_label=None, top_groups=None):
    """Detail-card body for one (SKU, Customer Grouping) row of a summary table.

    Shared by Optimized Projections and Quick Projections so the two views read
    identically: a date-range picker + per-SKU chart, FULL WIDTH, scoped to the
    single group on the clicked row.

    This used to be a chart-left / metrics-right split, which put the card's KPIs in
    two places at once — the tile grid ``_render_row_detail`` draws above, plus seven
    ``st.metric`` calls here — with ``Data Source`` appearing in both. All seven now
    live in that one grid (they read straight off ``row``; see ``QUICK_CARD_COLS`` /
    ``BEST_MIX_CARD_COLS``), except ``Projected Revenue``, which is derived rather
    than a column and comes through ``projected_revenue_kpi`` below. The chart gets
    the whole card width as a result.

    ``(row, key_base)`` sit where ``render_selectable_table``'s ``detail_chart``
    contract passes them — positionally, and last of the required args. Callers
    bind the leading five with functools.partial and the two trailing options **by
    keyword**, so a positional ``row`` can never land in ``model_label``.

    Two per-view differences are parameters:

    * ``model_label`` names the model behind the row for views whose table has no
      ``MODEL_USED_COL`` (Quick Projections fits one chosen model for the whole
      view). Optimized leaves it None and the column on the row wins.
    * ``top_groups`` maps ``str(SKU)`` -> the SKU's top-volume customer-group
      breakdown. Only the combined and region-rollup Quick views have one (it
      comes from ``compute_view``'s ``breakdown_df``, which the per-group fits
      never pass), and it is a whole-view figure, not this group's share — the
      caption says so.

    The card computes no weekly-demand average of its own. Both averages are already
    on the row and rendered as tiles, so recomputing one here only created a second
    number that disagreed with the column beside it — same window, but dividing by
    weeks that had a row instead of the SKU's full span.
    """
    sku = str(row["SKU"])
    group = str(row["Customer Grouping"])
    _, lcw, _ = anchors
    # Per-card widget keys: several cards (same SKU, different groups) can be
    # open at once, each with its own independent range picker and chart.
    key = re.sub(r"[^0-9A-Za-z_]+", "_", f"{key_base}__sku__{sku}__{group}")

    # The group's frames, NOT pre-filtered to this SKU: sku_chart filters by SKU
    # internally, and chart_range_control's history floor then matches what the
    # old section showed when a single customer group was picked.
    sku_agg = agg_by_group[agg_by_group["Customer Grouping"].astype(str) == group]
    sku_weekly = weekly_by_group[
        weekly_by_group["Customer Grouping"].astype(str) == group
    ]
    if sku_agg.empty or sku_weekly.empty:
        st.caption("No weekly data for this SKU / customer group.")
        return

    desc = row["Description"] if isinstance(row.get("Description"), str) else ""
    # One row → exactly one source (the old section's "(mixed)" case can't arise).
    source = row["Data Source"] if isinstance(row.get("Data Source"), str) else "POS"

    sku_range = chart_range_control(sku_agg, sku_weekly, lcw, key=key)
    st.plotly_chart(
        sku_chart(sku, desc, source, sku_agg, sku_weekly, chart_anchors,
                  date_range=sku_range, prices=pm),
        width="stretch", key=f"{key}_plot",
    )
    # The row's own model wins when the table carries one (Optimized); else the
    # caller's single-model label, and if neither, just name the group.
    model = row[MODEL_USED_COL] if MODEL_USED_COL in row.index else model_label
    st.caption(
        f"Customer group **{group}** — forecast with {model}."
        if model else f"Customer group **{group}**."
    )
    if top_groups:
        breakdown = top_groups.get(sku)
        if breakdown:
            st.markdown("**Top Volume Groups**")
            st.caption(breakdown)
            st.caption(
                ":gray[Across all customer groups in this view, not this "
                "group's share.]"
            )


def _sku_detail_source(summary_s):
    """The ``Data Source`` label for one SKU's chart title.

    One value when the SKU's rows agree, ``MIXED_SOURCE`` when they don't. Quick
    passes a SKU-grain roll-up, where ``roll_up_summary`` has already collapsed
    disagreement into exactly that constant; Optimized passes its SKU × customer
    frame, where the groups can still disagree and the reduction has to happen here.
    Defaults to POS when the column is missing or empty, as the section always has.
    """
    if "Data Source" not in summary_s.columns:
        return "POS"
    vals = [v for v in summary_s["Data Source"].dropna().unique() if isinstance(v, str)]
    if not vals:
        return "POS"
    return vals[0] if len(vals) == 1 else MIXED_SOURCE


def render_sku_detail_section(summary, agg, weekly, by_cust, anchors, prices,
                              container_load=None, avg_col=None, key="sku"):
    """The ``SKU detail`` section: ONE SKU's total weekly demand across every
    customer group in the view — the order-sizing view of the page ("how many units
    of this SKU do I need per week?"). The mirror of ``Customer detail``, drilled the
    other way.

    Shared by Quick Projections and Optimized Projections so the two read identically
    (the same reason ``render_sku_detail_card`` is shared). The numbers are the
    ROLL-UP of that SKU's per-customer forecasts — the same figures behind the KPI
    row, the aggregate chart and the by-customer table — so the tiles here and the
    donut beside them are the same number twice, once summed and once itemised. They
    tie exactly, and ``tests/test_rollup_ties.py`` holds them to it.

    ``summary`` supplies the SKU list and the KPI tiles; ``by_cust`` the customer
    breakdown and the supply figures. The two frames are at different grains per view
    and that is deliberate:

    ==========  =============================  ==================
    argument    Quick passes                   Optimized passes
    ==========  =============================  ==================
    summary     ``summary`` (SKU grain)        ``combined`` (SKU × customer)
    by_cust     ``by_cust``                    ``combined``
    ==========  =============================  ==================

    ``_render_kpis`` counts SKUs with ``nunique`` and sums the projection columns, so
    a SKU × customer frame scoped to one SKU yields precisely its roll-up — which is
    why Optimized needs no per-SKU summary frame of its own. Building one would be a
    second path to the same number, and two paths are how two numbers start.

    ``container_load`` is the SKU -> units-per-container map (Plytix). Optional: the
    two container tiles simply read an em dash without it, the same way Revenue Risk
    does without list prices.

    ``key`` namespaces the widgets (``{key}_sku`` selector, ``range_sku_{key}`` range
    picker) so both views can be open in one session without colliding.
    """
    _, lcw, ffw = anchors
    st.markdown("### SKU detail")
    st.caption(
        "One SKU's total weekly demand across every customer group in this view — "
        "the sum of that SKU's per-customer forecasts, so it ties exactly to its "
        "rows in the breakdown below and in the by-customer table."
    )

    # Label carries the description so the search box matches on either; the stored
    # value stays the raw SKU (the same value-vs-label split as quick_group_label on
    # the Customer-group selector). drop_duplicates because `summary` is one row per
    # SKU on Quick but one row per (SKU, customer) on Optimized.
    desc_by_sku = (
        dict(zip(summary.drop_duplicates("SKU")["SKU"].astype(str),
                 summary.drop_duplicates("SKU")["Description"]))
        if "Description" in summary.columns else {}
    )

    def _sku_label(s):
        # No .strip(): the warehouse's fixed-width padding comes off at ingestion
        # (agent.data_io._clean), and a description that was nothing but padding
        # arrives as "" and falls back to the bare SKU here.
        desc = desc_by_sku.get(s)
        return f"{s} — {desc}" if isinstance(desc, str) and desc else str(s)

    skus = sorted(summary["SKU"].astype(str).unique())
    sku = st.selectbox("SKU", skus, key=f"{key}_sku", help="Type to search",
                       format_func=_sku_label)
    summary_s = summary[summary["SKU"].astype(str) == sku]
    agg_s = agg[agg["SKU"].astype(str) == sku]
    weekly_s = weekly[weekly["SKU"].astype(str) == sku]

    if agg_s.empty or weekly_s.empty:
        st.caption("No weekly data for this SKU in this snapshot.")
        return

    desc = desc_by_sku.get(sku) if isinstance(desc_by_sku.get(sku), str) else ""
    source = _sku_detail_source(summary_s)
    # Chart-only history floor, per SKU — same reasoning as the view's chart_anchors
    # (`lb` is as short as 8 weeks under the 8-Week Moving Average model, which would
    # trap the range picker inside that window), but keyed to this SKU's first week.
    sku_anchors = (pd.to_datetime(agg_s["WeekDate"]).min(), lcw, ffw)

    scL, scR = st.columns([3, 1])
    with scL:
        # Own key => this picker is independent of the aggregate / customer ones.
        # Handed the SKU-sliced frames (rather than the full ones, which sku_chart
        # would filter itself) so the picker's history floor is this SKU's first
        # week, not the view's.
        sku_range = chart_range_control(agg_s, weekly_s, lcw, key=f"range_sku_{key}")
        st.plotly_chart(
            sku_chart(sku, desc, source, agg_s, weekly_s, sku_anchors,
                      date_range=sku_range, prices=prices),
            width="stretch",
        )

        # --- Customer group breakdown --------------------------------------
        # Where this SKU's volume comes from. These are the PER-GROUP fits, so
        # they sum to the SKU's rows in the by-customer table below.
        #
        # A donut rather than a table: the question here is a SHARE ("where does
        # this SKU's volume come from?"), which a table makes the reader compute.
        # Every column such a table would carry is a row of the slice's hover
        # instead — see charts.customer_share_donut.
        #
        # Inside the chart column, not full width below the row: it is part of
        # THIS section, and drawn across the whole page a 460px figure is a small
        # circle in a wide empty band.
        st.markdown("#### Customer group breakdown")
        bd = (by_cust[by_cust["SKU"].astype(str) == sku]
              if by_cust is not None and not by_cust.empty else None)
        share_fig = (customer_share_donut(bd)
                     if bd is not None and not bd.empty else None)
        if bd is None or bd.empty:
            st.caption("No per-customer forecasts for this SKU in this snapshot.")
        elif share_fig is None:
            st.caption(
                "Every customer group's updated forecast for this SKU is zero, "
                "so there is no share to chart."
            )
        else:
            st.plotly_chart(share_fig, width="stretch")
            n_groups = int(
                (pd.to_numeric(bd["Updated Projection Average"],
                               errors="coerce") > 0).sum()
            )
            st.caption(
                "Each customer group's share of this SKU's updated forecast. "
                "Hover a slice for that group's full figures — current and "
                "updated forecast, 8-week run rate, projection difference, "
                "revenue risk and data source. These are the parts the tiles "
                "beside it are the sum of; the two tie exactly."
                + (f" Groups past the largest 8 (of {n_groups}) are folded into "
                   "one grey slice, whose hover totals them."
                   if n_groups > 9 else "")
            )
    with scR:
        # The same metrics as the top of the view, scoped to this SKU and stacked to
        # fit the side column. Section anchors (not the widened chart range) so the
        # historical-demand window lines up with the KPI row above.
        #
        # Minus "SKUs Forecasted", which is 1 by construction here. Its help text
        # carried the "forecast from Orders (no POS)" flag; for a single SKU that
        # fact is still on screen twice — sku_chart names the source in its title
        # and axis, and the breakdown hover carries Data Source per customer group.
        _render_kpis(summary_s, agg_s, anchors, stacked=True, avg_col=avg_col,
                     show_sku_count=False)
        # On Hand / Weeks of Supply — the two figures that turn a weekly forecast
        # into an order quantity. Both are SKU-level constants across a SKU's
        # customer rows (see attach_supply_columns), so this is a lookup, not a
        # second computation. Absent when no warehouse snapshot was loaded:
        # "unknown stock" must not read as zero.
        if bd is not None and not bd.empty:
            for col, fmt in ((ONHAND_COL, "{:,.0f}"), (WOS_COL, "{:,.1f}")):
                if col not in bd.columns:
                    continue
                vals = bd[col].dropna()
                if vals.empty:
                    continue
                st.metric(col, fmt.format(vals.iloc[0]), help=KPI_HELP.get(col))
        # This SKU's weekly demand restated in containers — the unit the order is
        # actually placed in.
        #
        # This one DELIBERATELY diverges from the tiles above it: it follows the
        # chart's Date range selector rather than the model's window. Containers are
        # booked against what is selling over a horizon the planner chooses, and the
        # model's window is either 8 weeks or 3 years with nothing in between. The
        # tiles above keep the model window because their labels name it — "Total
        # Weekly Demand (8-Week avg)" has to be an 8-week average. This tile's label
        # names the selected range instead, so both are honest about their window.
        sku_hist = historical_window(
            agg_s, summary_s, _history_window_from_range(sku_range, anchors))
        _render_container_tiles(
            sku_hist, weekly_s, container_load,
            range_label=_range_label(chart_range_preset(f"range_sku_{key}")))


def _sheet_slug(category):
    """A category name as a legal Excel sheet name.

    Excel caps sheet names at 31 characters and rejects ``[]:*?/\\`` — the same
    truncation ``exceptions.py`` applies to its per-slug exports.
    """
    slug = "".join("-" if c in '[]:*?/\\' else c for c in str(category))
    return slug.strip()[:31] or "category"


def _file_slug(category):
    """A category name as a filename fragment: lowercase, no spaces or separators."""
    slug = "".join("_" if c in ' []:*?/\\' else c for c in str(category).lower())
    return slug.strip("_") or "category"


def _category_members(summary, category):
    """The SKUs of ``category`` that are actually present in ``summary``.

    The curated list in ``PRODUCT_CATEGORIES`` is snapshot-independent; this
    intersects it with what the loaded snapshot and the active view really carry, so
    a US-only view offers the liners that sell in the US rather than 76 rows of
    which half are empty.
    """
    members = PRODUCT_CATEGORIES.get(category, frozenset())
    present = set(summary["SKU"].astype(str).unique())
    return sorted(present & set(members))


def render_category_detail_section(summary, agg, weekly, by_cust, anchors, prices,
                                   container_load=None, today_str="", avg_col=None,
                                   key="cat"):
    """The ``Category detail`` section: a PRODUCT GROUP's total weekly demand across
    every customer group in the view — SKU detail drilled one level out.

    The question this answers is the one that sizes a container booking or a factory
    run: "what is the whole liners business doing?" Answering it from SKU detail
    means opening 76 panes and adding them up by hand.

    Deliberately assembled from the SAME pieces as the rest of the page rather than
    new ones — ``_render_kpis`` for the tiles, ``charts.aggregate_chart`` for the
    chart, ``chart_range_control`` for the picker. ``aggregate_chart`` is already
    exactly this shape: it sums actuals (via the resolved ``demand`` column), the
    updated forecast and the snapshot's original ``Projection`` across whatever SKUs
    it is handed. Scoped to a category's SKUs it IS a category chart, with no new
    code and no second way for these numbers to be computed.

    ``summary``/``by_cust`` arrive at different grains per view, exactly as they do
    for ``render_sku_detail_section`` — Quick passes SKU-grain ``summary`` plus
    ``by_cust``; Optimized passes ``combined`` for both. ``_render_kpis`` sums the
    projection columns, so either grain scoped to the category yields its roll-up.

    ``today_str`` dates the Excel export's filename, the same way every other
    download button on the page is dated.

    NOTE on scope: ``PRODUCT_CATEGORIES`` lists ACTIVE SKUs only, so these totals are
    deliberately not the sum of every liner row in the table below — discontinued
    liners still present in the snapshot are excluded. The caption says so, so the
    difference reads as intent rather than as a bug.
    """
    _, lcw, ffw = anchors
    st.markdown("### Category detail")

    # Only offer categories this view actually has SKUs for: an empty entry is a dead
    # end, and on a regional view a category can legitimately vanish.
    cats = [c for c in PRODUCT_CATEGORIES if _category_members(summary, c)]
    if not cats:
        st.caption("No product category has SKUs in this view.")
        return

    def _cat_label(c):
        return f"{c} ({len(_category_members(summary, c))} SKUs)"

    category = st.selectbox("Category", cats, key=f"{key}_category",
                            help="Type to search", format_func=_cat_label)
    members = _category_members(summary, category)
    st.caption(
        f"Every active SKU in {category}, summed across every customer group in "
        "this view — the one roll-up the tiles, the chart and the SKU list below "
        "all read from. Scoped to SKUs Plytix marks Active, so it deliberately "
        f"leaves out discontinued {category.lower()} still present in the "
        "snapshot: it will not tie to the sum of every matching row in the table "
        "further down."
    )

    def in_cat(frame):
        return frame[frame["SKU"].astype(str).isin(members)]

    summary_c = in_cat(summary)
    agg_c = in_cat(agg)
    weekly_c = in_cat(weekly)

    if agg_c.empty or weekly_c.empty:
        st.caption("No weekly data for this category in this snapshot.")
        return

    # Chart-only history floor, same reasoning as the SKU section's sku_anchors: `lb`
    # is as short as 8 weeks under the 8-Week Moving Average model, which would trap
    # the range picker inside that window. The KPIs keep the section `anchors`.
    cat_anchors = (pd.to_datetime(agg_c["WeekDate"]).min(), lcw, ffw)

    ccL, ccR = st.columns([3, 1])
    with ccL:
        cat_range = chart_range_control(agg_c, weekly_c, lcw, key=f"range_cat_{key}")
        # Derived ONCE, here, because both halves of the section read it: the per-SKU
        # container columns in the listing below, and the container tiles in the right
        # column. Two calls would be two chances for the table and the tile it sums to
        # to describe different windows.
        #
        # It has to sit INSIDE ccL rather than above the split, because it depends on
        # cat_range, which the picker above only produces once this column runs. ccL
        # executes before ccR, so the frame is ready by the time the tiles are drawn.
        cat_hist_window = _history_window_from_range(cat_range, anchors)
        hist_c = historical_window(agg_c, summary_c, cat_hist_window)
        st.plotly_chart(
            aggregate_chart(agg_c, summary_c, weekly_c, cat_anchors, category,
                            date_range=cat_range, prices=prices),
            width="stretch",
        )

        # --- The SKUs behind the number ------------------------------------
        # Collapsed by default: this is the audit trail for the tiles, not something
        # to read every visit. A table rather than the SKU section's donut — with 76
        # members a share chart is 76 unreadable slivers, and the question here is
        # "which SKUs are in this, and what does each contribute?".
        with st.expander(f"SKUs in {category} ({len(members)})"):
            cols = [c for c in ("SKU", "Description", EIGHT_WK_AVG_COL,
                                "Current Projection Average",
                                "Updated Projection Average")
                    if c in summary_c.columns]
            listing = summary_c.drop_duplicates("SKU")[cols].copy()
            skus_col = listing["SKU"].astype(str)
            load = _as_container_load(container_load)
            if load is not None:
                listing["Container Load"] = skus_col.map(load)
                # Each SKU's share of the two container tiles beside the chart. From
                # _container_breakdown, so these columns SUM to those tiles exactly
                # (see its docstring for why a per-SKU mean would not).
                for col, frame, value_col in (
                    (CONTAINER_HIST_COL, hist_c, "demand"),
                    (CONTAINER_FC_COL, weekly_c, "projected_pos"),
                ):
                    _, per_sku, _, _ = _container_breakdown(
                        frame, value_col, container_load)
                    listing[col] = skus_col.map(per_sku)
            if "Updated Projection Average" in listing.columns:
                listing = listing.sort_values("Updated Projection Average",
                                              ascending=False)
            st.caption(
                "The rows the tiles beside the chart are the sum of — including the "
                "two container columns, which add up to the container tiles. The "
                "historical container column covers the same Date range as the "
                "chart, so it moves with that selector; the forecast one does not. "
                "A blank Container Load means that SKU sits out all three."
            )
            st.dataframe(listing, width="stretch", hide_index=True)
            st.download_button(
                f"⬇️ Download the {category} SKU list",
                data=summary_to_excel(with_export_flags(listing),
                                      sheet_name=_sheet_slug(category)),
                file_name=f"{_file_slug(category)}_skus_{today_str}.xlsx",
                mime="application/vnd.openxmlformats-officedocument."
                     "spreadsheetml.sheet",
                # Namespaced by `key`: this section renders on both Quick and
                # Optimized, and a bare key would collide across the two.
                key=f"dl_category_skus_{key}",
            )
    with ccR:
        # The same seven metrics as the top of the view, scoped to the category and
        # stacked to fit the side column. Section anchors (not the widened chart
        # range) so the historical window lines up with the KPI row above.
        #
        # SKUs Forecasted is KEPT here, unlike the SKU section where it is 1 by
        # construction: "how many SKUs is this" is the first thing a reader needs in
        # order to size everything under it.
        _render_kpis(summary_c, agg_c, anchors, stacked=True, avg_col=avg_col)

        # On Hand / WOS at CATEGORY grain. Neither can be the SKU section's
        # `.iloc[0]` lookup: both are SKU-level constants repeated on every one of a
        # SKU's customer rows, so de-duplicate on SKU before summing, or a SKU sold
        # to five groups contributes its stock five times (the same trap
        # exceptions._sum_distinct_skus exists for).
        #
        # WOS is then RE-DERIVED from the two category totals rather than averaged
        # over the per-SKU column: a mean of ratios is not the ratio of the totals,
        # and it is the latter that answers "how long does this category's stock
        # last". Same formula as compute.attach_supply_columns, one level up.
        bd = in_cat(by_cust) if by_cust is not None and not by_cust.empty else None
        if bd is not None and not bd.empty and ONHAND_COL in bd.columns:
            per_sku = bd.drop_duplicates("SKU")
            onhand = pd.to_numeric(per_sku[ONHAND_COL], errors="coerce")
            if onhand.notna().any():
                total_onhand = float(onhand.sum())
                st.metric(
                    ONHAND_COL, f"{total_onhand:,.0f}",
                    help=f"Total On Hand across the {int(onhand.notna().sum())} "
                         "SKUs in this category that have a warehouse figure "
                         "(counted once per SKU, not once per customer row).",
                )
                current = pd.to_numeric(
                    summary_c["Current Projection Average"], errors="coerce").sum()
                if current > 0:
                    st.metric(
                        WOS_COL, f"{total_onhand / current:,.1f}",
                        help="The category's total On Hand ÷ its total current "
                             "weekly projection — the ratio of the totals, not "
                             "the average of the per-SKU ratios.",
                    )

        # The category's weekly demand in containers. The per-SKU division happens
        # inside the helper: a container holds a different number of every SKU, so
        # there is no category-level load to divide a category total by.
        # Follows the Date range selector — see the note in SKU detail for why this
        # tile diverges from the model-window tiles above it.
        _render_container_tiles(
            hist_c, weekly_c, container_load,
            range_label=_range_label(chart_range_preset(f"range_cat_{key}")))


def projection_kpi_extras(row):
    """Derived KPI tiles for a projections-table row (the ``extra_kpis`` contract).

    Two things the row's own columns can't express:

    * **Projected Revenue** — list price × this row's updated weekly forecast. Every
      other money figure is a column; this one is a product of two, so it has to be
      computed at render time. It matches the page-top KPI of the same name, which is
      this summed over the view.
    * **% Deviation on Projection Difference** — the unit gap is meaningless without
      its base (−4 is nothing on 2,000 and fatal on 20), so the gap's tile gets the
      percentage in ``st.metric``'s delta slot, coloured by direction. The page-top
      KPI row already does exactly this; the per-row card did not.

    Returns ``[]`` when list prices aren't loaded — the same graceful degradation the
    Revenue Risk column has, rather than a tile reading "—" for every row.
    """
    tiles = []
    price = row[PRICE_COL] if PRICE_COL in row.index else None
    price = None if price is None or pd.isna(price) else price
    updated = row.get("Updated Projection Average")
    if price is not None and pd.notna(updated):
        tiles.append((
            "Projected Revenue", fmt_dollar(price * updated), None,
            KPI_HELP.get("Projected Revenue"), "stat",
        ))
    return tiles


def projection_difference_delta(row):
    """``Projection Difference`` as a percent of the current projection, or None.

    None (rather than "0.0%") when the current projection is 0 or missing: there is
    no base to be a percentage of, and showing 0% would imply agreement where there
    is actually no plan to agree with.
    """
    pdiff = row.get("Projection Difference")
    current = row.get("Current Projection Average")
    if pd.isna(pdiff) or pd.isna(current) or not current:
        return None
    return f"{pdiff / current * 100:+.1f}%"


def _render_best_model_combined(df, today_ts, today_str, prices, n_excluded_rows,
                                anchors, P=None, data_sig=None,
                                onhand_by_sku=None, container_load=None):
    """Render the BEST_MODEL_COMBINED_VIEW: per-group best-model table.

    Builds (and session-caches) the mixed table via ``compute_by_customer_best``,
    renders the winners table + a model-usage line + a download, and lists any
    groups that had no best model (no summary, or too little history to backtest)
    in a dropdown. Called from main() in place of the single-model page body. The
    page title is already rendered by main() before this branch, so we start at the
    section subheader to avoid showing it twice.
    """
    st.subheader("Optimized Projections")
    st.caption(
        "Each customer group is forecast with its own most-accurate model "
        "(from the latest model-analysis recommendations) and stitched into one "
        "table. The sidebar model choice does not apply to this view."
    )

    # Cache on a structural signature so search-box reruns don't rebuild it. The
    # agent-summaries mtime is part of the signature so the table rebuilds as soon
    # as a batch writes fresh summaries (e.g. right after "Agent Summary (all
    # views)" finishes) — without it a stale "run the batch first" result would
    # linger in this session until an unrelated structural change.
    price_marker = None if prices is None else int(len(prices))
    sig = (BEST_MODEL_COMBINED_VIEW, today_str, price_marker, n_excluded_rows,
           _agent_summaries_mtime())
    if st.session_state.get("bestmix_structural") != sig:
        prog = st.progress(0.0, text="Preparing…")
        try:
            def _bump(done, total, group):
                prog.progress(
                    min(0.05 + 0.93 * done / max(total, 1), 0.98),
                    text=f"Forecasting each group with its best model… "
                         f"({done}/{total})",
                )
            result = compute_by_customer_best(
                df, today_ts, prices, min_weeks=None, progress_cb=_bump,
                data_sig=data_sig,
            )
            prog.progress(1.0, text="Done")
        finally:
            prog.empty()
        st.session_state["bestmix_result"] = result
        # Oldest, not newest: the table stitches together every group's summary,
        # and a partially-finished batch leaves most of them from the prior run.
        # The oldest stamp says "everything here is at least this fresh."
        st.session_state["bestmix_generated_at"] = _agent_summaries_oldest_at()
        st.session_state["bestmix_structural"] = sig
    else:
        result = st.session_state.get("bestmix_result")

    combined, weekly_all, agg_all, weekly_by_group, agg_by_group, excluded = (
        result if result is not None else (None, None, None, None, None, [])
    )
    # On Hand / Weeks of Supply for the detail cards. Attached here, outside the
    # session-cached result, because On Hand comes from a separately loaded map
    # rather than from the fit (see attach_supply_columns).
    combined = attach_supply_columns(combined, onhand_by_sku)

    # Freshness caption. While a batch is rewriting summaries, the table mixes
    # freshly-recomputed and prior-run recommendations, so say so plainly rather
    # than implying the whole set just regenerated. `oldest` is the honest "as
    # of": every group in the table is at least this fresh.
    oldest = st.session_state.get("bestmix_generated_at")
    running, _ = batch_in_progress()
    if running:
        msg = ("⏳ A new recommendation run is in progress — this table currently "
               "mixes freshly updated and prior-run recommendations, and fills in "
               "as each view finishes.")
        if oldest:
            msg += (f" Oldest recommendation shown: "
                    f"{_format_generated_at(oldest)}.")
        st.caption(msg)
    elif oldest:
        st.caption(
            f"All recommendations are from {_format_generated_at(oldest)} or later."
        )

    def _render_excluded(title):
        """Dropdown listing groups left out (bullet-pointed, one per line)."""
        if not excluded:
            return
        with st.expander(f"{title} ({len(excluded)})"):
            st.caption(
                "These groups had no published summary, or too little history "
                "for any model to be backtested, so no best model could be "
                "chosen — they're left out of the table."
            )
            st.markdown("\n".join(f"- {g}" for g in excluded))

    # No group had a resolvable best model → prompt to run the batch.
    if combined is None or getattr(combined, "empty", True):
        st.warning(
            "No customer group has a recommended model yet. Click **Recommend "
            "models (all views)** in the sidebar (or run `python -m "
            "agent.batch`), then reopen this view."
        )
        _render_excluded("Groups without a best model")
        return

    # Model-usage summary: how many groups each model won.
    counts = (
        combined.drop_duplicates("Customer Grouping")[MODEL_USED_COL].value_counts()
    )
    parts = "\n".join(f"- {m} ×{c}" for m, c in counts.items())
    st.caption(f"{int(counts.sum())} groups:\n{parts}")

    _, lcw, ffw = anchors
    view_label = "Optimized Projections"
    # SKU->list-price map for the charts' "vs plan" revenue-difference hover
    # (empty when list prices weren't loaded → plain hovers).
    pm = price_map_from_summary(combined)

    # Chart-only anchors: the passed-in `anchors` come from the sidebar model's
    # week_anchors, whose lookback start (lb) is as short as 8 weeks (8-Week
    # Moving Average). That model choice is irrelevant here, so widen the charts'
    # history floor to the earliest available week — otherwise the date-range
    # picker can only narrow within an ~8-week window. KPIs keep the original
    # `anchors` so their numbers don't shift.
    chart_lb = pd.to_datetime(agg_all["WeekDate"]).min()
    chart_anchors = (chart_lb, lcw, ffw)

    # Window label for the total-weekly-demand KPI. It has to describe `anchors`,
    # which come from the SIDEBAR model's week_anchors — NOT from `combined`, which
    # carries both averages, so _render_kpis' resolve_avg_col fallback would pick
    # All-Time and mislabel an 8-week window.
    anchors_avg_col = (
        getattr(P, "AVG_COL_LABEL", EIGHT_WK_AVG_COL)
        if P is not None else None
    )

    # ----- KPIs -------------------------------------------------------------
    # Same seven metrics as every other view. The combined frame carries one row
    # per (SKU, Customer Grouping); _render_kpis counts distinct SKUs and the
    # forecast/risk totals sum naturally across a SKU's groups.
    _render_kpis(combined, agg_all, anchors, avg_col=anchors_avg_col)

    # ----- Aggregate chart --------------------------------------------------
    # Total actual demand + total forecast, summed across every group. Actuals
    # match the Executive Overview; only the forecast line differs (each group
    # uses its backtest-winning model).
    agg_ctrl, _ = st.columns([1, 2])
    with agg_ctrl:
        agg_range = chart_range_control(agg_all, weekly_all, lcw, key="range_agg_best")
    st.plotly_chart(
        aggregate_chart(agg_all, combined, weekly_all, chart_anchors, view_label,
                        date_range=agg_range, prices=pm),
        width="stretch",
    )
    st.caption(
        "Actual demand uses each SKU's forecast source (POS or Orders); where a "
        "SKU is forecast from different sources across groups, the most recent "
        "group's source labels the actual-demand line."
    )

    # ----- Per-customer detail ----------------------------------------------
    # One customer group's total weekly demand (same shape as the aggregate
    # chart, drawn from that group's un-summed per-group frames).
    st.markdown("### Customer detail")
    customers = sorted(combined["Customer Grouping"].astype(str).unique())
    customer = st.selectbox(
        "Customer", customers, help="Type to search", key="best_customer"
    )
    agg_c = agg_by_group[agg_by_group["Customer Grouping"].astype(str) == customer]
    wk_c = weekly_by_group[weekly_by_group["Customer Grouping"].astype(str) == customer]
    summary_c = combined[combined["Customer Grouping"].astype(str) == customer]
    ccL, ccR = st.columns([3, 1])
    with ccL:
        cust_range = chart_range_control(agg_c, wk_c, lcw, key="range_cust_best")
        st.plotly_chart(
            aggregate_chart(
                agg_c, summary_c, wk_c,
                (pd.to_datetime(agg_c["WeekDate"]).min(), lcw, ffw),
                customer, date_range=cust_range, prices=pm,
            ),
            width="stretch",
        )
    with ccR:
        # Same seven metrics as the top of the view, scoped to this customer group
        # and stacked to fit the side column (like the SKU detail chart). Use the
        # section's original `anchors` (not the widened chart range) so the
        # historical-demand window lines up with the combined KPI row.
        _render_kpis(summary_c, agg_c, anchors, stacked=True,
                     avg_col=anchors_avg_col)

    # ----- Per-SKU detail ---------------------------------------------------
    # The mirror of Customer detail, drilled the other way. `combined` serves as both
    # the KPI frame and the breakdown frame: scoped to one SKU it IS that SKU's
    # per-customer rows, and _render_kpis sums them — which is exactly the roll-up
    # this section shows. No gate: unlike Quick, this view always spans every group.
    render_sku_detail_section(combined, agg_all, weekly_all, combined, anchors, pm,
                              container_load=container_load,
                              avg_col=anchors_avg_col, key="best")

    # ----- Per-category detail ----------------------------------------------
    # SKU detail drilled one level out: the same tiles and the same chart over a
    # product group instead of a single SKU. `combined` again serves as both the KPI
    # frame and the breakdown frame, for the same reason it does above.
    render_category_detail_section(combined, agg_all, weekly_all, combined, anchors,
                                   pm, container_load=container_load,
                                   today_str=today_str,
                                   avg_col=anchors_avg_col, key="best")

    st.markdown("### Summary table by SKU and customer")

    # Keep each SKU's rows together; largest revenue risk first when present.
    if RISK_COL in combined.columns and combined[RISK_COL].notna().any():
        table = (
            combined.assign(_abs=combined[RISK_COL].abs())
            .sort_values(["SKU", "_abs"], ascending=[True, False], na_position="last")
            .drop(columns="_abs").reset_index(drop=True)
        )
        st.caption("Each SKU broken out by customer group; within a SKU, "
                   "largest revenue risk first (by magnitude). Click a row to "
                   "open its chart and metrics.")
    else:
        table = combined.sort_values(["SKU", "Customer Grouping"]).reset_index(drop=True)
        st.caption("Each SKU broken out by customer group. Click a row to open "
                   "its chart and metrics.")

    # Condensed rows (five scannable columns); clicking one reveals that exact
    # (SKU, customer group) combination's detail card below — the chart, date
    # range and metrics for one row of what the SKU detail section above totals. The
    # filters and the Excel download still run on the full frame.
    # model_label is left unbound: every row carries its own MODEL_USED_COL, which
    # the card prefers. top_groups is unbound too — this view has no compute_view
    # summary, so no breakdown exists (see render_sku_detail_card).
    #
    # `fixed`: SKU / Customer / Region / Key SKU, on screen from the first paint and
    # cross-filtered against each other. Data Source and Model Used were reachable
    # through the old add-filter menu and are not here — both are still on every
    # detail card and in the download, and neither is how a planner narrows this table.
    sku_card = functools.partial(render_sku_detail_card, agg_by_group,
                                 weekly_by_group, anchors, chart_anchors, pm)
    render_selectable_table(
        table, "filter_best_mix", P,
        condensed_cols=BEST_MIX_CONDENSED_COLS, style=True,
        detail_chart=sku_card, detail_cols=BEST_MIX_CARD_COLS,
        extra_kpis=projection_kpi_extras,
        kpi_deltas={"Projection Difference": projection_difference_delta},
        fixed=FIXED_FILTER_LABELS,
    )
    st.download_button(
        "⬇️ Download the combined best-model table",
        data=summary_to_excel(with_export_flags(table)),
        file_name=f"Combined_best_model_demand_projections_{today_str}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_best_mix",
    )

    _render_excluded("Groups excluded — no backtest-winning model")
