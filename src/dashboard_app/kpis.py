"""KPI row and the Optimal Projections (best-model-per-group) combined view."""
import functools
import re

import pandas as pd
import streamlit as st

from dashboard_app.config import (
    PRICE_COL, RISK_COL, fmt_dollar, MODEL_USED_COL, BEST_MODEL_COMBINED_VIEW,
)
from dashboard_app.summaries import (
    resolve_avg_col, avg_window_phrase, historical_window,
    historical_window_label, _format_generated_at, price_map_from_summary,
)
from dashboard_app.compute import (
    compute_by_customer_best, _agent_summaries_mtime, _agent_summaries_oldest_at,
    summary_to_excel, ALL_HIST_AVG_COL, EIGHT_WK_AVG_COL,
)
from dashboard_app.refresh import batch_in_progress
from dashboard_app.charts import chart_range_control, aggregate_chart, sku_chart
from dashboard_app.tables import render_selectable_table

# The summary table's condensed row: the five columns a planner scans. Every
# other field is one click away in the detail card, and the Excel download still
# ships the full frame.
BEST_MIX_CONDENSED_COLS = ["SKU", "Customer Grouping", EIGHT_WK_AVG_COL,
                           "Current Projection Average", RISK_COL]
# Field row at the top of each detail card — identifies which group's card this
# is (one SKU can have several cards open at once, one per customer group), and
# carries the long-run average that the condensed row has no space for (the
# 8-Week run-rate is in the row; both ship in the Excel download, so both must be
# visible somewhere). The chart + stacked metrics below it are drawn by
# render_sku_detail_card.
BEST_MIX_CARD_COLS = ["Customer Grouping", MODEL_USED_COL,
                      ALL_HIST_AVG_COL, "Weeks with data"]


def _render_kpis(summary, agg, anchors, stacked=False, avg_col=None):
    """Render the 7-metric KPI row shared by every view.

    Uses only ``summary`` + the SKU-week ``agg`` + the week ``anchors``. SKU
    counts use ``nunique`` (not row count) so the Optimal Projections combined
    view — which carries one row per (SKU, Customer Grouping) — reports distinct
    SKUs; for single-model views SKU is unique per row, so this is unchanged.

    ``stacked`` lays the seven metrics out vertically (one per line) instead of
    across a 7-column row, so they fit a narrow side column like the SKU/Customer
    detail charts. The trailing informational captions are shown only in the wide
    row layout.

    ``avg_col`` names the descriptive-average column whose window label describes
    ``anchors``, for the historical-demand metric's help text only. It has to be
    passed explicitly by any caller whose ``summary`` carries BOTH averages
    (``attach_descriptive_averages`` puts All-History first, so the ``resolve_avg_col``
    fallback would claim an all-history window even when ``anchors`` spans 8 weeks).
    """
    lb, lcw, ffw = anchors
    # Avg. weekly demand = the mean of the TOTAL weekly demand actually plotted
    # on the chart's "Actual demand" line (POS/Orders summed across SKUs per
    # week, then averaged over the weeks in the window). Do NOT sum the per-SKU
    # "N Week POS/Orders Average" column here: that per-SKU average divides each
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
    k1.metric(
        "SKUs Forecasted", f"{n_skus:,}",
        help=f"{n_orders} forecast from Orders (no POS)" if n_orders else None,
    )
    # The window is in the LABEL, not just the help: it follows the selected model
    # (8 weeks vs all history), so two runs of this row can show very different
    # numbers under an otherwise identical caption.
    window = historical_window_label(avg_col)
    k2.metric(
        f"{window} Historical Demand (avg/wk)", f"{total_avg:,.0f}",
        help=f"Mean of total weekly actual demand (POS/Orders) over the "
             f"{avg_window_phrase(avg_col).lower()} window — the average of the "
             f"chart's actual-demand line. This is the window the selected model "
             f"fits on."
             + (" Same window as the per-SKU 'All-History POS/Orders Average' "
                "column in the tables below." if window == "All-Time" else
                " Same window as the per-SKU '8-Week POS/Orders Average' column "
                "in the tables below."),
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


def render_sku_detail_card(agg_by_group, weekly_by_group, anchors, chart_anchors,
                           pm, row, key_base, model_label=None, top_groups=None,
                           avg_col=None):
    """Detail-card body for one (SKU, Customer Grouping) row of a summary table.

    Shared by Optimized Projections and Quick Projections so the two views read
    identically. Renders what the old dropdown-driven "SKU detail" sections
    rendered — a date-range picker + per-SKU chart on the left, the stacked
    metrics on the right — but scoped to the single group on the clicked row, so
    the SKU and customer-group selectboxes that used to drive it are gone.

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
    * ``avg_col`` names the average column describing ``anchors``, so the
      historical-demand metric can say WHICH window it covers. It must come from
      the caller: every row reaching this card carries BOTH averages, so
      resolving it from ``row`` would always answer All-History and would label
      an 8-week span "All-Time" — the precise confusion the label exists to
      prevent.
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

    cL, cR = st.columns([3, 1])
    with cL:
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
    with cR:
        # Every figure is this one (SKU, group) row's, so the metrics read
        # straight off it — no summing across a SKU's groups any more.
        st.metric("Data Source", source)
        sku_hist = historical_window(
            sku_agg[sku_agg["SKU"].astype(str) == sku],
            pd.DataFrame([{"SKU": sku, "Data Source": source}]),
            anchors,
        )
        sku_weekly_tot = sku_hist.groupby("WeekDate")["demand"].sum(min_count=1)
        # Same window naming as the KPI row above (see avg_col in the docstring).
        hist_label = (
            f"{historical_window_label(avg_col)} Historical Demand (avg/wk)"
            if avg_col else "Historical Demand (avg/wk)"
        )
        st.metric(
            hist_label,
            f"{float(sku_weekly_tot.mean()) if not sku_weekly_tot.empty else 0.0:,.1f}",
            help="Mean of this SKU's weekly actual demand for this customer group "
                 "over the same window the KPI row above uses.",
        )
        sysv = row["Current Projection Average"]
        st.metric(
            "Current Forecast (avg/wk)",
            "—" if pd.isna(sysv) else f"{sysv:,.0f}",
        )
        updated = row["Updated Projection Average"]
        st.metric("Updated Forecast (avg/wk)",
                  "—" if pd.isna(updated) else f"{updated:,.0f}")
        pdiff = row["Projection Difference"]
        st.metric(
            "Projection Difference (avg/wk)",
            f"{pdiff:+,.0f}" if pd.notna(pdiff) else "—",
        )
        if RISK_COL in row.index:
            price = row[PRICE_COL] if PRICE_COL in row.index else None
            price = None if price is None or pd.isna(price) else price
            rv = row[RISK_COL]
            st.metric("List Price", fmt_dollar(price, decimals=2))
            st.metric(
                "Revenue Risk (avg/wk)", fmt_dollar(rv, signed=True),
                help="Projection difference × list price for this SKU / group.",
            )
            prv = price * updated if price is not None and pd.notna(updated) else None
            st.metric(
                "Projected Revenue (avg/wk)", fmt_dollar(prv),
                help="List price × this group's updated weekly-avg forecast.",
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


def _render_best_model_combined(df, today_ts, today_str, prices, n_excluded_rows,
                                anchors, P=None, data_sig=None):
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

    # Window label for the historical-demand KPI's help text. It has to describe
    # `anchors`, which come from the SIDEBAR model's week_anchors — NOT from
    # `combined`, which carries both averages, so _render_kpis' resolve_avg_col
    # fallback would pick All-History and mislabel an 8-week window.
    anchors_avg_col = (
        getattr(P, "AVG_COL_LABEL", "8 Week POS/Orders Average")
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
    # range and metrics the standalone "SKU detail" section used to hold. The
    # filter chips and the Excel download still run on the full frame.
    # model_label is left unbound: every row carries its own MODEL_USED_COL, which
    # the card prefers. top_groups is unbound too — this view has no compute_view
    # summary, so no breakdown exists (see render_sku_detail_card).
    sku_card = functools.partial(render_sku_detail_card, agg_by_group,
                                 weekly_by_group, anchors, chart_anchors, pm,
                                 avg_col=anchors_avg_col)
    render_selectable_table(
        table, "filter_best_mix", P,
        condensed_cols=BEST_MIX_CONDENSED_COLS, style=True,
        detail_chart=sku_card, detail_cols=BEST_MIX_CARD_COLS,
    )
    st.download_button(
        "⬇️ Download the combined best-model table",
        data=summary_to_excel(table),
        file_name=f"Combined_best_model_demand_projections_{today_str}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_best_mix",
    )

    _render_excluded("Groups excluded — no backtest-winning model")
