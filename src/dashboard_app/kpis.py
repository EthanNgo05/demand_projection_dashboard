"""KPI row and the Optimal Projections (best-model-per-group) combined view."""
import pandas as pd
import streamlit as st

from dashboard_app.config import (
    PRICE_COL, RISK_COL, fmt_dollar, MODEL_USED_COL, BEST_MODEL_COMBINED_VIEW,
    ALL_CUSTOMERS_VIEW, model_display,
)
from dashboard_app.summaries import (
    resolve_avg_col, avg_window_phrase, historical_window, _format_generated_at,
)
from dashboard_app.compute import (
    compute_by_customer_best, _agent_summaries_mtime, _agent_summaries_generated_at,
    summary_to_excel,
)
from dashboard_app.charts import chart_range_control, aggregate_chart, sku_chart
from dashboard_app.tables import render_filtered_table


def _render_kpis(summary, agg, anchors, stacked=False):
    """Render the 7-metric KPI row shared by every view.

    Uses only ``summary`` + the SKU-week ``agg`` + the week ``anchors``. SKU
    counts use ``nunique`` (not row count) so the Optimal Projections combined
    view — which carries one row per (SKU, Customer Grouping) — reports distinct
    SKUs; for single-model views SKU is unique per row, so this is unchanged.

    ``stacked`` lays the seven metrics out vertically (one per line) instead of
    across a 7-column row, so they fit a narrow side column like the SKU/Customer
    detail charts. The trailing informational captions are shown only in the wide
    row layout.
    """
    lb, lcw, ffw = anchors
    # Avg. weekly demand = the mean of the TOTAL weekly demand actually plotted
    # on the chart's "Actual demand" line (POS/Orders summed across SKUs per
    # week, then averaged over the weeks in the window). Do NOT sum the per-SKU
    # "N Week POS/Orders Average" column here: that per-SKU average divides each
    # SKU by its own weeks-with-data, so summing it counts a SKU that sold in
    # only a few weeks as if it sold every week and overstates the total.
    n_skus = int(summary["SKU"].nunique())
    avg_col = resolve_avg_col(summary)
    hist_demand = historical_window(agg, summary, (lb, lcw, ffw))
    weekly_totals = hist_demand.groupby("WeekDate")["demand"].sum(min_count=1)
    total_avg = float(weekly_totals.mean()) if not weekly_totals.empty else 0.0
    total_updated = summary["Updated Projection Average"].sum()
    total_initial = summary["Initial Projection Average"].sum()
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
    k1, k2, k3, k4, k5, k6, k7 = [st] * 7 if stacked else st.columns(7)
    k1.metric(
        "SKUs Forecasted", f"{n_skus:,}",
        help=f"{n_orders} forecast from Orders (no POS)" if n_orders else None,
    )
    k2.metric(
        "Historical Demand (avg/wk)", f"{total_avg:,.0f}",
        help=f"Mean of total weekly actual demand (POS/Orders) over the "
             f"{avg_window_phrase(avg_col).lower()} window — the average of the "
             f"chart's actual-demand line.",
    )
    k3.metric(
        "Initial Forecast (avg/wk)", f"{total_initial:,.0f}",
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


def _render_best_model_combined(df, today_ts, today_str, prices, n_excluded_rows,
                                anchors, P=None):
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
            )
            prog.progress(1.0, text="Done")
        finally:
            prog.empty()
        st.session_state["bestmix_result"] = result
        st.session_state["bestmix_generated_at"] = _agent_summaries_generated_at()
        st.session_state["bestmix_structural"] = sig
    else:
        result = st.session_state.get("bestmix_result")

    combined, weekly_all, agg_all, weekly_by_group, agg_by_group, excluded = (
        result if result is not None else (None, None, None, None, None, [])
    )

    generated_at = st.session_state.get("bestmix_generated_at")
    if generated_at:
        st.caption(
            f"Recommendations last generated {_format_generated_at(generated_at)}"
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

    # Chart-only anchors: the passed-in `anchors` come from the sidebar model's
    # week_anchors, whose lookback start (lb) is as short as 8 weeks (8-Week
    # Moving Average). That model choice is irrelevant here, so widen the charts'
    # history floor to the earliest available week — otherwise the date-range
    # picker can only narrow within an ~8-week window. KPIs keep the original
    # `anchors` so their numbers don't shift.
    chart_lb = pd.to_datetime(agg_all["WeekDate"]).min()
    chart_anchors = (chart_lb, lcw, ffw)

    # ----- KPIs -------------------------------------------------------------
    # Same seven metrics as every other view. The combined frame carries one row
    # per (SKU, Customer Grouping); _render_kpis counts distinct SKUs and the
    # forecast/risk totals sum naturally across a SKU's groups.
    _render_kpis(combined, agg_all, anchors)

    # ----- Aggregate chart --------------------------------------------------
    # Total actual demand + total forecast, summed across every group. Actuals
    # match the Executive Overview; only the forecast line differs (each group
    # uses its backtest-winning model).
    agg_ctrl, _ = st.columns([1, 2])
    with agg_ctrl:
        agg_range = chart_range_control(agg_all, weekly_all, lcw, key="range_agg_best")
    st.plotly_chart(
        aggregate_chart(agg_all, combined, weekly_all, chart_anchors, view_label,
                        date_range=agg_range),
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
                customer, date_range=cust_range,
            ),
            width="stretch",
        )
    with ccR:
        # Same seven metrics as the top of the view, scoped to this customer group
        # and stacked to fit the side column (like the SKU detail chart). Use the
        # section's original `anchors` (not the widened chart range) so the
        # historical-demand window lines up with the combined KPI row.
        _render_kpis(summary_c, agg_c, anchors, stacked=True)

    # ----- Per-SKU detail ---------------------------------------------------
    st.markdown("### SKU detail")
    c_sku, c_cust = st.columns(2)
    with c_sku:
        skus = sorted(combined["SKU"].astype(str).unique())
        sku = st.selectbox("SKU", skus, help="Type to search", key="best_sku")
    sku_rows_all = combined[combined["SKU"].astype(str) == sku]
    # Customer-group filter: default keeps the combined total across every group
    # carrying this SKU (the original behaviour); pick a group to narrow the chart
    # and metrics to just that group. Options depend on the selected SKU.
    sku_groups = sorted(sku_rows_all["Customer Grouping"].astype(str).unique())
    with c_cust:
        cust_pick = st.selectbox(
            "Customer group", [ALL_CUSTOMERS_VIEW] + sku_groups,
            key="best_sku_cust",
            help="Keep the combined total, or narrow to one customer group.",
        )
    if cust_pick == ALL_CUSTOMERS_VIEW:
        rows = sku_rows_all
        sku_agg, sku_weekly = agg_all, weekly_all
    else:
        rows = sku_rows_all[sku_rows_all["Customer Grouping"].astype(str) == cust_pick]
        sku_agg = agg_by_group[agg_by_group["Customer Grouping"].astype(str) == cust_pick]
        sku_weekly = weekly_by_group[
            weekly_by_group["Customer Grouping"].astype(str) == cust_pick
        ]

    row0 = rows.iloc[0]
    desc = row0["Description"] if isinstance(row0["Description"], str) else ""
    # Resolve one source for the SKU's chart (same last-wins rule as source_map).
    src_vals = rows["Data Source"].dropna().unique().tolist() \
        if "Data Source" in rows.columns else []
    source = src_vals[-1] if src_vals else "POS"
    mixed_source = len(src_vals) > 1

    cL, cR = st.columns([3, 1])
    with cL:
        sku_range = chart_range_control(sku_agg, sku_weekly, lcw, key="range_sku_best")
        st.plotly_chart(
            sku_chart(sku, desc, source, sku_agg, sku_weekly, chart_anchors,
                      date_range=sku_range),
            width="stretch",
        )
        # One line per group, annotated with the best model used for that group
        # (rows carries one row per Customer Grouping, each with MODEL_USED_COL).
        group_models = sorted(
            zip(rows["Customer Grouping"].astype(str),
                rows[MODEL_USED_COL].astype(str)),
            key=lambda gm: gm[0],
        )
        group_lines = "\n".join(f"- {g} — {m}" for g, m in group_models)
        if cust_pick == ALL_CUSTOMERS_VIEW:
            st.caption(
                f"Totals across every group carrying this SKU "
                f"({len(group_models)} group{'s' if len(group_models) != 1 else ''}), "
                f"with the model used for each:\n{group_lines}"
            )
        else:
            st.caption(f"Customer group **{cust_pick}** only:\n{group_lines}")
    with cR:
        # Metrics aggregate the SKU's per-group rows: forecast/risk totals sum;
        # the historical avg/wk is derived from the stitched actuals (do NOT sum
        # the per-SKU average column across groups — it would over-count). When a
        # single group is picked, `rows` is one row so every sum collapses to it.
        st.metric("Data Source", f"{source} (mixed)" if mixed_source else source)
        sku_hist = historical_window(
            sku_agg[sku_agg["SKU"].astype(str) == sku], rows, anchors
        )
        sku_weekly_tot = sku_hist.groupby("WeekDate")["demand"].sum(min_count=1)
        st.metric(
            "Historical Demand (avg/wk)",
            f"{float(sku_weekly_tot.mean()) if not sku_weekly_tot.empty else 0.0:,.1f}",
        )
        sysv = rows["Initial Projection Average"].sum(min_count=1)
        st.metric(
            "Initial Forecast (avg/wk)",
            "—" if pd.isna(sysv) else f"{sysv:,.0f}",
        )
        updated = rows["Updated Projection Average"].sum()
        st.metric("Updated Forecast (avg/wk)", f"{updated:,.0f}")
        pdiff = rows["Projection Difference"].sum(min_count=1)
        st.metric(
            "Projection Difference (avg/wk)",
            f"{pdiff:+,.0f}" if pd.notna(pdiff) else "—",
        )
        if RISK_COL in combined.columns:
            price = rows[PRICE_COL].dropna().iloc[0] \
                if PRICE_COL in rows.columns and rows[PRICE_COL].notna().any() \
                else None
            rv = rows[RISK_COL].sum(min_count=1)
            st.metric("List Price", fmt_dollar(price, decimals=2))
            st.metric(
                "Revenue Risk (avg/wk)", fmt_dollar(rv, signed=True),
                help="Σ (projection difference × list price) across this SKU's groups.",
            )
            prv = price * updated if price is not None else None
            st.metric(
                "Projected Revenue (avg/wk)", fmt_dollar(prv),
                help="List price × total updated weekly-avg forecast for this SKU.",
            )

    st.markdown("### Summary table by SKU and customer")

    # Keep each SKU's rows together; largest revenue risk first when present.
    if RISK_COL in combined.columns and combined[RISK_COL].notna().any():
        table = (
            combined.assign(_abs=combined[RISK_COL].abs())
            .sort_values(["SKU", "_abs"], ascending=[True, False], na_position="last")
            .drop(columns="_abs").reset_index(drop=True)
        )
        st.caption("Each SKU broken out by customer group; within a SKU, "
                   "largest revenue risk first (by magnitude).")
    else:
        table = combined.sort_values(["SKU", "Customer Grouping"]).reset_index(drop=True)
        st.caption("Each SKU broken out by customer group.")

    # Display copy: mark the All-History average with a '*' where the value is
    # really the 8-Week Moving Average model's 8-week run-rate (that model has no
    # all-history average). The numeric `table` is kept for the Excel download;
    # its "Model Used" column already identifies those groups there.
    display_table = table
    avg_col = "All-History POS/Orders Average"
    eight_wk_label = model_display("8-Week Moving Average")
    if avg_col in table.columns and MODEL_USED_COL in table.columns:
        is_8wk = table[MODEL_USED_COL] == eight_wk_label

        def _fmt_avg(v, star):
            if pd.isna(v):
                return ""
            return f"{v:,.1f}*" if star else f"{v:,.1f}"

        display_table = table.copy()
        display_table[avg_col] = [
            _fmt_avg(v, s) for v, s in zip(table[avg_col], is_8wk)
        ]

    render_filtered_table(display_table, "filter_best_mix", P, style=True)
    if avg_col in table.columns and MODEL_USED_COL in table.columns \
            and (table[MODEL_USED_COL] == eight_wk_label).any():
        st.caption(
            f"\\* This group is forecast by the {eight_wk_label} model, so its "
            f"{avg_col} is that model's recent 8-week run-rate rather than an "
            "all-history average."
        )
    st.download_button(
        "⬇️ Download the combined best-model table",
        data=summary_to_excel(table),
        file_name=f"Combined_best_model_demand_projections_{today_str}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_best_mix",
    )

    _render_excluded("Groups excluded — no backtest-winning model")
