"""Forecasting compute core: view enumeration, per-view/per-group forecasts."""
import os
import glob
import json
import inspect
from io import BytesIO

import pandas as pd
import streamlit as st

from dashboard_app.config import (
    ALL_CUSTOMERS_VIEW, region_from_view, MODEL_OPTIONS, MODEL_USED_COL,
    model_display, REPO_ROOT,
)
from dashboard_app.pipeline import (
    load_pipeline, pipeline_path,
    _supports_prices, _supports_smoothing, _supports_min_weeks, _supports_autofit,
)


def _region_frame(df, P, region):
    """Rows of ``df`` whose customer group belongs to ``region``.

    str() on region_for_group: a custom pipeline may return non-string labels
    (see the key=str note in the sidebar), and the view string the region was
    parsed from was built from the str form.
    """
    groups = df["Customer Grouping"].map(lambda g: str(P.region_for_group(g)))
    return df[groups == region]


def list_views(df):
    """Group views organised by region, plus the combined ALL CUSTOMERS view."""
    P = load_pipeline(pipeline_path())
    groups = sorted(df["Customer Grouping"].dropna().unique().tolist())
    by_region = {}
    for g in groups:
        by_region.setdefault(P.region_for_group(g), []).append(g)
    return by_region


@st.cache_data(show_spinner="Building forecast…")
def compute_view(df, view, today_ts, model_path, prices=None, alpha=None,
                 beta=None, phi=None, min_weeks=None):
    """Recompute summary + weekly + per-week aggregate for the selected view.

    Returns (summary_df, weekly_df, agg_frame) where agg_frame is the SKU-week
    POS/Orders/Projection table (used to draw historical actuals and the original
    projection). For ALL CUSTOMERS the breakdown is included so the summary
    carries 'Top Volume Customer Groups'. When ``prices`` (a SKU -> price Series)
    is supplied and the pipeline supports it, the summary also carries
    'List Price (USD)' and 'Revenue Risk (avg/wk)'. ``alpha`` / ``beta`` / ``phi``,
    when given, override the pipeline's smoothing constants for this call, and
    ``min_weeks`` overrides MIN_WEEKS_FOR_TREND (all are part of the cache key, so
    moving a slider recomputes the forecast). ``model_path`` selects the
    pipeline and keys the cache, so toggling the model recomputes too.
    """
    P = load_pipeline(model_path)
    kwargs = {}
    if prices is not None and _supports_prices(P):
        kwargs["list_prices"] = prices
    if None not in (alpha, beta, phi) and _supports_smoothing(P):
        kwargs.update(alpha=alpha, beta=beta, phi=phi)
    if min_weeks is not None and _supports_min_weeks(P):
        kwargs["min_weeks_for_trend"] = min_weeks
    if view == ALL_CUSTOMERS_VIEW:
        combined_label = getattr(
            P, "ALL_CUSTOMERS_LABEL", getattr(P, "ALL_SKUS_LABEL", ALL_CUSTOMERS_VIEW)
        )
        agg = P.aggregate_to_sku_week(df)
        summary, weekly = P.fit_regression(
            agg, today_ts, grouping_label=combined_label,
            breakdown_df=df, **kwargs,
        )
    elif (region_all := region_from_view(view)) is not None:
        # Per-region rollup: every customer group in the region, combined.
        # breakdown_df mirrors the ALL CUSTOMERS branch so the summary carries
        # 'Top Volume Customer Groups' (here: the region's groups).
        sub = _region_frame(df, P, region_all)
        agg = P.aggregate_to_sku_week(sub)
        summary, weekly = P.fit_regression(
            agg, today_ts, grouping_label=view, breakdown_df=sub, **kwargs
        )
    else:
        sub = df[df["Customer Grouping"] == view]
        agg = P.aggregate_to_sku_week(sub)
        summary, weekly = P.fit_regression(
            agg, today_ts, grouping_label=view, **kwargs
        )
    return summary, weekly, agg


def view_to_excel(summary_df, weekly_df):
    """Build an in-memory .xlsx (same two-sheet layout as the pipeline output)."""
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        summary_df.to_excel(w, sheet_name="summary", index=False)
        weekly_df.to_excel(w, sheet_name="weekly_forecast", index=False)
    buf.seek(0)
    return buf.getvalue()


def summary_to_excel(summary_df, sheet_name="summary"):
    """Build an in-memory single-sheet .xlsx of a summary table.

    Used for the by-SKU-and-customer table, which mirrors the pipeline's
    ALL_CUSTOMERS_demand_projections file (a single concatenated summary sheet).
    """
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        summary_df.to_excel(w, sheet_name=sheet_name, index=False)
    buf.seek(0)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def run_autofit(df, view, today_ts, model_path, min_weeks=None):
    """Grid-search the best alpha/beta/phi for the selected view (cached).

    Builds the same SKU-week aggregate ``compute_view`` fits on, then delegates
    to the pipeline's ``autofit_smoothing`` backtest. Cached on
    (data, view, snapshot, model, min_weeks) so clicking Autofit twice — or
    returning to a view already fitted this session — is instant.
    """
    P = load_pipeline(model_path)
    if not _supports_autofit(P):
        return None
    if view == ALL_CUSTOMERS_VIEW:
        agg = P.aggregate_to_sku_week(df)
    elif (region_all := region_from_view(view)) is not None:
        agg = P.aggregate_to_sku_week(_region_frame(df, P, region_all))
    else:
        agg = P.aggregate_to_sku_week(df[df["Customer Grouping"] == view])
    kwargs = {}
    if min_weeks is not None and "min_weeks_for_trend" in inspect.signature(
        P.autofit_smoothing
    ).parameters:
        kwargs["min_weeks_for_trend"] = min_weeks
    return P.autofit_smoothing(agg, today_ts, **kwargs)


@st.cache_data(show_spinner=False)
def _forecast_one_group(df_group, today_ts, model_path, group_label,
                        prices=None, alpha=None, beta=None, phi=None,
                        min_weeks=None):
    """Forecast a single customer group's SKUs. Cached; calls NO Streamlit
    element, so it is safe to replay on a cache hit. ``group_label`` is a
    normal (hashable) argument so distinct groups get distinct cache entries.

    Returns ``(summary, weekly, agg)`` — the same three frames ``compute_view``
    produces for a single view, so callers that stitch groups together (the
    Optimal Projections combined view) can build charts, not just the summary.
    """
    P = load_pipeline(model_path)
    kwargs = {}
    if prices is not None and _supports_prices(P):
        kwargs["list_prices"] = prices
    if None not in (alpha, beta, phi) and _supports_smoothing(P):
        kwargs.update(alpha=alpha, beta=beta, phi=phi)
    if min_weeks is not None and _supports_min_weeks(P):
        kwargs["min_weeks_for_trend"] = min_weeks
    agg = P.aggregate_to_sku_week(df_group)
    summary, weekly = P.fit_regression(
        agg, today_ts, grouping_label=group_label, **kwargs
    )
    return summary, weekly, agg


def compute_by_customer(df, today_ts, model_path, prices=None, alpha=None,
                        beta=None, phi=None, min_weeks=None, progress_cb=None):
    """Per-(SKU, Customer Grouping) summary — the rows behind ALL_CUSTOMERS.

    The pipeline's ``ALL_CUSTOMERS_demand_projections`` file is just a
    concatenation of every per-customer-group summary sheet. This reproduces it
    live: for each Customer Grouping we run the identical per-group forecast via
    the cached ``_forecast_one_group`` helper, then stack the summaries.
    Recomputing rather than reading the saved workbook keeps this table on the
    same snapshot / prices / smoothing as the rest of the page.

    This orchestrator is intentionally NOT cached: it may call ``progress_cb``
    (which drives a progress bar), and Streamlit element calls are not allowed
    inside a cached function. Each group's forecast is cached instead, so the
    expensive work is still memoised. On plain reruns this function isn't called
    at all — the result is held in session_state (see main()).

    Returns a DataFrame in the pipeline's SUMMARY_COLUMNS order, or None if no
    group had anything to forecast.
    """
    frames = []
    groups = sorted(df["Customer Grouping"].dropna().unique().tolist())
    n_groups = len(groups)
    for i, group in enumerate(groups):
        sub = df[df["Customer Grouping"] == group]
        summary, _, _ = _forecast_one_group(
            sub, today_ts, model_path, group,
            prices, alpha, beta, phi, min_weeks,
        )
        if summary is not None and not summary.empty:
            frames.append(summary)
        if progress_cb is not None:
            progress_cb(i + 1, n_groups, group)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _agent_summaries_mtime():
    """Newest mtime among outputs/agent_summary_*.json, or 0.0 if none exist.

    Folded into the combined view's cache signature so the table rebuilds
    automatically as soon as a batch (the "Agent Summary (all views)" button, the
    nightly job, or `agent.batch`) writes fresh summaries — no manual reload."""
    paths = glob.glob(os.path.join(REPO_ROOT, "outputs", "agent_summary_*.json"))
    return max((os.path.getmtime(p) for p in paths), default=0.0)


def _agent_summaries_generated_at():
    """Latest ``generated_at`` stamped across outputs/agent_summary_*.json.

    Reflects when the batch last produced the per-group recommendations that the
    Optimal Projections (Combined) view is stitched from. Returns the ISO string,
    or None if no summary carries a parseable timestamp. The stamps share one
    format (``YYYY-MM-DDTHH:MM:SS``), so a lexical max is also the newest."""
    latest = None
    for p in glob.glob(os.path.join(REPO_ROOT, "outputs", "agent_summary_*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                gen = json.load(f).get("generated_at")
        except (OSError, ValueError):
            continue
        if gen and (latest is None or str(gen) > latest):
            latest = str(gen)
    return latest


def _best_model_for_group(group):
    """(label, model_path) for a group's backtest-winning model, or None.

    Reads the group's published agent summary (agent_summary_<group>.json) and
    maps its ``best_model`` label to a MODEL_OPTIONS file path. Returns None when
    the summary is missing, has no best model, or names a label this deployment
    doesn't offer — the caller treats all three as "no summary yet".
    """
    payload = _load_agent_summary(group)
    if not payload:
        return None
    label = payload.get("best_model")
    path = MODEL_OPTIONS.get(label)
    if not label or path is None:
        return None
    return label, path


ALL_HIST_AVG_COL = "All-History POS/Orders Average"
EIGHT_WK_AVG_COL = "8-Week POS/Orders Average"


def _descriptive_averages(agg_by_group, today_ts):
    """Per-(Customer Grouping, SKU) all-history and 8-week demand averages.

    Computed straight from the stitched per-group SKU-week aggregates so BOTH
    averages exist for every group regardless of which model won its backtest.
    Definitions mirror the model files so the numbers agree:

    * source per SKU = POS if the SKU has ANY POS in the window, else Orders
      (SKUs with neither are skipped) -- the POS-then-Orders fallback the models use;
    * average = total demand / weeks-in-span, where the span runs from the SKU's
      first observation in the window through the last completed week (post-launch
      gaps count as real zeros). Over the 8-week window this reproduces
      ``regression.fit_regression``'s ``mean_val = y.sum() / weeks_since_first``.

    Discontinued SKUs (name ending in '*') are dropped, matching the models.
    Returns a frame: Customer Grouping, SKU, ALL_HIST_AVG_COL, EIGHT_WK_AVG_COL.
    """
    # In-progress week is excluded; the historical window ends last completed week.
    days_since_sunday = (today_ts.weekday() + 1) % 7          # Sun=0 ... Sat=6
    current_week_start = today_ts - pd.Timedelta(days=days_since_sunday)
    last_complete_week = current_week_start - pd.Timedelta(weeks=1)
    eight_wk_start = last_complete_week - pd.Timedelta(weeks=7)   # 8 weeks inclusive
    hist_start = today_ts - pd.DateOffset(years=3)   # matches HISTORY_YEARS (all history)

    A = agg_by_group.copy()
    A["SKU"] = A["SKU"].astype(str)
    A = A[~A["SKU"].str.endswith("*")]
    A["WeekDate"] = pd.to_datetime(A["WeekDate"])

    def _avg(start, out_col):
        win = A[(A["WeekDate"] >= start) & (A["WeekDate"] <= last_complete_week)]
        rows = []
        for (grp, sku), g in win.groupby(["Customer Grouping", "SKU"], sort=False):
            pos = g[g["POS"].notna()]
            if not pos.empty:
                vals, weeks = pos["POS"], pos["WeekDate"]
            else:
                orders = g[g["Orders"].notna()]
                if orders.empty:
                    continue  # no POS and no Orders -> nothing to average
                vals, weeks = orders["Orders"], orders["WeekDate"]
            weeks_span = int(round((last_complete_week - weeks.min()).days / 7)) + 1
            rows.append({
                "Customer Grouping": grp, "SKU": sku,
                out_col: round(vals.sum() / max(weeks_span, 1), 1),
            })
        return pd.DataFrame(rows, columns=["Customer Grouping", "SKU", out_col])

    all_hist = _avg(hist_start, ALL_HIST_AVG_COL)
    eight_wk = _avg(eight_wk_start, EIGHT_WK_AVG_COL)
    return all_hist.merge(eight_wk, on=["Customer Grouping", "SKU"], how="outer")


def compute_by_customer_best(df, today_ts, prices=None, min_weeks=None,
                             progress_cb=None):
    """Per-(SKU, Customer Grouping) summary using each group's BEST model.

    Like ``compute_by_customer``, but instead of one model for every group it
    forecasts each group with the model that won that group's backtest (from
    ``agent_summary_<group>.json``) and stamps a ``MODEL_USED_COL`` column. To
    match what the single-group view shows, groups whose best model supports
    autofit are tuned per group via ``run_autofit`` before forecasting.

    A group is only included if it has a resolvable best model. Groups with no
    published summary, or whose summary has no backtest winner (``best_model`` is
    null — history too short to score any model), are left OUT of the table and
    returned separately so the caller can list them.

    Returns ``(table, weekly_all, agg_all, weekly_by_group, agg_by_group,
    excluded)`` where ``table`` is a DataFrame (SUMMARY_COLUMNS + MODEL_USED_COL)
    or None when no group resolved / produced rows; ``weekly_all`` / ``agg_all``
    are the per-group forecast and SKU-week aggregate frames stitched together and
    summed by (SKU, WeekDate) so the view can draw the total-demand and per-SKU
    charts; ``weekly_by_group`` / ``agg_by_group`` are the SAME per-group frames
    stitched together but NOT summed — each row keeps its ``Customer Grouping`` so
    the view can draw one customer group's total on demand (all four frames are
    None alongside a None table); and ``excluded`` is the sorted list of group
    names with no best model. Groups are disjoint customer subsets, so summing by
    (SKU, WeekDate) is a plain total — no double counting — and the actuals match
    the Executive Overview.
    """
    groups = sorted(df["Customer Grouping"].dropna().unique().tolist())

    # First pass: split into groups with a resolvable best model vs. those without
    # (no summary file, or a summary whose best_model is null).
    resolved = {}
    excluded = []
    for group in groups:
        best = _best_model_for_group(group)
        if best is None:
            excluded.append(group)
        else:
            resolved[group] = best
    if not resolved:
        return None, None, None, None, None, excluded

    # Second pass: forecast each resolved group with its own model (autofit when
    # supported). Alongside each group's summary we keep its weekly forecast and
    # SKU-week aggregate so the charts have series to plot.
    frames = []
    weekly_frames = []
    agg_frames = []
    # Same per-group frames, tagged with the group and NOT summed away — feed the
    # per-customer "Customer detail" chart.
    weekly_by_group_frames = []
    agg_by_group_frames = []
    n = len(resolved)
    for i, (group, (label, path)) in enumerate(resolved.items()):
        sub = df[df["Customer Grouping"] == group]
        alpha = beta = phi = None
        P = load_pipeline(path)
        if _supports_autofit(P):
            fitted = run_autofit(df, group, today_ts, path, min_weeks)
            if fitted:
                alpha, beta, phi = fitted.get("alpha"), fitted.get("beta"), fitted.get("phi")
        summary, weekly, agg = _forecast_one_group(
            sub, today_ts, path, group, prices, alpha, beta, phi, min_weeks,
        )
        if summary is not None and not summary.empty:
            summary = summary.copy()
            summary[MODEL_USED_COL] = model_display(label)
            frames.append(summary)
            # Keep only the columns the charts need; models carry extra columns
            # (e.g. promo_uplift) that differ across models and would break concat.
            wk = weekly[["SKU", "WeekDate", "projected_pos"]].copy()
            wk["WeekDate"] = pd.to_datetime(wk["WeekDate"])
            weekly_frames.append(wk)
            ag = agg[["SKU", "WeekDate", "POS", "Orders", "Projection"]].copy()
            ag["WeekDate"] = pd.to_datetime(ag["WeekDate"])
            agg_frames.append(ag)
            # Tagged copies for the per-customer chart (kept separate so the
            # summed weekly_all/agg_all above are unaffected).
            weekly_by_group_frames.append(wk.assign(**{"Customer Grouping": group}))
            agg_by_group_frames.append(ag.assign(**{"Customer Grouping": group}))
        if progress_cb is not None:
            progress_cb(i + 1, n, group)

    if not frames:
        return None, None, None, None, None, excluded
    combined = pd.concat(frames, ignore_index=True)

    # Give every group BOTH descriptive averages regardless of its winning model.
    # Each model reports only one: the 8-Week Moving Average model reports an
    # "8 Week POS/Orders Average" (a recent run-rate), the others report
    # all-history. Compute both centrally from the stitched per-group aggregates
    # so the two columns are always populated and comparable.
    avgs = _descriptive_averages(
        pd.concat(agg_by_group_frames, ignore_index=True), today_ts
    )
    combined = combined.drop(columns=["8 Week POS/Orders Average"], errors="ignore")
    combined = combined.merge(
        avgs.rename(columns={
            ALL_HIST_AVG_COL: "_central_all_hist",
            EIGHT_WK_AVG_COL: "_central_8wk",
        }),
        on=["SKU", "Customer Grouping"], how="left",
    )
    # All-History: keep each model's own reported value; fill only the gaps (the
    # 8-week-model groups, which never compute an all-history average) so existing
    # non-8-week numbers are unchanged.
    if ALL_HIST_AVG_COL in combined.columns:
        combined[ALL_HIST_AVG_COL] = combined[ALL_HIST_AVG_COL].fillna(
            combined["_central_all_hist"]
        )
    else:
        combined[ALL_HIST_AVG_COL] = combined["_central_all_hist"]
    # 8-Week: the central value for every group (it equals the 8-Week Moving
    # Average model's own figure on the groups it won, so nothing shifts there).
    # A SKU with history but no POS/Orders in the last 8 weeks has no run-rate to
    # compute; its recent average is a genuine 0 (absent week = zero, matching the
    # models' gap-fill), so fill rather than leave a blank.
    combined[EIGHT_WK_AVG_COL] = combined["_central_8wk"].fillna(0.0)
    combined = combined.drop(columns=["_central_all_hist", "_central_8wk"])

    # Slot both averages right after "Weeks with data" (All-History then 8-Week),
    # immediately ahead of "Updated Projection Average", for a stable layout.
    if "Weeks with data" in combined.columns:
        cols = [c for c in combined.columns
                if c not in (ALL_HIST_AVG_COL, EIGHT_WK_AVG_COL)]
        pos = cols.index("Weeks with data") + 1
        cols[pos:pos] = [ALL_HIST_AVG_COL, EIGHT_WK_AVG_COL]
        combined = combined[cols]

    # Surface the model used right after the customer group for readability.
    if "Customer Grouping" in combined.columns:
        cols = [c for c in combined.columns if c != MODEL_USED_COL]
        pos = cols.index("Customer Grouping") + 1
        cols.insert(pos, MODEL_USED_COL)
        combined = combined[cols]

    # Stitch the per-group series into one total per (SKU, WeekDate). min_count=1
    # keeps a genuinely-absent cell NaN rather than coercing it to 0.
    weekly_all = (
        pd.concat(weekly_frames, ignore_index=True)
        .groupby(["SKU", "WeekDate"], as_index=False)["projected_pos"].sum()
    )
    agg_all = (
        pd.concat(agg_frames, ignore_index=True)
        .groupby(["SKU", "WeekDate"], as_index=False)[["POS", "Orders", "Projection"]]
        .sum(min_count=1)
    )
    weekly_by_group = pd.concat(weekly_by_group_frames, ignore_index=True)
    agg_by_group = pd.concat(agg_by_group_frames, ignore_index=True)
    return combined, weekly_all, agg_all, weekly_by_group, agg_by_group, excluded


def _agent_summary_path(view):
    """Path publish.py writes for a given view (same view->filename mangling)."""
    safe_view = view.replace(" ", "_").replace("/", "-")
    return os.path.join(REPO_ROOT, "outputs", f"agent_summary_{safe_view}.json")


def _load_agent_summary(view):
    """Last agent run for this view, or None if it hasn't run / is unreadable."""
    path = _agent_summary_path(view)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
