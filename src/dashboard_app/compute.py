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

    Returns ``(table, weekly_all, agg_all, excluded)`` where ``table`` is a
    DataFrame (SUMMARY_COLUMNS + MODEL_USED_COL) or None when no group resolved /
    produced rows; ``weekly_all`` / ``agg_all`` are the per-group forecast and
    SKU-week aggregate frames stitched together and summed by (SKU, WeekDate) so
    the view can draw the total-demand and per-SKU charts (None alongside a None
    table); and ``excluded`` is the sorted list of group names with no best model.
    Groups are disjoint customer subsets, so summing by (SKU, WeekDate) is a plain
    total — no double counting — and the actuals match the Executive Overview.
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
        return None, None, None, excluded

    # Second pass: forecast each resolved group with its own model (autofit when
    # supported). Alongside each group's summary we keep its weekly forecast and
    # SKU-week aggregate so the charts have series to plot.
    frames = []
    weekly_frames = []
    agg_frames = []
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
        if progress_cb is not None:
            progress_cb(i + 1, n, group)

    if not frames:
        return None, None, None, excluded
    combined = pd.concat(frames, ignore_index=True)
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
    return combined, weekly_all, agg_all, excluded


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
