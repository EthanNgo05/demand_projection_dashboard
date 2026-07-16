"""Forecast node: run every configured model against the selected view.

No winner is picked here — Phase 3's evaluate/select nodes do that. Each
model's summary/weekly/agg is stashed in ``state["results"]`` keyed by its
MODEL_OPTIONS label.
"""

import inspect

from agent.config import ALL_CUSTOMERS_VIEW, MODEL_OPTIONS, region_from_view
from agent.data_io import view_frame
from agent.model_loader import load_pipeline
from agent.state import AgentState


def _params(fn):
    """Parameter names of ``fn`` — same trick as dashboard._supports_prices."""
    try:
        return inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}


def run_all_models(state: AgentState) -> dict:
    df = state["cleaned_df"]
    view = state["view"]
    today_ts = state["today_ts"]
    is_all = view == ALL_CUSTOMERS_VIEW
    is_region_all = region_from_view(view) is not None
    sub = view_frame(df, view)

    # Models are run serially here on purpose: a single model (Holt-Winters via
    # statsmodels) dominates a view's runtime, so fanning the four models out
    # across threads/processes measured ~1x (threads even regress on the GIL —
    # much of the per-SKU model code is pure-Python pandas). Parallelism lives
    # ACROSS views instead (agent/batch.py runs a process per view).
    results = {}
    errors = list(state.get("errors", []))
    for label, path in MODEL_OPTIONS.items():
        try:
            P = load_pipeline(path)
            agg = P.aggregate_to_sku_week(sub)
            kwargs = (
                {"list_prices": state["prices"]}
                if state.get("prices") is not None
                and "list_prices" in _params(P.fit_regression)
                else {}
            )
            # Mirror dashboard.compute_view exactly: the ALL view uses the
            # pipeline's own combined label; both the ALL view and a region
            # rollup pass breakdown_df so the summary carries 'Top Volume
            # Customer Groups'. Skipping either makes the Phase 2 parity test
            # fail on the combined views.
            if is_all:
                label_for_fit = getattr(
                    P, "ALL_CUSTOMERS_LABEL", getattr(P, "ALL_SKUS_LABEL", view)
                )
                kwargs["breakdown_df"] = sub
            else:
                label_for_fit = view
                if is_region_all:
                    kwargs["breakdown_df"] = sub
            summary, weekly = P.fit_regression(
                agg, today_ts, grouping_label=label_for_fit, **kwargs
            )
            results[label] = {"summary_df": summary, "weekly_df": weekly, "agg": agg}
        except Exception as e:  # one model failing must not sink the others
            errors.append(f"{label} failed: {e}")
    return {"results": results, "errors": errors}
