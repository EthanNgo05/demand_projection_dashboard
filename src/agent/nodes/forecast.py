"""Forecast node: run every configured model against the selected view.

No winner is picked here — Phase 3's evaluate/select nodes do that. Each
model's summary/weekly/agg is stashed in ``state["results"]`` keyed by its
MODEL_OPTIONS label.
"""

import inspect

from agent.config import ALL_CUSTOMERS_VIEW, MODEL_OPTIONS
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
    sub = df if is_all else df[df["Customer Grouping"] == view]

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
            # pipeline's own combined label and passes breakdown_df so the
            # summary carries 'Top Volume Customer Groups'. Skipping either
            # makes the Phase 2 parity test fail on the combined view.
            if is_all:
                label_for_fit = getattr(
                    P, "ALL_CUSTOMERS_LABEL", getattr(P, "ALL_SKUS_LABEL", view)
                )
                kwargs["breakdown_df"] = sub
            else:
                label_for_fit = view
            summary, weekly = P.fit_regression(
                agg, today_ts, grouping_label=label_for_fit, **kwargs
            )
            results[label] = {"summary_df": summary, "weekly_df": weekly, "agg": agg}
        except Exception as e:  # one model failing must not sink the others
            errors.append(f"{label} failed: {e}")
    return {"results": results, "errors": errors}
