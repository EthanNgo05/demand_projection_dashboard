"""Shared state schema for the agent graph.

Every node reads from / writes partial updates to AgentState. `total=False`
lets nodes return only the keys they changed — LangGraph merges dict updates
into the running state.
"""

from typing import Optional, TypedDict

import pandas as pd


def report_progress(config, phase, model, done, total):
    """Best-effort progress ping to a UI callback passed via RunnableConfig.

    The dashboard runs the graph on a background thread and passes a callback as
    ``config={"configurable": {"progress_cb": fn}}`` so it can show which model
    is being fit/backtested (e.g. "Fitting XGBoost (3/4)"). The CLI and batch
    runner pass no config, so ``cb`` is None and this is a no-op. Any callback
    error is swallowed — progress reporting must never break the pipeline.
    """
    try:
        cb = (config or {}).get("configurable", {}).get("progress_cb")
        if cb:
            cb(phase, model, done, total)
    except Exception:
        pass


class ModelResult(TypedDict, total=False):
    """Output bundle for one forecasting model run (keyed by MODEL_OPTIONS label)."""

    summary_df: pd.DataFrame
    weekly_df: pd.DataFrame
    agg: pd.DataFrame
    mase: float  # comparison score from the shared _generic_backtest (pooled MASE vs an 8-week-average baseline)
    autofit_mae: float  # ES autofit internal MAE (audit only, NOT comparable)
    baseline_mae: float
    params: dict  # alpha/beta/phi if applicable


class AgentState(TypedDict, total=False):
    view: str  # config.ALL_CUSTOMERS_VIEW, "All Customers - <region>", or a Customer Grouping
    today_ts: pd.Timestamp
    raw_path: str
    price_path: Optional[str]
    cleaned_df: Optional[pd.DataFrame]
    prices: Optional[pd.Series]
    results: dict[str, ModelResult]  # keyed by MODEL_OPTIONS label
    best_model: Optional[str]
    mase_confidence_threshold: Optional[float]  # per-run override; select falls back to config
    confidence_flag: bool
    anomalies: list[str]
    narrative: Optional[str]
    # Active SKUs with demand history that fall outside the winning model's
    # history window (e.g. the 8-week moving average drops a SKU whose only
    # sales predate its window). Empty when the winner uses all history. Lets
    # the output name SKUs the model silently omits rather than dropping them
    # without a trace. Computed in the publish node.
    window_excluded_skus: list[dict]
    errors: list[str]
