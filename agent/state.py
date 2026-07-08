"""Shared state schema for the agent graph.

Every node reads from / writes partial updates to AgentState. `total=False`
lets nodes return only the keys they changed — LangGraph merges dict updates
into the running state.
"""

from typing import Optional, TypedDict

import pandas as pd


class ModelResult(TypedDict, total=False):
    """Output bundle for one forecasting model run (keyed by MODEL_OPTIONS label)."""

    summary_df: pd.DataFrame
    weekly_df: pd.DataFrame
    agg: pd.DataFrame
    mae: float  # comparison score from the shared _generic_backtest
    autofit_mae: float  # ES autofit internal MAE (audit only, NOT comparable)
    baseline_mae: float
    params: dict  # alpha/beta/phi if applicable


class AgentState(TypedDict, total=False):
    view: str  # config.ALL_CUSTOMERS_VIEW or a Customer Grouping
    today_ts: pd.Timestamp
    raw_path: str
    price_path: Optional[str]
    cleaned_df: Optional[pd.DataFrame]
    prices: Optional[pd.Series]
    results: dict[str, ModelResult]  # keyed by MODEL_OPTIONS label
    best_model: Optional[str]
    mae_confidence_threshold: Optional[float]  # per-run override; select falls back to config
    confidence_flag: bool
    anomalies: list[str]
    narrative: Optional[str]
    errors: list[str]
