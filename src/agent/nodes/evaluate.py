"""Evaluate node: score every model that ran through one shared backtest.

Every model is scored with the SAME single-holdout backtest against the *raw*
aggregated actuals so the MAEs are apples-to-apples. `autofit_smoothing` is
used only to pick ES's alpha/beta/phi — its internal MAE comes from a 3-fold
rolling-origin backtest on the *cleansed* series and is NOT comparable, so it
is recorded separately as ``autofit_mae`` (audit trail only), never used as
the comparison score.
"""

import inspect

import numpy as np
import pandas as pd

from agent.config import MODEL_OPTIONS
from agent.data_io import view_frame
from agent.logging_util import logger
from agent.model_loader import load_pipeline
from agent.state import AgentState

HOLDOUT_WEEKS = 6  # matches AUTOFIT_HOLDOUT_WEEKS in exponential_smoothing.py


def evaluate_models(state: AgentState) -> dict:
    df = state["cleaned_df"]
    view = state["view"]
    today_ts = state["today_ts"]
    sub = view_frame(df, view)

    results = dict(state["results"])
    errors = list(state.get("errors", []))
    for label, path in MODEL_OPTIONS.items():
        if label not in results:
            continue
        try:
            P = load_pipeline(path)
            fit_kwargs = {}
            if hasattr(P, "autofit_smoothing"):
                # autofit picks ES's parameters; its MAE is recorded for the
                # audit trail but NOT used as the comparison score (it is a
                # multi-fold backtest on the cleansed series — not comparable
                # to the shared single-holdout raw backtest below).
                fit = P.autofit_smoothing(
                    results[label]["agg"], today_ts, holdout_weeks=HOLDOUT_WEEKS
                )
                if fit:
                    results[label]["autofit_mae"] = fit["mae"]
                    results[label]["baseline_mae"] = fit["baseline_mae"]
                    results[label]["params"] = {
                        "alpha": fit["alpha"], "beta": fit["beta"], "phi": fit["phi"]
                    }
                    fit_kwargs = dict(results[label]["params"])
            # every model is scored through the SAME backtest so MAEs compare
            results[label]["mae"] = _generic_backtest(
                P, sub, today_ts, HOLDOUT_WEEKS, fit_kwargs
            )
        except Exception as e:  # one model failing must not sink the others
            results[label]["mae"] = None
            errors.append(f"evaluate {label} failed: {e}")
            logger.warning("Evaluate [%s]: %s backtest failed: %s", view, label, e)
    return {"results": results, "errors": errors}


def _generic_backtest(P, sub, today_ts, holdout_weeks, fit_kwargs=None):
    """Single-holdout backtest MAE for pipeline module ``P`` on view frame ``sub``.

    Aggregate to SKU-week FIRST, then split. Splitting the raw frame and
    merging forecasts back onto it would join one forecast row against many
    CUSTNMBR-level rows per (SKU, WeekDate) and skew the MAE. Scores against
    the *raw* aggregated actuals (POS, falling back to Orders) — the fair,
    planner-relevant target — even though ES/XGBoost cleanse promo spikes
    before fitting.
    """
    cutoff = today_ts - pd.Timedelta(weeks=holdout_weeks)
    agg = P.aggregate_to_sku_week(sub)
    train = agg[agg["WeekDate"] <= cutoff]
    actual = agg[agg["WeekDate"] > cutoff]
    if train.empty or actual.empty:
        return None

    # Only pass fit kwargs the pipeline actually accepts (alpha/beta/phi exist
    # on ES's fit_regression but not on regression/xgboost).
    fit_kwargs = fit_kwargs or {}
    try:
        accepted = inspect.signature(P.fit_regression).parameters
        fit_kwargs = {k: v for k, v in fit_kwargs.items() if k in accepted}
    except (TypeError, ValueError):
        fit_kwargs = {}

    summary, weekly = P.fit_regression(
        train, cutoff, grouping_label="backtest", **fit_kwargs
    )
    if weekly is None or len(weekly) == 0:
        return None

    # weekly_df's WeekDate is a python date and its forecast column is
    # ``projected_pos`` (see models/regression.py fit_regression /
    # models/xgboost.py fit_xgboost weekly_rows) — normalise before merging.
    weekly = weekly[["SKU", "WeekDate", "projected_pos"]].copy()
    weekly["WeekDate"] = pd.to_datetime(weekly["WeekDate"])
    merged = weekly.merge(actual, on=["SKU", "WeekDate"], how="inner")
    if merged.empty:
        return None
    target = merged["POS"].fillna(merged["Orders"])
    merged = merged[target.notna()]
    target = target.dropna()
    if merged.empty:
        return None
    return float(np.abs(merged["projected_pos"] - target).mean())
