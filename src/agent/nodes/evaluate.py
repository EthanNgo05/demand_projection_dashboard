"""Evaluate node: score every model that ran through one shared backtest.

Every model is scored with the SAME walk-forward one-step-ahead backtest
against the *raw* aggregated actuals so the MAEs are apples-to-apples. Because
the app re-runs weekly and only ever uses each forecast's first week, the
backtest steps ``today`` back one week at a time and scores only that first
week against the actual (see ``_generic_backtest``). `autofit_smoothing` is
used only to pick ES's alpha/beta/phi — its internal MAE comes from a rolling
one-step-ahead backtest on the *cleansed* series and is NOT comparable, so it
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
from agent.state import AgentState, report_progress

HOLDOUT_WEEKS = 6  # matches AUTOFIT_HOLDOUT_WEEKS in exponential_smoothing.py


def evaluate_models(state: AgentState, config=None) -> dict:
    df = state["cleaned_df"]
    view = state["view"]
    today_ts = state["today_ts"]
    sub = view_frame(df, view)

    # Serial by design — see the note in forecast.run_all_models. The backtest
    # is the runtime long pole, but it is dominated by one model, so per-model
    # parallelism does not help; views are parallelized across processes in
    # agent/batch.py instead.
    results = dict(state["results"])
    errors = list(state.get("errors", []))
    to_score = [(label, path) for label, path in MODEL_OPTIONS.items() if label in results]
    total = len(to_score)
    for idx, (label, path) in enumerate(to_score, start=1):
        report_progress(config, "backtest", label, idx, total)
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
    """Walk-forward one-step-ahead backtest MAE for pipeline ``P`` on view ``sub``.

    The app re-runs weekly and only ever uses each forecast's FIRST week, so we
    score exactly that: step ``today`` back one week at a time for the last
    ``holdout_weeks`` completed weeks, re-fit at each step, and compare the first
    forecast week against that week's actual. Errors pool across steps and SKUs
    into one MAE.

    Aggregate to SKU-week FIRST (splitting/merging the raw CUSTNMBR frame fans
    one forecast row onto many rows per (SKU, WeekDate) and skews the MAE). Each
    model windows its own training data off the ``today`` argument (trains up to
    the week before ``first_forecast_week``, forecasts starting at it), so
    passing the full history and only varying ``today`` leaks no future actuals.
    Scores against the *raw* aggregated actuals (POS, falling back to Orders) —
    the fair, planner-relevant target — even though ES/XGBoost cleanse promo
    spikes before fitting.
    """
    agg = P.aggregate_to_sku_week(sub)
    if agg.empty:
        return None
    agg = agg.copy()
    agg["WeekDate"] = pd.to_datetime(agg["WeekDate"])

    # Only pass fit kwargs the pipeline actually accepts (alpha/beta/phi exist
    # on ES's fit_regression but not on regression/xgboost).
    fit_kwargs = fit_kwargs or {}
    try:
        accepted = inspect.signature(P.fit_regression).parameters
        fit_kwargs = {k: v for k, v in fit_kwargs.items() if k in accepted}
    except (TypeError, ValueError):
        fit_kwargs = {}

    abs_errors = []
    for step in range(1, int(holdout_weeks) + 1):
        step_today = today_ts - pd.Timedelta(weeks=step)
        summary, weekly = P.fit_regression(
            agg, step_today, grouping_label="backtest", **fit_kwargs
        )
        if weekly is None or len(weekly) == 0:
            continue
        # weekly_df's WeekDate is a python date and its forecast column is
        # ``projected_pos`` — normalise before merging.
        weekly = weekly[["SKU", "WeekDate", "projected_pos"]].copy()
        weekly["WeekDate"] = pd.to_datetime(weekly["WeekDate"])
        step_week = weekly["WeekDate"].min()          # the first forecast week
        fc = weekly[weekly["WeekDate"] == step_week][["SKU", "projected_pos"]]
        merged = fc.merge(agg[agg["WeekDate"] == step_week], on="SKU", how="inner")
        if merged.empty:
            continue
        target = merged["POS"].fillna(merged["Orders"])
        mask = target.notna()
        if not mask.any():
            continue
        abs_errors.extend(
            np.abs(merged.loc[mask, "projected_pos"].to_numpy()
                   - target[mask].to_numpy())
        )
    if not abs_errors:
        return None
    return float(np.mean(abs_errors))
