"""Evaluate node: score every model that ran through one shared backtest.

Every model is scored with the SAME walk-forward one-step-ahead backtest
against the *raw* aggregated actuals, then scaled by a common baseline so the
scores are apples-to-apples ACROSS VIEWS as well as across models: the score
is a pooled MASE — the model's pooled MAE divided by the pooled MAE of a
plain 8-week moving average of each SKU's actuals over the same points.
MASE < 1 beats the 8-week average; a flat-zero forecast on a normal-volume
view scores >> 1 instead of a deceptively "small" raw MAE (which is what let
near-zero forecasts win on intermittent series). Because the app re-runs
weekly and only ever uses each forecast's first week, the backtest steps
``today`` back one week at a time and scores only that first week against the
actual (see ``_generic_backtest``). `autofit_smoothing` is used only to pick
ES's alpha/beta/phi — its internal MAE comes from a rolling one-step-ahead
backtest on the *cleansed* series and is NOT comparable, so it is recorded
separately as ``autofit_mae`` (audit trail only), never used as the
comparison score.
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
            # every model is scored through the SAME backtest so MASEs compare
            results[label]["mase"] = _generic_backtest(
                P, sub, today_ts, HOLDOUT_WEEKS, fit_kwargs
            )
        except Exception as e:  # one model failing must not sink the others
            results[label]["mase"] = None
            errors.append(f"evaluate {label} failed: {e}")
            logger.warning("Evaluate [%s]: %s backtest failed: %s", view, label, e)
    return {"results": results, "errors": errors}


def _generic_backtest(P, sub, today_ts, holdout_weeks, fit_kwargs=None):
    """Walk-forward one-step-ahead backtest MASE for pipeline ``P`` on view ``sub``.

    The app re-runs weekly and only ever uses each forecast's FIRST week, so we
    score exactly that: step ``today`` back one week at a time for the last
    ``holdout_weeks`` completed weeks, re-fit at each step, and compare the first
    forecast week against that week's actual. Absolute errors pool across steps
    and SKUs; the score is that pool's mean divided by the pooled mean absolute
    error of a *baseline* forecast over the SAME points — a plain 8-week moving
    average of each SKU's actuals over the 8 weeks before the scored week
    (NaN/absent weeks simply drop out of the mean; ``agg`` is not densified).
    Scaling by the baseline makes the score comparable across views of very
    different volume: < 1 beats the 8-week average, and a near-zero forecast on
    a real-volume view scores >> 1 instead of a deceptively small raw MAE.

    Pool alignment: a point whose baseline is unavailable (no observed actuals
    in that SKU's prior 8 weeks — e.g. the SKU's first recorded week falls in
    the holdout) is dropped from BOTH pools, so numerator and denominator always
    cover identical points. Zero denominator: 0/0 -> 0.0 (the model matched a
    perfectly baseline-predictable series); x/0 -> None, never ``inf`` (invalid
    strict JSON in agent_summary files; None already means "unscoreable" and
    sorts last everywhere).

    Aggregate to SKU-week FIRST (splitting/merging the raw CUSTNMBR frame fans
    one forecast row onto many rows per (SKU, WeekDate) and skews the score).
    Each model windows its own training data off the ``today`` argument (trains
    up to the week before ``first_forecast_week``, forecasts starting at it), so
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

    model_abs = []     # |model forecast - actual| per (SKU, step)
    baseline_abs = []  # |8-week-average baseline - actual| over the SAME points
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
        # Baseline: each SKU's plain mean of observed actuals over the 8 weeks
        # before step_week. Precomputed as a single ``baseline`` column (never
        # merge prior-week POS/Orders — suffix collisions), left-merged so a
        # SKU with no observed actuals in the window surfaces as NaN and the
        # point drops from BOTH pools via the shared mask below.
        win = agg[
            (agg["WeekDate"] >= step_week - pd.Timedelta(weeks=8))
            & (agg["WeekDate"] < step_week)
        ]
        baseline = (
            win["POS"].fillna(win["Orders"])
            .groupby(win["SKU"]).mean()      # NaN actuals drop out of the mean
            .rename("baseline").reset_index()
        )
        merged = merged.merge(baseline, on="SKU", how="left")
        target = merged["POS"].fillna(merged["Orders"])
        mask = target.notna() & merged["baseline"].notna()
        if not mask.any():
            continue
        t = target[mask].to_numpy()
        model_abs.extend(np.abs(merged.loc[mask, "projected_pos"].to_numpy() - t))
        baseline_abs.extend(np.abs(merged.loc[mask, "baseline"].to_numpy() - t))
    if not model_abs:
        return None
    num = float(np.mean(model_abs))
    denom = float(np.mean(baseline_abs))
    if denom == 0.0:
        # Perfectly baseline-predictable series: a perfect model scores 0.0;
        # anything else is unbacktestable (None — never inf, which json.dump
        # would write as invalid ``Infinity``).
        return 0.0 if num == 0.0 else None
    return num / denom
