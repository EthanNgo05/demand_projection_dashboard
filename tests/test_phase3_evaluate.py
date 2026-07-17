"""Phase 3: _generic_backtest sanity + evaluate_models apples-to-apples rule."""

import numpy as np
import pandas as pd
import pytest

from agent.config import ALL_CUSTOMERS_VIEW, MODEL_OPTIONS
from agent.model_loader import load_pipeline
from agent.nodes.evaluate import _generic_backtest, evaluate_models
from agent.nodes.forecast import run_all_models


@pytest.fixture(scope="module")
def regression_module():
    return load_pipeline(MODEL_OPTIONS["8-Week Moving Average"])


@pytest.fixture(scope="module")
def sample_results_state():
    """State after run_all_models on a synthetic cleaned_df with LONG history.

    The shared workbook fixture only carries 9 historical weeks -- too short
    for autofit's rolling folds (each fold needs >= AUTOFIT_MIN_TRAIN_WEEKS=8
    training weeks behind a 6-week holdout). 30 weeks gives every model and
    autofit enough room, which this file's tests depend on.
    """
    rng = np.random.default_rng(42)
    weeks = pd.date_range("2025-11-02", periods=30, freq="7D")  # Sundays
    frames = []
    for i, sku in enumerate(["SKU-A", "SKU-B", "SKU-C"]):
        frames.append(
            pd.DataFrame(
                {
                    "SKU": sku,
                    "Description": f"Synth product {sku}",
                    "CUSTNMBR": "CUST-1",
                    "Customer Grouping": "TEST-GROUP",
                    "WeekDate": weeks,
                    "POS": 80 + 25 * i + rng.normal(0, 5, len(weeks)),
                    "Orders": np.nan,
                    "Projection": np.nan,
                }
            )
        )
    cleaned_df = pd.concat(frames, ignore_index=True)
    today_ts = pd.to_datetime(cleaned_df["WeekDate"]).max() + pd.Timedelta(days=3)
    state = {
        "view": ALL_CUSTOMERS_VIEW,
        "today_ts": today_ts,
        "cleaned_df": cleaned_df,
        "prices": None,
        "errors": [],
    }
    state.update(run_all_models(state))
    assert state["results"], f"no models ran: {state['errors']}"
    return state


def _flat_series(weeks=20, level=100):
    dates = pd.date_range("2026-01-05", periods=weeks, freq="W")
    return pd.DataFrame(
        {
            "SKU": "TEST-SKU",
            "Description": "test",
            "WeekDate": dates,
            "POS": level + np.random.default_rng(0).normal(0, 2, weeks),
            "Orders": np.nan,
            "Projection": np.nan,  # aggregate_to_sku_week requires the column
        }
    )


def test_generic_backtest_returns_sane_mase_on_flat_series(regression_module):
    # On a flat low-noise series the MA-plus-trend-nudge model and the pure
    # 8-week-MA baseline make near-identical errors, so MASE lands near 1
    # (the trend nudge adds a little estimation noise on pure noise).
    df = _flat_series()
    mase = _generic_backtest(
        regression_module, df, df["WeekDate"].max(), holdout_weeks=4
    )
    assert mase is not None
    assert 0 <= mase < 1.5


def test_generic_backtest_none_when_no_holdout_data(regression_module):
    # Walk-forward: every step needs at least one training week BEFORE the week
    # it scores. With only 2 weeks each step's training window is empty, so no
    # step is scoreable and the backtest returns None.
    df = _flat_series(weeks=2)
    mase = _generic_backtest(
        regression_module, df, df["WeekDate"].max(), holdout_weeks=6
    )
    assert mase is None


def test_evaluate_models_populates_mase_for_all_models(sample_results_state):
    out = evaluate_models(sample_results_state)
    assert len(out["results"]) == len(MODEL_OPTIONS), (
        f"expected every model in MODEL_OPTIONS to run, errors: {out['errors']}"
    )
    for label, r in out["results"].items():
        assert r.get("mase") is not None, f"{label} missing a mase after evaluate_models"


def test_es_score_comes_from_generic_backtest_not_autofit(sample_results_state):
    # guards the apples-to-apples rule: ES's comparison score ("mase") must come
    # from the shared backtest; autofit's own non-comparable MAE is stored
    # separately as "autofit_mae"
    out = evaluate_models(sample_results_state)
    es = out["results"]["Holt's (double) exponential smoothing"]
    assert es.get("mase") is not None
    assert "autofit_mae" in es  # recorded for the audit trail
    assert "mae" not in es  # the old raw-MAE comparison key must be gone
    assert es["mase"] is not es["autofit_mae"]  # not just copied through


class _StubPipeline:
    """Minimal pipeline-contract stand-in with an exactly controllable forecast.

    ``aggregate_to_sku_week`` is the identity; ``fit_regression`` forecasts a
    fixed value (or ``value_fn(step_week)``) for every SKU at the first agg
    week strictly after ``today`` — mirroring where the real models place
    their first forecast week — so both MASE pools can be computed by hand.
    """

    def __init__(self, value=None, value_fn=None):
        self._value = value
        self._value_fn = value_fn

    def aggregate_to_sku_week(self, df):
        return df[["SKU", "WeekDate", "POS", "Orders", "Projection"]].copy()

    def fit_regression(self, agg, today, grouping_label="", **kwargs):
        future = sorted(pd.to_datetime(agg["WeekDate"]).unique())
        future = [w for w in future if w > pd.Timestamp(today)]
        if not future:
            return None, None
        step_week = future[0]
        value = self._value_fn(step_week) if self._value_fn else self._value
        weekly = pd.DataFrame(
            {
                "SKU": sorted(agg["SKU"].unique()),
                "WeekDate": step_week,
                "projected_pos": value,
            }
        )
        return None, weekly


def test_mase_scale_free_zero_forecast_penalized():
    # The behavior raw MAE got wrong: on a level-100 view a flat-zero forecast
    # used to look "small" against an absolute threshold. Scaled by the 8-week
    # average baseline it scores >> 1, while a perfect forecast scores 0.
    df = _flat_series()
    today = df["WeekDate"].max()
    zero_mase = _generic_backtest(_StubPipeline(value=0), df, today, holdout_weeks=4)
    assert zero_mase is not None and zero_mase > 5

    actual_by_week = df.set_index("WeekDate")["POS"]
    perfect = _StubPipeline(value_fn=lambda wk: float(actual_by_week.loc[wk]))
    assert _generic_backtest(perfect, df, today, holdout_weeks=4) == 0.0


def test_mase_zero_denominator_rules():
    # Exactly-constant actuals: the 8-week-average baseline is exact (denom 0).
    # A matching model returns 0.0; a mismatched one is unbacktestable (None,
    # never inf — inf is invalid strict JSON in the published summaries).
    df = _flat_series(weeks=20, level=50)
    df["POS"] = 50.0  # overwrite the noise: perfectly constant
    today = df["WeekDate"].max()
    assert _generic_backtest(_StubPipeline(value=50), df, today, holdout_weeks=4) == 0.0
    assert _generic_backtest(_StubPipeline(value=60), df, today, holdout_weeks=4) is None


def test_mase_pools_stay_aligned_when_baseline_missing():
    # SKU-NEW first appears in the scored week itself (no actuals in the prior
    # 8 weeks -> no baseline), so its point must drop from BOTH pools. SKU-OLD
    # also has one NaN week inside the baseline window, which must be skipped
    # by the mean (not treated as 0). Expected MASE is hand-computed from the
    # single surviving (SKU-OLD, W10) point.
    weeks = pd.date_range("2026-01-04", periods=10, freq="7D")  # W1..W10 Sundays
    old_pos = [5, 10, 20, np.nan, 40, 50, 60, 70, 80, 100]  # W4 NaN, W10 actual=100
    frames = [
        pd.DataFrame(
            {"SKU": "SKU-OLD", "Description": "old", "WeekDate": weeks,
             "POS": old_pos, "Orders": np.nan, "Projection": np.nan}
        ),
        pd.DataFrame(
            {"SKU": "SKU-NEW", "Description": "new", "WeekDate": [weeks[-1]],
             "POS": [55.0], "Orders": np.nan, "Projection": np.nan}
        ),
    ]
    df = pd.concat(frames, ignore_index=True)
    today = weeks[-1] + pd.Timedelta(days=3)  # holdout step 1 scores W10

    mase = _generic_backtest(_StubPipeline(value=90), df, today, holdout_weeks=1)
    # Baseline window for W10 = W2..W9; SKU-OLD mean skips the W4 NaN:
    # (10+20+40+50+60+70+80)/7 = 330/7. Pools: num = |90-100| = 10,
    # denom = |330/7 - 100| = 370/7 -> MASE = 70/370. If SKU-NEW leaked into
    # the numerator (|90-55| = 35) or the NaN counted as 0 (denom 58.75), the
    # value would differ.
    assert mase == pytest.approx(70 / 370)
