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
    return load_pipeline(MODEL_OPTIONS["Simple Regression"])


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


def test_generic_backtest_returns_low_mae_on_flat_series(regression_module):
    df = _flat_series()
    mae = _generic_backtest(
        regression_module, df, df["WeekDate"].max(), holdout_weeks=4
    )
    assert mae is not None
    assert mae < 10  # flat, low-noise series should backtest tightly


def test_generic_backtest_none_when_no_holdout_data(regression_module):
    df = _flat_series(weeks=3)  # too short for a 6-week holdout
    mae = _generic_backtest(
        regression_module, df, df["WeekDate"].max(), holdout_weeks=6
    )
    assert mae is None


def test_evaluate_models_populates_mae_for_all_three(sample_results_state):
    out = evaluate_models(sample_results_state)
    assert len(out["results"]) == 3, f"expected 3 models, errors: {out['errors']}"
    for label, r in out["results"].items():
        assert r.get("mae") is not None, f"{label} missing an mae after evaluate_models"


def test_es_score_comes_from_generic_backtest_not_autofit(sample_results_state):
    # guards the apples-to-apples rule: ES's comparison score ("mae") must come
    # from the shared backtest; autofit's own non-comparable MAE is stored
    # separately as "autofit_mae"
    out = evaluate_models(sample_results_state)
    es = out["results"]["Holt's Exponential Smoothing"]
    assert es.get("mae") is not None
    assert "autofit_mae" in es  # recorded for the audit trail
    assert es["mae"] is not es["autofit_mae"]  # not just copied through
