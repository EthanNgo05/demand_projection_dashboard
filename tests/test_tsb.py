"""TSB model: recursion math, obsolescence decay, fallbacks, pipeline contract.

Loads src/models/tsb.py exactly the way both front-ends do (by file path via
the agent's model loader), then drives it with hand-built zero-inflated series
where every expected value can be computed by re-running the TSB recursion in
the test. The shared-scaffolding behaviours (POS/Orders fallback, summary
columns, parity with dashboard.compute_view) are covered by the phase 2/3/6
suites, which iterate MODEL_OPTIONS and pick TSB up automatically.
"""

import numpy as np
import pandas as pd
import pytest

from agent.config import MODEL_OPTIONS
from agent.model_loader import load_pipeline

TSB_LABEL = "TSB (intermittent demand)"


@pytest.fixture(scope="module")
def TSB():
    return load_pipeline(MODEL_OPTIONS[TSB_LABEL])


def _intermittent_df(pattern, sku="SKU-INT", orders=False, start="2025-11-02"):
    """cleaned_df-shaped frame with an exact weekly demand pattern (zeros incl.).

    Weeks are consecutive Sundays starting at ``start``. Pair with
    ``_today_for(weeks)`` so the LAST pattern week is a completed week: today
    lands mid-way through the following week, and week_anchors excludes only
    that (empty) in-progress week from training.
    """
    weeks = pd.date_range(start, periods=len(pattern), freq="7D")
    values = [float(v) for v in pattern]
    return pd.DataFrame(
        {
            "SKU": sku,
            "Description": "synthetic intermittent",
            "WeekDate": weeks,
            "POS": np.nan if orders else values,
            "Orders": values if orders else np.nan,
            "Projection": np.nan,
        }
    )


def _today_for(df):
    return df["WeekDate"].max() + pd.Timedelta(days=10)  # mid next (empty) week


def _expected_rate(pattern, alpha_p, alpha_z):
    """Re-run the TSB recursion by hand: p0/z0 from overall series stats."""
    y = np.asarray(pattern, dtype="float64")
    nz = y > 0
    p = float(nz.mean())
    z = float(y[nz].mean()) if nz.any() else 0.0
    for v in y:
        d = 1.0 if v > 0 else 0.0
        p += alpha_p * (d - p)
        if d:
            z += alpha_z * (v - z)
    return max(p * z, 0.0)


# Mostly-zero pattern: global MAD is 0, so cleanse_series' auto-detection
# disarms itself (scale > 0 never fires) and the fit sees the raw values --
# the expected forecast is computable straight from the pattern.
PATTERN = [0, 0, 12, 0, 0, 0, 10, 0, 0, 8, 0, 0, 0, 9, 0, 0]


def test_tsb_forecast_matches_hand_run_recursion(TSB):
    rate = _expected_rate(PATTERN, TSB.ALPHA_P, TSB.ALPHA_Z)
    out = TSB.tsb_forecast(np.asarray(PATTERN, dtype="float64"), 15)
    assert len(out) == 15
    assert out == pytest.approx([rate] * 15)


def test_fit_tsb_weekly_output_is_flat_rounded_rate(TSB):
    df = _intermittent_df(PATTERN)
    agg = TSB.aggregate_to_sku_week(df)
    summary, weekly = TSB.fit_regression(agg, _today_for(df), grouping_label="t")

    expected = max(int(round(_expected_rate(PATTERN, TSB.ALPHA_P, TSB.ALPHA_Z))), 0)
    assert len(weekly) == 15  # 15 rows per SKU, not 15 columns
    assert (weekly["projected_pos"] == expected).all()
    assert weekly["promo_uplift"].eq(1.0).all()  # uplift disabled while flattened
    import datetime

    assert isinstance(weekly["WeekDate"].iloc[0], datetime.date)


def test_obsolescence_decay_toward_zero(TSB):
    # TSB's distinguishing trait vs Croston: trailing zero weeks keep pulling
    # the probability (and the forecast) down, so a SKU that stops selling
    # decays toward 0 instead of freezing at its last positive estimate.
    prefix = [10, 0, 12, 0, 0, 11, 0, 9]
    rates = [
        TSB.tsb_forecast(np.asarray(prefix + [0] * n, dtype="float64"), 1)[0]
        for n in (0, 8, 20, 60)
    ]
    assert rates[0] > rates[1] > rates[2] > rates[3]  # strictly decaying
    assert round(rates[3]) == 0  # long-dead SKU projects 0 whole units


def test_stationary_intermittent_rate_near_series_mean(TSB):
    # Fixed period-3 demand of constant size: p*z should sit near the series
    # mean (p ~= 1/3, z ~= 9), neither exploding nor collapsing.
    pattern = [0, 0, 9] * 12
    rate = TSB.tsb_forecast(np.asarray(pattern, dtype="float64"), 1)[0]
    assert rate == pytest.approx(np.mean(pattern), rel=0.15)


def test_short_history_falls_back_to_flat_mean(TSB):
    df = _intermittent_df([6, 0, 3])  # 3 weeks < MIN_WEEKS_FOR_TREND=4
    agg = TSB.aggregate_to_sku_week(df)
    today = _today_for(df)

    summary, weekly = TSB.fit_regression(agg, today, grouping_label="t")
    assert (weekly["projected_pos"] == round(np.mean([6, 0, 3]))).all()
    assert summary["Weeks with data"].iloc[0] == 3

    # The dashboard's min-weeks slider passes min_weeks_for_trend through;
    # lowering it flips the same series onto the TSB path.
    _, weekly_tsb = TSB.fit_regression(
        agg, today, grouping_label="t", min_weeks_for_trend=2
    )
    expected = max(int(round(_expected_rate([6, 0, 3], TSB.ALPHA_P, TSB.ALPHA_Z))), 0)
    assert (weekly_tsb["projected_pos"] == expected).all()


def test_all_zero_series_projects_zero(TSB):
    out = TSB.tsb_forecast(np.zeros(20), 15)
    assert out == [0.0] * 15


def test_deterministic_across_runs(TSB):
    df = _intermittent_df(PATTERN)
    agg = TSB.aggregate_to_sku_week(df)
    today = _today_for(df)
    s1, w1 = TSB.fit_regression(agg, today, grouping_label="t")
    s2, w2 = TSB.fit_regression(agg, today, grouping_label="t")
    pd.testing.assert_frame_equal(s1, s2)
    pd.testing.assert_frame_equal(w1, w2)


def test_contract_shape_and_orders_fallback(TSB):
    df = _intermittent_df(PATTERN, orders=True)  # POS all NaN -> Orders signal
    agg = TSB.aggregate_to_sku_week(df)
    summary, weekly = TSB.fit_regression(agg, _today_for(df), grouping_label="t")

    assert summary.columns.tolist() == TSB.SUMMARY_COLUMNS
    assert summary["Data Source"].iloc[0] == "Orders"
    assert summary["Customer Grouping"].iloc[0] == "t"
    # Orders-only series forecasts the same rate as the POS version.
    expected = max(int(round(_expected_rate(PATTERN, TSB.ALPHA_P, TSB.ALPHA_Z))), 0)
    assert (weekly["projected_pos"] == expected).all()


def test_no_smoothing_or_autofit_surface(TSB):
    # Guards the dashboard introspection contract: TSB must never grow
    # alpha/beta/phi fit args (sliders) or autofit_smoothing (Autofit button).
    import inspect

    params = set(inspect.signature(TSB.fit_regression).parameters)
    assert not ({"alpha", "beta", "phi"} & params)
    assert "min_weeks_for_trend" in params and "list_prices" in params
    assert not hasattr(TSB, "autofit_smoothing")
