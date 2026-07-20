"""Deterministic demand-pattern profiling (agent.demand_profile).

Pure function over a cleaned demand frame — no LLM, no network — so these assert
exact classification behaviour on hand-built synthetic series.
"""

import numpy as np
import pandas as pd

from agent.demand_profile import demand_profile

COLS = ["SKU", "Description", "Customer", "WeekDate", "POS", "Orders",
        "Projection", "Customer Grouping"]


def _frame(rows):
    return pd.DataFrame(rows, columns=COLS)


def _weeks(n, start="2024-01-07"):
    return pd.date_range(start, periods=n, freq="7D")


def test_intermittent_series_classified_intermittent():
    """A mostly-zero series (demand every ~5th week, steady size) is intermittent:
    high % zero weeks, ADI >= 1.32, but low size-lumpiness (CV² < 0.49)."""
    rows = []
    for i, wk in enumerate(_weeks(40)):
        pos = 10.0 if i % 5 == 0 else 0.0  # demand every 5th week
        rows.append(["SKU-INT", "widget", "CUST", wk, pos, np.nan, 0, "GRP"])
    prof = demand_profile("GRP", _frame(rows))

    assert prof["pattern"] == "intermittent"
    assert prof["avg_demand_interval"] >= 1.32
    assert prof["cv2_demand_size"] < 0.49
    assert prof["pct_zero_weeks"] > 50
    assert prof["sku_count"] == 1


def test_smooth_series_classified_smooth():
    """A dense series with demand every week and steady size is smooth:
    ADI ~= 1 and low CV²."""
    rng = np.random.default_rng(0)
    rows = []
    for wk in _weeks(40):
        pos = float(100 + rng.integers(-3, 4))  # steady ~100/wk
        rows.append(["SKU-SMOOTH", "widget", "CUST", wk, pos, np.nan, 0, "GRP"])
    prof = demand_profile("GRP", _frame(rows))

    assert prof["pattern"] == "smooth"
    assert prof["avg_demand_interval"] < 1.32
    assert prof["cv2_demand_size"] < 0.49
    assert prof["pct_zero_weeks"] == 0.0


def test_lumpy_series_classified_lumpy():
    """Sparse AND highly variable demand sizes -> lumpy quadrant."""
    sizes = {0: 5.0, 6: 200.0, 13: 15.0, 27: 400.0, 35: 8.0}  # rare + wildly varying
    rows = []
    for i, wk in enumerate(_weeks(40)):
        rows.append(["SKU-LUMP", "widget", "CUST", wk, sizes.get(i, 0.0), np.nan, 0, "GRP"])
    prof = demand_profile("GRP", _frame(rows))

    assert prof["pattern"] == "lumpy"
    assert prof["avg_demand_interval"] >= 1.32
    assert prof["cv2_demand_size"] >= 0.49


def test_absent_weeks_count_as_zero_demand():
    """A week simply missing from the file is a zero-demand week (matches TSB's
    fill-gaps-with-zero), so intermittency is measured even without explicit 0 rows."""
    weeks = _weeks(40)
    rows = [["SKU-GAP", "w", "CUST", weeks[i], 10.0, np.nan, 0, "GRP"]
            for i in (0, 8, 16, 24, 32)]  # 5 demand weeks; span first->last = 33 weeks
    prof = demand_profile("GRP", _frame(rows))

    # History spans first observed demand week to the last (index 0..32 = 33),
    # not the full 40 — weeks after the last sale carry no demand signal.
    assert prof["weeks_of_history"] == 33
    assert prof["pct_zero_weeks"] > 80
    assert prof["avg_demand_interval"] >= 1.32


def test_orders_used_when_pos_missing():
    """POS falling back to Orders — the backtest target convention."""
    rows = [["SKU-O", "w", "CUST", wk, np.nan, 50.0, 0, "GRP"] for wk in _weeks(20)]
    prof = demand_profile("GRP", _frame(rows))
    assert prof["sku_count"] == 1
    assert prof["total_volume"] == 1000.0  # 20 weeks * 50


def test_empty_and_no_demand_return_unknown():
    """No frame, an empty frame, or a view with only zeros -> the unknown profile,
    never a raise."""
    assert demand_profile("GRP", None)["pattern"] == "unknown"
    assert demand_profile("GRP", _frame([]))["pattern"] == "unknown"

    zeros = [["SKU-Z", "w", "CUST", wk, 0.0, np.nan, 0, "GRP"] for wk in _weeks(10)]
    prof = demand_profile("GRP", _frame(zeros))
    assert prof["pattern"] == "unknown"
    assert prof["avg_demand_interval"] is None


def test_view_filtering_scopes_the_profile():
    """demand_profile honours the view filter (via view_frame): a group with dense
    demand profiles differently from a sibling intermittent group in the frame."""
    rows = []
    for wk in _weeks(30):
        rows.append(["SKU-A", "w", "CUST", wk, 100.0, np.nan, 0, "DENSE"])
    for i, wk in enumerate(_weeks(30)):
        rows.append(["SKU-B", "w", "CUST", wk, (20.0 if i % 6 == 0 else 0.0),
                     np.nan, 0, "SPARSE"])
    frame = _frame(rows)

    assert demand_profile("DENSE", frame)["pattern"] == "smooth"
    assert demand_profile("SPARSE", frame)["avg_demand_interval"] >= 1.32
