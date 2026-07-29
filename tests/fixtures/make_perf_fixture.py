"""Deterministic multi-year demand frame for the performance-parity goldens.

The Phase-2 fixture (``make_fixture.py``) is deliberately tiny — 12 SKUs and 9
historical weeks. That is too short to reach the code the performance work
touches: ``cleanse_series`` skips detection below ``OUTLIER_MIN_WEEKS``, and
Holt-Winters falls back to non-seasonal Holt below ``MIN_WEEKS_FOR_SEASONAL``
(104 weeks). A golden master built on it would pass no matter what we did to the
rolling-MAD or densification code.

This builder produces ~3 years of weekly history across several customer
groupings, with promo spikes, stockout dips and dead-SKU zero runs injected on
purpose so outlier detection, gap densification and the intermittent-demand
paths all actually fire. Seeded RNG -> identical data on every call, which is
what makes exact-match golden comparison meaningful.

Returns the frame in RAW export column names, so it feeds ``data_io._clean``
exactly as a real snapshot does. No .xlsx round-trip: the goldens only need the
numbers to be reproducible, not the file layout (``make_fixture.py`` already
covers the export layout).
"""

import numpy as np
import pandas as pd

# Pinned run date for every golden. A Wednesday, so week_anchors' "last complete
# week" logic has a partial current week to exclude (same shape as make_fixture).
TODAY = pd.Timestamp("2026-07-01")

# Raw Custnmbr values. AMAZON-DC + AMAZON-DS fold into one group; "Others - UK"
# is in CUSTOMERS_TO_IGNORE and must be dropped by _clean; the rest each become
# their own single-member group. Spans several fulfillment regions so the
# region-rollup views have something to roll up.
CUSTOMERS = [
    "AMAZON-DC",
    "AMAZON-DS",
    "COSTCO",
    "COSTCO-CAN",
    "SANIKAL-KG",
    "Web Sales - AU",
    "Others - UK",
]

N_SKUS = 24
HIST_WEEKS = 160          # > MIN_WEEKS_FOR_SEASONAL (104) -> real HW seasonal fits
FUT_WEEKS = 15            # matches the models' forward horizon

# Week grid: Sunday-start, ending at the last complete week before TODAY.
_CURRENT_WEEK_START = TODAY - pd.Timedelta(days=(TODAY.weekday() + 1) % 7)
_LAST_COMPLETE = _CURRENT_WEEK_START - pd.Timedelta(weeks=1)
HIST_INDEX = pd.date_range(end=_LAST_COMPLETE, periods=HIST_WEEKS, freq="W-SUN")
FUT_INDEX = pd.date_range(start=_CURRENT_WEEK_START, periods=FUT_WEEKS, freq="W-SUN")


def _series_for(rng, i, n_weeks):
    """One SKU's weekly demand level, with a distinct character per SKU index.

    The mix is deliberate — each ``i % 6`` bucket lands on a different branch of
    the model code, so the goldens cover more than one shape:
      0: smooth trend + annual seasonality  (Holt-Winters' seasonal path)
      1: intermittent, mostly zero          (TSB's path, cleanse_series edge cases)
      2: flat + promo spikes                (outlier detection, upper tail)
      3: trend + stockout dips              (outlier detection, lower tail)
      4: dead SKU - sells then goes to zero (probability decay, zero runs)
      5: short history - starts late        (gap densification, min_weeks guards)
    """
    t = np.arange(n_weeks, dtype="float64")
    kind = i % 6
    base = 40.0 + 12.0 * (i % 5)

    if kind == 0:
        level = base + 0.25 * t + 10.0 * np.sin(2 * np.pi * t / 52.0)
    elif kind == 1:
        level = np.where(rng.random(n_weeks) < 0.25, base * rng.uniform(0.5, 2.0, n_weeks), 0.0)
    elif kind == 2:
        level = np.full(n_weeks, base)
        level[rng.choice(n_weeks, size=max(n_weeks // 26, 1), replace=False)] *= 6.0
    elif kind == 3:
        level = base + 0.15 * t
        level[rng.choice(n_weeks, size=max(n_weeks // 30, 1), replace=False)] = 0.0
    elif kind == 4:
        level = base + 0.1 * t
        level[int(n_weeks * 0.7):] = 0.0
    else:
        level = base + 0.2 * t
        level[: int(n_weeks * 0.8)] = np.nan   # no data -> rows omitted entirely

    noise = rng.normal(0, base * 0.07, n_weeks)
    return np.maximum(level + noise, 0.0)


def build_frame():
    """Raw-export-shaped DataFrame (one row per SKU x customer x week)."""
    rng = np.random.default_rng(20260701)
    rows = []
    for i in range(N_SKUS):
        sku = f"PF-{i + 1:03d}"
        desc = f"Perf Product {i + 1}"
        n_cust = 2 + (i % 3)
        custs = [CUSTOMERS[(i + k) % len(CUSTOMERS)] for k in range(n_cust)]
        # Every 5th SKU reports Orders only (no POS) -> exercises the fallback.
        orders_only = i % 5 == 0
        for cust in custs:
            level = _series_for(rng, i, HIST_WEEKS)
            for wk, lv in zip(HIST_INDEX, level):
                if np.isnan(lv):
                    continue          # genuinely absent week -> a gap to densify
                pos = np.nan if orders_only else float(round(lv))
                orders = float(round(lv * rng.uniform(0.7, 1.1)))
                proj = float(round(lv * rng.uniform(0.85, 1.15)))
                rows.append([sku, desc, cust, wk, pos, orders, proj])
            # Forward weeks carry the existing plan only, like a real export.
            tail = float(np.nanmean(level[-8:])) if np.isfinite(level[-8:]).any() else 0.0
            for wk in FUT_INDEX:
                # Every 4th SKU is projected 0 forward -> feeds the spikes scan.
                proj = 0.0 if i % 4 == 0 else float(round(tail))
                rows.append([sku, desc, cust, wk, np.nan, np.nan, proj])

    return pd.DataFrame(
        rows,
        columns=["'Demand'[DisplaySKU]", "Description", "Custnmbr", "WeekDate",
                 "POS", "Sum of Quantity", "Projection"],
    )


def build_prices():
    """SKU -> list price Series, so revenue-risk columns are exercised too."""
    rng = np.random.default_rng(7)
    skus = [f"PF-{i + 1:03d}" for i in range(N_SKUS)]
    return pd.Series(
        [round(float(rng.uniform(9.99, 249.99)), 2) for _ in skus],
        index=pd.Index(skus, name="SKU"), name="List Price",
    )


if __name__ == "__main__":
    f = build_frame()
    print(f"{len(f):,} rows, {f['\'Demand\'[DisplaySKU]'].nunique()} SKUs, "
          f"{f['WeekDate'].nunique()} weeks")
