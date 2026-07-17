"""Deterministic demand-pattern profiling for the agent's model-fit reasoning.

Pure, LLM-free, unit-testable. ``demand_profile`` characterizes a view's demand
with the classic Syntetos-Boylan intermittency features so the reasoning node can
hand the LLM real numbers ("high intermittency, short history") instead of asking
it to guess whether demand is lumpy from a customer-group name.

The features it returns:

- ``weeks_of_history`` / ``sku_count`` / ``total_volume`` — scale.
- ``pct_zero_weeks``       — pooled fraction of zero SKU-week cells (intermittency).
- ``avg_demand_interval``  — ADI, median across SKUs of (periods / non-zero periods).
- ``cv2_demand_size``      — median across SKUs of the CV² of non-zero demand sizes
                             (lumpiness of the demand *quantity* when it occurs).
- ``pattern``              — the Syntetos-Boylan quadrant label from ADI + CV².

It scores against POS falling back to Orders — the same planner-relevant target the
backtest uses (see ``agent.nodes.evaluate._generic_backtest``). A demand week that
is simply absent from the file is a zero week (matching TSB's FILL_GAPS_WITH_ZERO):
each SKU is densified to weekly frequency from its first sale to the view's last week.

Kept out of ``reasoning.py`` so it carries no LLM dependency and can be tested with
plain synthetic frames.
"""

import numpy as np
import pandas as pd

from agent.data_io import view_frame

# Syntetos-Boylan (2005) cut-offs for the demand-pattern quadrant.
ADI_CUTOFF = 1.32
CV2_CUTOFF = 0.49


def _empty_profile() -> dict:
    """A well-defined 'unknown' profile for an empty/no-demand view.

    Every numeric feature is None so the prompt renderer can print 'n/a' and the
    LLM is told outright there is nothing to characterize, rather than being fed a
    fabricated zero that reads like a real (smooth) pattern."""
    return {
        "weeks_of_history": 0,
        "sku_count": 0,
        "total_volume": 0.0,
        "pct_zero_weeks": None,
        "avg_demand_interval": None,
        "cv2_demand_size": None,
        "pattern": "unknown",
    }


def _classify(adi: float, cv2: float) -> str:
    """Syntetos-Boylan quadrant label from ADI and CV²."""
    intermittent = adi >= ADI_CUTOFF
    lumpy_size = cv2 >= CV2_CUTOFF
    if not intermittent and not lumpy_size:
        return "smooth"
    if intermittent and not lumpy_size:
        return "intermittent"
    if not intermittent and lumpy_size:
        return "erratic"
    return "lumpy"


def demand_profile(view: str, cleaned_df, P=None) -> dict:
    """Characterize ``view``'s demand pattern from the cleaned demand frame.

    Returns a small JSON/prompt-safe dict (plain ints/floats/str). A view with no
    positive demand — or an empty/absent frame — returns ``_empty_profile()``
    rather than raising, so the reasoning node can always render *something*.
    """
    if cleaned_df is None or getattr(cleaned_df, "empty", True):
        return _empty_profile()

    sub = view_frame(cleaned_df, view, P)
    if sub is None or sub.empty:
        return _empty_profile()

    df = sub[["SKU", "WeekDate", "POS", "Orders"]].copy()
    df["WeekDate"] = pd.to_datetime(df["WeekDate"])
    # Planner-relevant target: POS, falling back to Orders (matches the backtest).
    demand = pd.to_numeric(df["POS"], errors="coerce")
    demand = demand.fillna(pd.to_numeric(df["Orders"], errors="coerce"))
    df["demand"] = demand

    # Aggregate customers up to one demand value per (SKU, week). ``min_count=1``
    # keeps a week NaN only when it had no observed value at all (an absent week),
    # which densification below then treats as a zero-demand week.
    sw = (
        df.groupby(["SKU", "WeekDate"], as_index=False)["demand"].sum(min_count=1)
    ).dropna(subset=["demand"])
    if sw.empty:
        return _empty_profile()

    # Weekly grid spanning the whole view. Weeks are Sunday-anchored and exactly
    # 7 days apart, so a 7-day date_range from first to last observed week
    # reproduces the real week anchors (and fills any gap weeks as zeros).
    weeks = pd.date_range(sw["WeekDate"].min(), sw["WeekDate"].max(), freq="7D")
    weeks_of_history = int(len(weeks))
    total_volume = float(sw["demand"].sum())

    adis: list[float] = []
    cv2s: list[float] = []
    zero_cells = 0
    total_cells = 0
    sku_count = 0
    for _, s in sw.groupby("SKU"):
        series = s.set_index("WeekDate")["demand"]
        # Collapse any duplicate week keys defensively, then start each SKU at its
        # first *positive* demand week (leading absence isn't intermittency).
        series = series.groupby(level=0).sum()
        positive = series[series > 0]
        if positive.empty:
            continue
        first = positive.index.min()
        grid = weeks[weeks >= first]
        dense = series.reindex(grid).fillna(0.0)
        n = int(len(dense))
        nonzero = int((dense > 0).sum())
        if n == 0 or nonzero == 0:
            continue
        sku_count += 1
        total_cells += n
        zero_cells += n - nonzero
        adis.append(n / nonzero)
        sizes = dense[dense > 0]
        # CV² of the non-zero demand sizes; a single demand point has no spread.
        cv2s.append(float((sizes.std(ddof=0) / sizes.mean()) ** 2) if nonzero > 1 else 0.0)

    if sku_count == 0:
        return _empty_profile()

    adi = float(np.median(adis))
    cv2 = float(np.median(cv2s))
    return {
        "weeks_of_history": weeks_of_history,
        "sku_count": sku_count,
        "total_volume": round(total_volume, 1),
        "pct_zero_weeks": round(100.0 * zero_cells / total_cells, 1) if total_cells else None,
        "avg_demand_interval": round(adi, 2),
        "cv2_demand_size": round(cv2, 2),
        "pattern": _classify(adi, cv2),
    }
