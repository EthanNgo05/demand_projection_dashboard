"""Pure summary/column helpers and timestamp formatting (no streamlit)."""
import datetime

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Demand-signal helpers (POS-then-Orders, matching the pipeline)              #
# --------------------------------------------------------------------------- #
def resolve_avg_col(df):
    """Name of the descriptive-average column, whatever window it covers.

    The label varies by pipeline: regression always fits exactly 8 weeks
    ("8 Week POS/Orders Average"), while exponential-smoothing and XGBoost
    default to LOOKBACK_WEEKS=None ("All-History POS/Orders Average") or an
    explicit N-week window if LOOKBACK_WEEKS is set. Matching by suffix keeps
    the dashboard correct regardless of which pipeline produced the summary.
    """
    matches = [c for c in df.columns if c.endswith("POS/Orders Average")]
    return matches[0] if matches else "8 Week POS/Orders Average"


def avg_window_phrase(avg_col):
    """Human-readable window description derived from the average column's
    own label, e.g. "8 Week" or "All-History" -- so KPI captions never say
    "8 wk" when the underlying average actually covers a different window."""
    return avg_col.replace(" POS/Orders Average", "")


def source_map(summary):
    """SKU -> 'POS' or 'Orders' (whichever the forecast used)."""
    if "Data Source" not in summary.columns:
        return {}
    return dict(zip(summary["SKU"].astype(str), summary["Data Source"]))


def customer_source_map(summary):
    """(Customer Grouping, SKU) -> 'POS' or 'Orders' from a summary frame.

    Keyed per customer group so a table that carries raw CUSTNMBRs (e.g. the
    missing-projections table) can be labelled with the same source the forecast
    used for that SKU in that group. SKUs are '*'-stripped on both sides so a
    trailing-star SKU still matches. Works for either the by-SKU summary (single
    group) or the by-SKU-and-customer table (every group)."""
    if summary is None or summary.empty:
        return {}
    if not {"Customer Grouping", "SKU", "Data Source"} <= set(summary.columns):
        return {}
    return {
        (str(g), str(s).rstrip("*")): src
        for g, s, src in zip(
            summary["Customer Grouping"],
            summary["SKU"],
            summary["Data Source"],
        )
    }


def historical_window(agg, summary, anchors):
    """Per SKU-week actual demand in the 8-week window, using each SKU's source.

    Adds a single 'demand' column = POS for POS-based SKUs, Orders for
    Orders-based SKUs, so totals line up with the (mixed-source) forecast.
    """
    lb, lcw, _ = anchors
    src = source_map(summary)
    h = agg[(agg["WeekDate"] >= lb) & (agg["WeekDate"] <= lcw)].copy()
    h["SKU"] = h["SKU"].astype(str)
    use_orders = h["SKU"].map(src).eq("Orders")
    orders = h["Orders"] if "Orders" in h.columns else np.nan
    h["demand"] = np.where(use_orders, orders, h["POS"])
    return h


def _format_generated_at(gen):
    """Format an ISO timestamp (e.g. '2026-07-17T14:12:00') as '2026-07-17 2:12 PM'.

    Falls back to the raw string if it can't be parsed.
    """
    try:
        dt = datetime.datetime.fromisoformat(str(gen))
    except (ValueError, TypeError):
        return gen
    # %I is zero-padded (02); lstrip("0") on the hour gives a cleaner "2:12 PM".
    hour = dt.strftime("%I").lstrip("0") or "12"
    return dt.strftime(f"%Y-%m-%d {hour}:%M %p")
