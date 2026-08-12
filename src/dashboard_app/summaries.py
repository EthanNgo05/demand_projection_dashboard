"""Pure summary/column helpers and timestamp formatting (no streamlit)."""
import numpy as np
import pandas as pd

from dashboard_app.config import PRICE_COL, fmt_when


# --------------------------------------------------------------------------- #
# Demand-signal helpers (POS-then-Orders, matching the pipeline)              #
# --------------------------------------------------------------------------- #
def resolve_avg_col(df):
    """Name of the descriptive-average column, whatever window it covers.

    The label varies by pipeline: regression always fits exactly 8 weeks
    ("8-Week POS/Orders Average"), while exponential-smoothing and XGBoost
    default to LOOKBACK_WEEKS=None ("All-Time POS/Orders Average") or an
    explicit N-week window if LOOKBACK_WEEKS is set. Matching by suffix keeps
    the dashboard correct regardless of which pipeline produced the summary.
    """
    # Literal rather than compute.EIGHT_WK_AVG_COL: this module is deliberately
    # streamlit-free, and compute.py imports streamlit.
    matches = [c for c in df.columns if c.endswith("POS/Orders Average")]
    return matches[0] if matches else "8-Week POS/Orders Average"


def avg_window_phrase(avg_col):
    """Human-readable window description derived from the average column's
    own label, e.g. "8-Week" or "All-Time" -- so KPI captions never say
    "8 wk" when the underlying average actually covers a different window."""
    return avg_col.replace(" POS/Orders Average", "")


def historical_window_label(avg_col):
    """Short window prefix for a weekly-demand metric label.

    "8-Week" or "All-Time", read straight off the average column's own name. The
    window a demand average covers follows the SELECTED MODEL (8 weeks for the
    8-Week Moving Average, all history for the other four), so without this in the
    label the two are indistinguishable on screen -- a planner comparing an 8-week
    run-rate against an all-time average would have no way to tell which they were
    looking at.

    Deliberately a pass-through, not a translation: the metric label and the table
    column must use the SAME word for the same window. Two vocabularies for one
    window ("All-History" in the column, "All-Time" in the metric) is exactly the
    confusion this function used to create. The ``" Week"`` -> ``"-Week"`` fixup is
    the one remaining concession, for frames persisted by an older build.
    """
    return avg_window_phrase(avg_col).replace(" Week", "-Week")


def source_map(summary):
    """SKU -> 'POS' or 'Orders' (whichever the forecast used).

    Only meaningful on a frame with ONE row per SKU. On a per-(SKU, customer)
    frame the dict comprehension silently lets the LAST group win, which for a
    mixed-source SKU is an arbitrary answer — use ``customer_source_map`` (and
    ``resolve_demand``) for those. ``historical_window`` prefers a precomputed
    ``demand`` column over this for exactly that reason.
    """
    if "Data Source" not in summary.columns:
        return {}
    return dict(zip(summary["SKU"].astype(str), summary["Data Source"]))


def resolve_demand(agg_by_group, by_cust):
    """Copy of a per-group SKU-week frame carrying a single ``demand`` column.

    Each (SKU, Customer Grouping) contributes the signal ITS OWN forecast was fit
    on — POS for the groups reporting sell-through, Orders for the ones that don't
    — so summing ``demand`` across customers is a total that covers every customer
    and is comparable to the summed forecast. Resolving one source per SKU instead
    is what made the old combined fit silently drop Orders-only customers.

    Falls back to POS for a pair ``by_cust`` has no row for (nothing was forecast
    there, so nothing sums into a forecast total either), and to POS wholesale when
    the frames don't carry what's needed — the pre-existing behaviour.
    """
    out = agg_by_group.copy()
    pos = out["POS"] if "POS" in out.columns else np.nan
    orders = out["Orders"] if "Orders" in out.columns else np.nan
    needed = {"Customer Grouping", "SKU", "Data Source"}
    if (by_cust is None or by_cust.empty or not needed <= set(by_cust.columns)
            or "Customer Grouping" not in out.columns):
        out["demand"] = pos
        return out

    # Vectorised on purpose: this frame is the per-group SKU-week aggregate — ~500k
    # rows for the all-customers view — so a dict lookup per row (which is what
    # customer_source_map + a comprehension would be) costs about a second of every
    # cold render. Reindexing a (group, SKU) Series does the same join in C.
    src = by_cust[["Customer Grouping", "SKU", "Data Source"]].copy()
    src["Customer Grouping"] = src["Customer Grouping"].astype(str)
    src["SKU"] = src["SKU"].astype(str).str.rstrip("*")
    lookup = (
        src.drop_duplicates(["Customer Grouping", "SKU"])
        .set_index(["Customer Grouping", "SKU"])["Data Source"]
    )
    idx = pd.MultiIndex.from_arrays([
        out["Customer Grouping"].astype(str),
        out["SKU"].astype(str).str.rstrip("*"),
    ])
    use_orders = lookup.reindex(idx).to_numpy() == "Orders"
    out["demand"] = np.where(use_orders, orders, pos)
    return out


def customer_source_map(summary):
    """(Customer Grouping, SKU) -> 'POS' or 'Orders' from a summary frame.

    Keyed per customer group so a table that carries raw Customers (e.g. the
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


def price_map_from_summary(df):
    """SKU -> list price (USD) from a summary/combined frame's ``PRICE_COL``.

    Returns ``{}`` when the frame is empty or carries no price column (list prices
    not loaded), so a chart built from it simply shows plain hovers. Keys are
    str-normalized to match the chart frames' str SKU comparisons.
    """
    if df is None or df.empty or PRICE_COL not in df.columns:
        return {}
    prices = pd.to_numeric(df[PRICE_COL], errors="coerce")
    return {
        str(s): float(p)
        for s, p in zip(df["SKU"].astype(str), prices)
        if not pd.isna(p)
    }


def historical_window(agg, summary, anchors):
    """Per SKU-week actual demand in the 8-week window, using each SKU's source.

    Adds a single 'demand' column = POS for POS-based SKUs, Orders for
    Orders-based SKUs, so totals line up with the (mixed-source) forecast.

    An ``agg`` that ALREADY carries ``demand`` is taken at its word and only
    windowed. That is how the summed views stay additive: ``resolve_demand``
    resolves the source per (SKU, customer) before the roll-up, which a per-SKU
    ``source_map`` cannot reproduce once the customers have been added together.
    """
    lb, lcw, _ = anchors
    h = agg[(agg["WeekDate"] >= lb) & (agg["WeekDate"] <= lcw)].copy()
    h["SKU"] = h["SKU"].astype(str)
    if "demand" in h.columns:
        return h
    src = source_map(summary)
    use_orders = h["SKU"].map(src).eq("Orders")
    orders = h["Orders"] if "Orders" in h.columns else np.nan
    h["demand"] = np.where(use_orders, orders, h["POS"])
    return h


def _format_generated_at(gen):
    """Format an ISO timestamp (e.g. '2026-07-17T14:12:00') as '2026-07-17 2:12 PM'.

    Thin wrapper over ``config.fmt_when`` — this used to hand-roll the 12-hour
    conversion, as did ``refresh.batch_elapsed_suffix`` and dashboard.py's
    "Latest snapshot" caption, and the three had drifted apart. Kept as a named
    function so its callers don't move.

    Falls back to the raw string if it can't be parsed, which ``fmt_when``
    signals by returning an em dash.
    """
    formatted = fmt_when(gen)
    return gen if formatted == "—" else formatted
