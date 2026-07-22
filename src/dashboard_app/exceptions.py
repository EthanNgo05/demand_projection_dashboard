"""The Exceptions view: SKUs whose recent actual sell-through has diverged
sharply from the existing system projection (the plan of record).

Unlike "Projection Difference"/"Revenue Risk" (which compare OUR model forecast
to the system projection) this is a pure ACTUALS-vs-PLAN comparison, so it needs
no forecasting fit and does not depend on the agent batch:

    recent = 8-week POS/Orders run-rate      (POS if the SKU has any, else Orders)
    proj   = system Projection averaged over the same 15 forward weeks the models use
    gap    = recent - proj                    (>0 under-projected, <0 over-projected)
    pct    = gap / proj                        (signed; undefined when proj is 0/NaN)
    impact = gap * list_price                  (per week, in USD)

Recent run-rate and the forward window come from the same helpers the models use
(`_descriptive_averages`, the pipeline's `week_anchors`/`aggregate_to_sku_week`),
so the numbers agree with what the other views show.
"""
import numpy as np
import pandas as pd
import streamlit as st

from dashboard_app.compute import EIGHT_WK_AVG_COL, _descriptive_averages
from dashboard_app.config import PRICE_COL
from dashboard_app.tables import render_filtered_table

# Display column names for the exceptions table. RECENT_COL / PROJ_COL reuse the
# names style_summary already formats (1-decimal run-rate, comma'd integer
# projection); the rest render as pre-rounded numbers under self-describing
# headers.
RECENT_COL = EIGHT_WK_AVG_COL            # "8-Week POS/Orders Average"
PROJ_COL = "System Projection (avg/wk)"
GAP_COL = "POS - Projection (avg/wk)"
PCT_COL = "% vs Projection"
IMPACT_COL = "$ Impact/wk"
FLAG_COL = "Flag"
DIRECTION_COL = "Direction"

UNDER = "Under-projected (stockout risk)"   # recent >> plan: selling faster than planned
OVER = "Over-projected (overstock risk)"    # recent << plan: planned but not selling

# Session-signature tag kept distinct from the view ID string so a rename of the
# user-facing label never silently reuses another view's cache entry.
EXCEPTIONS_VIEW_SIG = "exceptions-v1"

_DISPLAY_COLS = [
    "SKU", "Description", "Customer Grouping", "Region", "Data Source",
    RECENT_COL, PROJ_COL, GAP_COL, PCT_COL, IMPACT_COL, FLAG_COL, PRICE_COL,
]


def _forward_projection_avg(agg_by_group, first_forecast_week, last_forecast_week):
    """Per-(Customer Grouping, SKU) mean of the system Projection over the 15
    forward weeks — the same definition as the models' ``initial_projection_avg``
    (weeks with a missing projection are skipped so a SKU whose projection runs
    out mid-horizon isn't penalised for the blank weeks)."""
    fwd = agg_by_group[
        (agg_by_group["WeekDate"] >= pd.Timestamp(first_forecast_week))
        & (agg_by_group["WeekDate"] <= pd.Timestamp(last_forecast_week))
    ]
    return (
        fwd.dropna(subset=["Projection"])
        .groupby(["Customer Grouping", "SKU"], as_index=False)["Projection"]
        .mean()
        .rename(columns={"Projection": PROJ_COL})
    )


def _recent_data_source(agg_by_group, today_ts):
    """Per-(Customer Grouping, SKU) label of which signal fed the recent run-rate
    — "POS" if the SKU had any POS in the 8-week window, else "Orders" — matching
    the POS-then-Orders fallback ``_descriptive_averages`` uses."""
    days_since_sunday = (today_ts.weekday() + 1) % 7
    current_week_start = today_ts - pd.Timedelta(days=days_since_sunday)
    last_complete_week = current_week_start - pd.Timedelta(weeks=1)
    eight_wk_start = last_complete_week - pd.Timedelta(weeks=7)
    win = agg_by_group[
        (agg_by_group["WeekDate"] >= eight_wk_start)
        & (agg_by_group["WeekDate"] <= last_complete_week)
    ]
    src = (
        win.groupby(["Customer Grouping", "SKU"])["POS"]
        .apply(lambda s: "POS" if s.notna().any() else "Orders")
        .reset_index(name="Data Source")
    )
    return src


def compute_exceptions(df, today_ts, prices, P):
    """Build the (unfiltered, unsorted-for-display) exceptions frame.

    Pure/deterministic — no Streamlit — so it is unit-testable. ``df`` is the
    cleaned demand frame (SKU/Customer/WeekDate/POS/Orders/Projection +
    "Customer Grouping"); ``prices`` is the SKU→list-price map (or None); ``P``
    is the loaded pipeline (only its model-agnostic ``aggregate_to_sku_week`` /
    ``week_anchors`` / ``region_for_group`` are used). Returns a DataFrame with
    ``_DISPLAY_COLS`` plus a hidden ``_sort`` key, one row per flagged
    (SKU, Customer Grouping); direction lives in ``DIRECTION_COL``.
    """
    empty = pd.DataFrame(columns=_DISPLAY_COLS + [DIRECTION_COL])
    if df is None or df.empty:
        return empty

    # Per-group SKU-week aggregates, tagged with the group (mirrors compute.py).
    agg_frames = []
    for group, sub in df.groupby("Customer Grouping"):
        ag = P.aggregate_to_sku_week(sub)
        ag["Customer Grouping"] = group
        agg_frames.append(ag)
    if not agg_frames:
        return empty
    agg_by_group = pd.concat(agg_frames, ignore_index=True)
    agg_by_group["WeekDate"] = pd.to_datetime(agg_by_group["WeekDate"])
    # Discontinued SKUs (trailing '*') are handled by the data-quality tables;
    # drop them here so they don't double-surface (also matches _descriptive_averages).
    agg_by_group = agg_by_group[~agg_by_group["SKU"].astype(str).str.endswith("*")]
    if agg_by_group.empty:
        return empty

    # recent run-rate (shared helper) and forward system-projection average.
    recent = _descriptive_averages(agg_by_group, today_ts)[
        ["Customer Grouping", "SKU", RECENT_COL]
    ]
    _, _, first_forecast_week = P.week_anchors(today_ts)
    forecast_weeks = pd.date_range(start=first_forecast_week, periods=15, freq="W-SUN")
    proj = _forward_projection_avg(agg_by_group, first_forecast_week, forecast_weeks[-1])

    # Universe = every SKU with recent activity OR a forward projection.
    frame = recent.merge(proj, on=["Customer Grouping", "SKU"], how="outer")
    # A SKU with history but nothing in the last 8 weeks has a genuine 0 run-rate
    # (absent week = zero, matching the models' gap-fill).
    frame[RECENT_COL] = frame[RECENT_COL].fillna(0.0)
    proj_missing = frame[PROJ_COL].isna()          # no plan of record at all
    # Keep the filled projection as a column so it survives the later merge (which
    # resets the index) — carrying it as a separate Series would misalign.
    frame["_proj"] = frame[PROJ_COL].fillna(0.0)

    frame["_gap"] = frame[RECENT_COL] - frame["_proj"]
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["_pct"] = np.where(
            frame["_proj"] != 0, frame["_gap"] / frame["_proj"], np.nan
        )

    # Flags for the two edge cases that make % undefined or degenerate.
    frame[FLAG_COL] = ""
    frame.loc[proj_missing | (frame["_proj"] == 0), FLAG_COL] = "no plan"
    frame.loc[(frame[RECENT_COL] == 0) & (frame["_proj"] > 0), FLAG_COL] = "no recent sales"
    # "no recent sales" is a full over-projection: recent 0 vs a real plan = -100%.
    frame.loc[(frame[RECENT_COL] == 0) & (frame["_proj"] > 0), "_pct"] = -1.0

    # Drop rows with no signal either way (nothing planned and nothing selling)
    # and rows with no gap to flag.
    frame = frame[(frame[RECENT_COL] != 0) | (frame["_proj"] != 0)]
    frame = frame[frame["_gap"] != 0]
    if frame.empty:
        return empty

    frame[DIRECTION_COL] = np.where(frame["_gap"] > 0, UNDER, OVER)

    # Revenue impact of the gap, valued at list price (blank price → blank impact).
    price_map = prices if prices is not None else {}
    frame[PRICE_COL] = frame["SKU"].astype(str).map(price_map)
    frame["_impact"] = frame["_gap"] * pd.to_numeric(frame[PRICE_COL], errors="coerce")

    # Attach Data Source (POS/Orders) and Region.
    src = _recent_data_source(agg_by_group, today_ts)
    frame = frame.merge(src, on=["Customer Grouping", "SKU"], how="left")
    frame["Data Source"] = frame["Data Source"].fillna("Orders")
    frame["Region"] = frame["Customer Grouping"].map(lambda g: str(P.region_for_group(g)))

    # Description (first non-null per SKU from the aggregates).
    desc = (
        agg_by_group.dropna(subset=["Description"])
        .drop_duplicates("SKU")
        .set_index("SKU")["Description"]
    )
    frame["Description"] = frame["SKU"].map(desc)

    # Display-shaped, pre-rounded numerics (kept numeric so the table sorts right).
    out = pd.DataFrame({
        "SKU": frame["SKU"].astype(str),
        "Description": frame["Description"],
        "Customer Grouping": frame["Customer Grouping"],
        "Region": frame["Region"],
        "Data Source": frame["Data Source"],
        RECENT_COL: frame[RECENT_COL].round(1),
        PROJ_COL: frame["_proj"].round().astype("Int64"),
        GAP_COL: frame["_gap"].round().astype("Int64"),
        PCT_COL: (frame["_pct"] * 100).round(),
        IMPACT_COL: frame["_impact"].round(),
        FLAG_COL: frame[FLAG_COL],
        PRICE_COL: pd.to_numeric(frame[PRICE_COL], errors="coerce"),
        DIRECTION_COL: frame[DIRECTION_COL],
        # Sort worst-first by $ impact where known, else by unit gap magnitude.
        "_sort": frame["_impact"].abs().fillna(frame["_gap"].abs()),
    })
    return out.reset_index(drop=True)


def _apply_thresholds(frame, min_pct, min_dollar):
    """Keep only material exceptions. A row passes the % gate if its |%| meets the
    threshold OR its % is undefined ("no plan" — inherently extreme); it passes the
    $ gate if its |impact| meets the threshold OR is unknown (no list price)."""
    pct_abs = frame[PCT_COL].abs()
    pct_pass = pct_abs.isna() | (pct_abs >= min_pct * 100)
    imp_abs = frame[IMPACT_COL].abs()
    dollar_pass = imp_abs.isna() | (imp_abs >= min_dollar)
    return frame[pct_pass & dollar_pass]


def _section(frame, direction, key, P):
    """Render one direction's ranked, filterable table (worst first)."""
    sub = frame[frame[DIRECTION_COL] == direction].sort_values(
        "_sort", ascending=False
    )
    st.markdown(f"#### {direction}")
    if sub.empty:
        st.caption("No SKUs flagged in this section at the current thresholds.")
        return
    st.caption(f"{len(sub):,} SKUs flagged.")
    render_filtered_table(sub[_DISPLAY_COLS], key, P, style=True)


def render_exceptions(df, today_ts, today_str, prices, n_excluded_rows, anchors, P=None):
    """Render the EXCEPTIONS_VIEW. Mirrors _render_best_model_combined's call
    signature so main() can dispatch it the same way; the page title is already
    drawn by main(), so we start at the subheader."""
    st.subheader("Exceptions")
    st.caption(
        "SKUs whose recent actual sell-through (last 8 weeks, POS or Orders) has "
        "diverged sharply from the existing **system projection** — the plan of "
        "record, not our forecast. Under-projected = selling faster than planned "
        "(stockout risk); over-projected = planned but not selling (overstock risk)."
    )

    # Cache on a structural signature so filter/threshold reruns don't rebuild it.
    price_marker = None if prices is None else int(len(prices))
    sig = (EXCEPTIONS_VIEW_SIG, today_str, price_marker, n_excluded_rows)
    if st.session_state.get("exceptions_structural") != sig:
        with st.spinner("Scanning for exceptions…"):
            st.session_state["exceptions_frame"] = compute_exceptions(
                df, today_ts, prices, P
            )
        st.session_state["exceptions_structural"] = sig
    frame = st.session_state.get("exceptions_frame")

    if frame is None or frame.empty:
        st.info("No exceptions found — every SKU's recent sell-through tracks its projection.")
        return

    # Severity thresholds (both filters; defaults hide sub-50% moves, $ off).
    c1, c2, _ = st.columns([1, 1, 2])
    min_pct = c1.number_input(
        "Min % deviation", min_value=0, max_value=1000, value=50, step=10,
        help="Hide SKUs whose recent run-rate is within this % of the projection.",
    ) / 100.0
    min_dollar = c2.number_input(
        "Min $ impact / wk", min_value=0, max_value=1_000_000, value=0, step=100,
        help="Hide SKUs whose weekly revenue impact is below this (0 = off). "
             "SKUs with no list price are always kept.",
    )

    flagged = _apply_thresholds(frame, min_pct, min_dollar)
    total_active = frame["SKU"].nunique()
    st.caption(
        f"{flagged['SKU'].nunique():,} SKUs flagged of {total_active:,} scanned "
        f"(≥{int(min_pct * 100)}% deviation"
        + (f" and ≥${min_dollar:,}/wk impact" if min_dollar else "") + ")."
    )

    if flagged.empty:
        st.info("No exceptions at the current thresholds — try lowering them.")
        return

    _section(flagged, UNDER, "exc_under", P)
    st.divider()
    _section(flagged, OVER, "exc_over", P)
