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
import os
import re
from functools import partial

import numpy as np
import pandas as pd
import streamlit as st

from dashboard_app.compute import (
    EIGHT_WK_AVG_COL, _descriptive_averages, optimal_projection_for, summary_to_excel,
)
from dashboard_app.config import ALL_CUSTOMERS_VIEW, PRICE_COL, RISK_COL, fmt_dollar
from dashboard_app.dataquality import (
    render_discontinued_section,
    render_inactive_section,
    render_missing_pos_section,
    render_missing_section,
)
from dashboard_app.datasources import (
    compute_active_products,
    compute_missing_pos_orders,
    compute_missing_projections,
    container_load_from_plytix,
    discover_key_skus_file,
    load_key_skus,
    _region_code,
)
from dashboard_app.refresh import (
    key_skus_refresh_in_progress,
    start_key_skus_refresh,
)
from dashboard_app.charts import actuals_vs_plan_chart, chart_range_control
from dashboard_app.summaries import customer_source_map
from dashboard_app.tables import render_selectable_table

# Display column names for the exceptions table. These deliberately reuse the
# names style_summary already formats/colours so the table matches the other
# summary tables: RECENT_COL → 1-decimal run-rate; PROJ_COL ("Current Projection
# Average") → comma'd integer; GAP_COL ("Projection Difference") and IMPACT_COL
# ("Revenue Risk (avg/wk)") → integer/$ formatting AND green(+)/red(−) colouring.
RECENT_COL = EIGHT_WK_AVG_COL            # "8-Week POS/Orders Average"
PROJ_COL = "Current Projection Average"  # system (datawarehouse) projection, forward 15-wk avg
GAP_COL = "Projection Difference"        # 8-Week Avg − Current Projection Average
PCT_COL = "% Deviation"                  # 100 × Projection Difference / Current Projection Average
IMPACT_COL = RISK_COL                    # "Revenue Risk (avg/wk)" = Projection Difference × list price
FLAG_COL = "Note"                        # data annotation: "No forecasts given" / "No recent sales" / blank
DIRECTION_COL = "Direction"

UNDER = "Under-projected (stockout risk)"   # recent >> plan: selling faster than planned
OVER = "Over-projected (overstock risk)"    # recent << plan: planned but not selling
ON_PLAN = "On-plan"                         # recent ≈ plan (no material gap)

# Short status labels for the Key SKUs watchlist (the long section headers above
# are used to title the All-Exceptions Under/Over sections).
STATUS_SHORT = {UNDER: "Under-projected", OVER: "Over-projected", ON_PLAN: "On-plan"}

# Session-signature tag kept distinct from the view ID string so a rename of the
# user-facing label never silently reuses another view's cache entry.
EXCEPTIONS_VIEW_SIG = "exceptions-v1"

STATUS_COL = "Status"           # short direction label (Under/Over/On-plan)
WEEKS_COL = "Weeks with data"   # weeks with any POS/Orders activity for this SKU

_DISPLAY_COLS = [
    "SKU", "Description", "Customer Grouping", "Region", "Data Source", STATUS_COL,
    RECENT_COL, PROJ_COL, WEEKS_COL, GAP_COL, PCT_COL, PRICE_COL, IMPACT_COL, FLAG_COL,
]

# The Key SKUs watchlist table: the same columns as the All-Exceptions table
# (so names stay consistent across tabs) plus a Status column for the direction,
# minus the Flag column.
KEY_DISPLAY_COLS = [
    "SKU", "Description", "Customer Grouping", "Region", STATUS_COL, "Data Source",
    RECENT_COL, PROJ_COL, WEEKS_COL, PRICE_COL, IMPACT_COL, PCT_COL, FLAG_COL,
]

# The detail-card field order (both Exceptions tabs), decoupled from the frame's
# full column set above. "Note" is peeled to a full-width bottom row by the card
# renderer, so the grid renders as three rows:
#   Customer Grouping · Region · Data Source
#   Status · 8-Week Avg · Current Projection Average
#   List Price · Weeks with data
EXCEPTION_CARD_COLS = [
    "Customer Grouping", "Region", "Data Source",
    STATUS_COL, RECENT_COL, PROJ_COL,
    PRICE_COL, WEEKS_COL, FLAG_COL,
]

# Per-column widths for the exception tables. Without these, st.dataframe
# auto-sizes columns and the long free-text Description hogs width, squeezing
# the trailing Note column so its text clips with no way to expand it (Streamlit
# TextColumn can't wrap). Bounding Description and widening Note keeps both
# readable. Reused for both the All-Exceptions and Key SKUs tables.
_COLUMN_CONFIG = {
    "Description": st.column_config.TextColumn(width="medium"),
    FLAG_COL: st.column_config.TextColumn(width="medium"),
}

# The condensed row shown in the click-to-expand Exceptions tables: just the
# essentials for triage. Every other column surfaces in the detail card on click.
# The column_config only relabels the "Customer Grouping" header to "Customer" —
# the underlying column name is unchanged so the filter chips keep working.
CONDENSED_EXCEPTION_COLS = ["SKU", "Customer Grouping", RECENT_COL, PROJ_COL, IMPACT_COL]
_CONDENSED_COLUMN_CONFIG = {
    "Customer Grouping": st.column_config.Column("Customer"),
}

# ---------------------------------------------------------------------------
# "Recent spikes in POS/Orders with no projections" table
# ---------------------------------------------------------------------------
# SKU x customer combos we project 0 for that have started selling (a recent
# spike in POS or Orders). Surfaced so planners catch demand we aren't planning
# for before it turns into a stockout. Distinct from the Under/Over sections:
# those rank by deviation; this one adds the onset week + cumulative $ at risk.
SPIKE_FIRST_WEEK_COL = "First Week Spike"
SPIKE_WEEKS_SINCE_COL = "Weeks Since Spike"
# SKU-level (constant across a SKU's customer rows): Container Impact = the SKU's
# total cumulative spike units ÷ its Container Load (containers of unplanned
# demand); WOS Impact = the SKU's total On Hand ÷ its total weekly projection.
CONTAINER_IMPACT_COL = "Container Impact"
WOS_COL = "WOS Impact"

SPIKE_DISPLAY_COLS = [
    "SKU", "Description", "Region", "Region Code", "Active in", "Customer Grouping",
    "Data Source", SPIKE_FIRST_WEEK_COL, SPIKE_WEEKS_SINCE_COL,
    RECENT_COL, CONTAINER_IMPACT_COL, WOS_COL,
]

# The condensed click-to-expand row: triage essentials incl. Container Impact. WOS
# Impact and List Price move to the detail card on click.
SPIKE_CONDENSED_COLS = [
    "SKU", "Customer Grouping", SPIKE_FIRST_WEEK_COL, SPIKE_WEEKS_SINCE_COL,
    RECENT_COL, CONTAINER_IMPACT_COL,
]
# Detail-card field order (3 per row), matching the other exception cards. SKU +
# Description form the card title (handled by the card renderer):
#   Customer Grouping · Region · Data Source
#   Active in · First Week Spike · Weeks Since Spike
#   8-Week POS/Orders Average · Container Impact · WOS Impact
#   List Price (USD)
SPIKE_CARD_COLS = [
    "Customer Grouping", "Region", "Data Source",
    "Active in", SPIKE_FIRST_WEEK_COL, SPIKE_WEEKS_SINCE_COL,
    RECENT_COL, CONTAINER_IMPACT_COL, WOS_COL,
    PRICE_COL,
]
# RECENT_COL is formatted by style_summary (like the other tables); Container
# Impact / WOS get one decimal here. First Week Spike is stored as a plain date
# (.dt.date) so it renders as an ISO string under the Styler.
_SPIKE_COLUMN_CONFIG = {
    "Customer Grouping": st.column_config.Column("Customer"),
    CONTAINER_IMPACT_COL: st.column_config.NumberColumn(format="%.1f"),
    WOS_COL: st.column_config.NumberColumn(format="%.1f"),
}


# ---------------------------------------------------------------------------
# "Group by" roll-up: re-aggregate the exception/spike tables to a single
# dimension (Customer / SKU / Region). The default keeps the (SKU × Customer)
# grain the tables were built at; the other three collapse rows and sum the
# additive metrics, recomputing the derived ones (see aggregate_exceptions).
# ---------------------------------------------------------------------------
GROUP_DETAIL = "SKU × Customer"   # default — one row per SKU/customer (as built)
GROUP_SKU = "SKU"                 # roll up across customers
GROUP_CUSTOMER = "Customer"       # roll up across SKUs
GROUP_REGION = "Region"           # roll up across SKUs and customers
GROUP_BY_OPTIONS = [GROUP_DETAIL, GROUP_SKU, GROUP_CUSTOMER, GROUP_REGION]

# The frame column each non-default grain groups by.
_GROUP_KEY = {
    GROUP_SKU: "SKU",
    GROUP_CUSTOMER: "Customer Grouping",
    GROUP_REGION: "Region",
}
# The column whose value titles the detail card at each grain (see tables.py's
# render_selectable_table ``title_col``).
_TITLE_COL = {
    GROUP_DETAIL: "SKU",
    GROUP_SKU: "SKU",
    GROUP_CUSTOMER: "Customer Grouping",
    GROUP_REGION: "Region",
}


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


def sku_week_by_group(df, P):
    """Per-(Customer Grouping, SKU) weekly frame (SKU, WeekDate, POS, Orders,
    Projection, Description, Customer Grouping), with discontinued '*' SKUs dropped.

    The canonical weekly aggregation shared by ``compute_exceptions`` (for the table)
    and the detail-card charts (for the per-SKU time series), so both tie out. Uses
    only the model-agnostic ``P.aggregate_to_sku_week``. Returns an empty frame when
    there's nothing to aggregate."""
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["SKU", "WeekDate", "POS", "Orders", "Projection",
                     "Description", "Customer Grouping"]
        )
    agg_frames = []
    for group, sub in df.groupby("Customer Grouping"):
        ag = P.aggregate_to_sku_week(sub)
        ag["Customer Grouping"] = group
        agg_frames.append(ag)
    if not agg_frames:
        return pd.DataFrame(
            columns=["SKU", "WeekDate", "POS", "Orders", "Projection",
                     "Description", "Customer Grouping"]
        )
    agg_by_group = pd.concat(agg_frames, ignore_index=True)
    agg_by_group["WeekDate"] = pd.to_datetime(agg_by_group["WeekDate"])
    # Discontinued SKUs (trailing '*') are handled by the data-quality tables;
    # drop them here so they don't double-surface (also matches _descriptive_averages).
    return agg_by_group[~agg_by_group["SKU"].astype(str).str.endswith("*")]


def _render_exception_chart(agg, anchors, df, prices, today_ts, row, key_base):
    """Detail-card chart + "Calculate Optimal Projection" for one (SKU, group).

    Draws this SKU's actual sell-through vs the system projection with its own
    date-range picker (``agg`` is the stashed per-SKU-week frame). A button computes
    the model-chosen optimized 15-week forecast (fast when the group already has a
    published best model / was forecast this session; runs the 5-model backtest live
    otherwise) and overlays it on the chart. Each card gets an independent, stable
    widget key so its range and computed result persist across reruns.

    Signature matches the ``detail_chart(row, key_base)`` contract in
    ``tables.render_selectable_table`` (bind the leading args via functools.partial).
    """
    if agg is None or agg.empty:
        return
    group, sku = row.get("Customer Grouping"), row.get("SKU")
    source = row.get("Data Source") or "POS"
    ag = agg[(agg["Customer Grouping"] == group)
             & (agg["SKU"].astype(str) == str(sku))]
    if ag.empty:
        st.caption("No weekly history to chart for this SKU.")
        return
    _, lcw, ffw = anchors
    # Widen the history bound to this SKU's earliest week so the date-range picker
    # isn't trapped in the model's short (8-week) lookback (cf. kpis.py best-model).
    chart_lb = pd.to_datetime(ag["WeekDate"]).min()
    key = re.sub(r"[^0-9A-Za-z_]+", "_", f"{key_base}__chart__{sku}__{group}")

    # ----- Optimized projection (best-model 5-model backtest) --------------
    opt_key = f"{key}__opt"
    if st.button("Calculate Optimal Projection", key=f"{key}_optbtn",
                 help="Forecast this SKU with the model that wins its 5-model "
                      "backtest. Reuses the Optimized Projections view's result when "
                      "available; otherwise backtests live (can be slow)."):
        with st.spinner("Finding the best model and forecasting…"):
            st.session_state[opt_key] = optimal_projection_for(
                df, group, sku, today_ts, prices
            )
    opt = st.session_state.get(opt_key)
    weekly = None
    if opt:
        if opt["status"] == "ok":
            weekly = opt["weekly"]
            current = row.get(PROJ_COL)
            has_current = current is not None and not pd.isna(current)
            diff = opt["optimized_avg"] - current if has_current else None
            delta = None if diff is None else f"{diff:+,.0f} vs current"
            st.metric("Optimized Projection (avg/wk)",
                      f"{opt['optimized_avg']:,.0f}", delta=delta, delta_color="off")
            # Revenue impact of moving the projection to the optimized value, valued
            # at list price. Green when the change adds revenue, red when it removes it.
            price = pd.to_numeric(row.get(PRICE_COL), errors="coerce")
            if diff is not None and not pd.isna(price):
                risk = diff * price
                colour = "#16a34a" if risk >= 0 else "#dc2626"
                st.markdown(
                    f"**Revenue Risk (avg/wk):** "
                    f"<span style='color:{colour};font-weight:600'>"
                    f"{fmt_dollar(risk, decimals=0)}</span>",
                    unsafe_allow_html=True,
                )
            st.caption(f"Winning model: {opt['label']}")
        elif opt["status"] == "no_model":
            st.info("Couldn't determine a best model for this view — its history is "
                    "too short to backtest any model.")
        elif opt["status"] == "no_data":
            st.info(f"The best model ({opt.get('label')}) produced no forecast for "
                    "this SKU.")

    # The date-range picker derives the visible window's end from the horizon
    # frame's max WeekDate. Include the optimized forecast weeks so its line isn't
    # clipped away for SKUs that carry no forward SYSTEM projection (whose ag ends
    # at the last actual week, before the forecast horizon).
    horizon = ag
    if weekly is not None and not weekly.empty:
        horizon = pd.concat([ag[["WeekDate"]], weekly[["WeekDate"]]], ignore_index=True)
    date_range = chart_range_control(ag, horizon, lcw, key=key)
    fig = actuals_vs_plan_chart(sku, row.get("Description"), source, ag,
                                (chart_lb, lcw, ffw), date_range=date_range,
                                weekly=weekly)
    st.plotly_chart(fig, width="stretch", key=f"{key}_plot")


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

    # Per-group SKU-week aggregates (shared helper — the detail-card charts reuse
    # the same aggregation so their series tie out with these table numbers).
    agg_by_group = sku_week_by_group(df, P)
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
    # Round the recent run-rate and the projection to whole units/wk BEFORE the
    # derivations, so every displayed column ties out exactly: recent − projection
    # = Projection Difference, and Projection Difference × list price = Revenue Risk.
    frame[RECENT_COL] = frame[RECENT_COL].fillna(0.0).round()
    proj_missing = frame[PROJ_COL].isna()          # no plan of record at all
    # Keep the filled+rounded projection as a column so it survives the later merge
    # (which resets the index) — carrying it as a separate Series would misalign.
    frame["_proj"] = frame[PROJ_COL].fillna(0.0).round()

    frame["_gap"] = frame[RECENT_COL] - frame["_proj"]
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["_pct"] = np.where(
            frame["_proj"] != 0, frame["_gap"] / frame["_proj"], np.nan
        )

    # Flags for the two edge cases that make % undefined or degenerate.
    frame[FLAG_COL] = ""
    frame.loc[proj_missing | (frame["_proj"] == 0), FLAG_COL] = "No forecasts given"
    frame.loc[(frame[RECENT_COL] == 0) & (frame["_proj"] > 0), FLAG_COL] = "No recent sales"
    # "No recent sales" is a full over-projection: recent 0 vs a real plan = -100%.
    frame.loc[(frame[RECENT_COL] == 0) & (frame["_proj"] > 0), "_pct"] = -1.0

    # Drop rows with no signal either way (nothing planned and nothing selling).
    # On-plan rows (gap == 0) are KEPT: the All-Exceptions tab filters them out,
    # but the Key SKUs watchlist shows every key SKU including those tracking plan.
    frame = frame[(frame[RECENT_COL] != 0) | (frame["_proj"] != 0)]
    if frame.empty:
        return empty

    # Three-way status by the sign of the (rounded) gap. Rounding keeps a sub-unit
    # difference from reading as a spurious under/over.
    rounded_gap = frame["_gap"].round()
    frame[DIRECTION_COL] = np.select(
        [rounded_gap > 0, rounded_gap < 0], [UNDER, OVER], default=ON_PLAN
    )

    # Revenue impact of the gap, valued at list price (blank price → blank impact).
    price_map = prices if prices is not None else {}
    frame[PRICE_COL] = frame["SKU"].astype(str).map(price_map)
    frame["_impact"] = frame["_gap"] * pd.to_numeric(frame[PRICE_COL], errors="coerce")

    # Attach Data Source (POS/Orders) and Region.
    src = _recent_data_source(agg_by_group, today_ts)
    frame = frame.merge(src, on=["Customer Grouping", "SKU"], how="left")
    frame["Data Source"] = frame["Data Source"].fillna("Orders")
    frame["Region"] = frame["Customer Grouping"].map(lambda g: str(P.region_for_group(g)))

    # Weeks with any POS/Orders activity (a coverage signal for the detail card —
    # a genuine count of active weeks, not any one model's windowed weeks_with_data).
    active = agg_by_group.assign(
        _active=(agg_by_group[["POS", "Orders"]].fillna(0) != 0).any(axis=1)
    )
    weeks = (
        active.groupby(["Customer Grouping", "SKU"])["_active"].sum()
        .reset_index(name=WEEKS_COL)
    )
    frame = frame.merge(weeks, on=["Customer Grouping", "SKU"], how="left")

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
        RECENT_COL: frame[RECENT_COL].round().astype("Int64"),
        PROJ_COL: frame["_proj"].round().astype("Int64"),
        WEEKS_COL: pd.to_numeric(frame[WEEKS_COL], errors="coerce").fillna(0).astype("Int64"),
        GAP_COL: frame["_gap"].round().astype("Int64"),
        PCT_COL: (frame["_pct"] * 100).round(2),
        IMPACT_COL: frame["_impact"].round(),
        FLAG_COL: frame[FLAG_COL],
        PRICE_COL: pd.to_numeric(frame[PRICE_COL], errors="coerce"),
        STATUS_COL: frame[DIRECTION_COL].map(STATUS_SHORT).fillna(frame[DIRECTION_COL]),
        DIRECTION_COL: frame[DIRECTION_COL],
        # Sort worst-first by $ impact where known, else by unit gap magnitude.
        "_sort": frame["_impact"].abs().fillna(frame["_gap"].abs()),
    })
    return out.reset_index(drop=True)


def _scalar_price(price_map, sku):
    """List price for one SKU as a float (NaN when unknown/unpriced). ``price_map``
    is the SKU→price Series/dict (or None)."""
    if price_map is None:
        return np.nan
    raw = price_map.get(str(sku)) if hasattr(price_map, "get") else None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return np.nan
    return np.nan if pd.isna(val) else val


def compute_spikes(agg_by_group, today_ts, prices, P, sku_active_in=None,
                   container_load=None, onhand_by_sku=None, min_container_impact=0.0):
    """Recent-spike table: (SKU, Customer Grouping) combos we project 0 for that
    have started selling in the last 8 weeks — demand we aren't planning for yet.

    Pure/deterministic — no Streamlit — so it is unit-testable. ``agg_by_group`` is
    the per-(Customer Grouping, SKU) weekly frame from ``sku_week_by_group`` (SKU,
    WeekDate, POS, Orders, Projection, Description, Customer Grouping). A row is
    included when the forward system projection (the same 15 weeks the models use)
    is 0 or missing AND the SKU has any POS/Orders in the recent 8-week window — the
    "spike" onset is that first selling week. POS-then-Orders per SKU (POS if the SKU
    has any POS in the window, else Orders), matching the rest of the view.

    ``container_load`` (SKU→units/container) and ``onhand_by_sku`` (SKU→current total
    On Hand) drive two **SKU-level** columns constant across a SKU's customer rows:
    Container Impact = the SKU's total cumulative spike units ÷ its Container Load;
    WOS Impact = the SKU's total On Hand ÷ its total weekly projection (summed across
    all customers). Both are blank when their inputs are missing (or projection
    totals 0, to avoid ÷0). ``min_container_impact`` filters the table to SKUs whose
    Container Impact meets that many containers (SKUs without a Container Load are
    never hidden — their impact is unknown, not below-threshold).

    Returns a DataFrame with ``SPIKE_DISPLAY_COLS`` plus hidden ``PROJ_COL``
    (always 0), ``PRICE_COL``, and ``_sort`` (worst-first), one row per flagged
    (SKU, Customer Grouping).
    """
    empty = pd.DataFrame(columns=SPIKE_DISPLAY_COLS + [PROJ_COL, PRICE_COL, "_sort"])
    if agg_by_group is None or agg_by_group.empty:
        return empty

    agg = agg_by_group.copy()
    agg["WeekDate"] = pd.to_datetime(agg["WeekDate"])

    # Recent 8-week window (Sunday-anchored), same math as _recent_data_source.
    days_since_sunday = (today_ts.weekday() + 1) % 7
    current_week_start = today_ts - pd.Timedelta(days=days_since_sunday)
    last_complete_week = current_week_start - pd.Timedelta(weeks=1)
    eight_wk_start = last_complete_week - pd.Timedelta(weeks=7)

    # Candidate universe: SKUs with no forward projection (0 or missing).
    _, _, first_forecast_week = P.week_anchors(today_ts)
    forecast_weeks = pd.date_range(start=first_forecast_week, periods=15, freq="W-SUN")
    proj = _forward_projection_avg(agg, first_forecast_week, forecast_weeks[-1])
    proj_lookup = dict(
        zip(zip(proj["Customer Grouping"], proj["SKU"]), proj[PROJ_COL])
    )

    # Recent run-rate (shared helper, so the number ties out with the other tables).
    recent = _descriptive_averages(agg, today_ts)[
        ["Customer Grouping", "SKU", RECENT_COL]
    ]
    recent_lookup = dict(
        zip(zip(recent["Customer Grouping"], recent["SKU"]), recent[RECENT_COL])
    )

    # First non-null Description per SKU, for the detail-card title.
    desc_map = (
        agg.dropna(subset=["Description"]).drop_duplicates("SKU")
        .assign(SKU=lambda d: d["SKU"].astype(str))
        .set_index("SKU")["Description"]
    )

    active_map = sku_active_in or {}
    win = agg[(agg["WeekDate"] >= eight_wk_start)
              & (agg["WeekDate"] <= last_complete_week)]

    rows = []
    for (group, sku), g in win.groupby(["Customer Grouping", "SKU"]):
        # Skip anything that already carries a real forward projection.
        pv = proj_lookup.get((group, sku))
        if pv is not None and not pd.isna(pv) and round(float(pv)) != 0:
            continue

        # POS-then-Orders: POS if the SKU has any POS in the window, else Orders.
        source = "POS" if g["POS"].notna().any() else "Orders"
        wk = (g.dropna(subset=[source]).groupby("WeekDate")[source].sum())
        wk = wk[wk > 0]
        if wk.empty:
            continue

        # Spike onset = first selling week; cumulative units sold from there to now.
        first_spike = pd.Timestamp(wk.index.min())
        weeks_since = int(round((current_week_start - first_spike).days / 7))
        spike_units = float(wk.sum())

        rows.append({
            "SKU": str(sku),
            "Description": desc_map.get(str(sku)),
            "Region": str(P.region_for_group(group)),
            "Region Code": _region_code(P, group),
            "Active in": active_map.get(str(sku)),
            "Customer Grouping": group,
            "Data Source": source,
            SPIKE_FIRST_WEEK_COL: first_spike,
            SPIKE_WEEKS_SINCE_COL: weeks_since,
            RECENT_COL: recent_lookup.get((group, sku), 0.0),
            PROJ_COL: 0,
            PRICE_COL: _scalar_price(prices, sku),
            "_spike_units": spike_units,
        })

    if not rows:
        return empty
    out = pd.DataFrame(rows)

    # --- SKU-level Container Impact + WOS (constant across a SKU's rows) --------
    # Container Impact = the SKU's total cumulative spike units (summed over its
    # flagged rows) ÷ its Container Load. WOS = the SKU's total On Hand ÷ its total
    # weekly projection across ALL customers (proj is per-group; sum by SKU).
    sku_spike_units = out.groupby("SKU")["_spike_units"].sum()
    sku_total_proj = proj.groupby("SKU")[PROJ_COL].sum() if not proj.empty \
        else pd.Series(dtype="float64")
    cl_map = container_load if container_load is not None else {}
    oh_map = onhand_by_sku if onhand_by_sku is not None else {}

    def _container_impact(sku):
        load = _scalar_price(cl_map, sku)          # generic SKU→float lookup
        units = float(sku_spike_units.get(sku, 0.0))
        return units / load if not pd.isna(load) and load > 0 else np.nan

    def _wos(sku):
        onhand = _scalar_price(oh_map, sku)
        total_proj = float(sku_total_proj.get(sku, 0.0))
        return onhand / total_proj if not pd.isna(onhand) and total_proj > 0 else np.nan

    out[CONTAINER_IMPACT_COL] = out["SKU"].map(_container_impact)
    out[WOS_COL] = out["SKU"].map(_wos)

    # Filter by minimum container impact. A blank impact (SKU has no Container Load)
    # is "unknown", not "below threshold", so it's kept — mirrors _apply_thresholds.
    if min_container_impact and min_container_impact > 0:
        ci = out[CONTAINER_IMPACT_COL]
        out = out[ci.isna() | (ci >= min_container_impact)]
        if out.empty:
            return empty

    out = out.drop(columns=["_spike_units"])
    # Plain date (not datetime) so it renders as a clean ISO string under the
    # Styler and exports cleanly to Excel — matches the data-quality date columns.
    out[SPIKE_FIRST_WEEK_COL] = pd.to_datetime(out[SPIKE_FIRST_WEEK_COL]).dt.date
    out[SPIKE_WEEKS_SINCE_COL] = pd.to_numeric(
        out[SPIKE_WEEKS_SINCE_COL], errors="coerce").astype("Int64")
    out[RECENT_COL] = pd.to_numeric(out[RECENT_COL], errors="coerce").round(1)
    out[PRICE_COL] = pd.to_numeric(out[PRICE_COL], errors="coerce")
    # Worst-first: by container impact where known, else by recent run-rate.
    out["_sort"] = out[CONTAINER_IMPACT_COL].fillna(out[RECENT_COL]).abs()
    out[CONTAINER_IMPACT_COL] = out[CONTAINER_IMPACT_COL].round(1)
    out[WOS_COL] = out[WOS_COL].round(1)
    return out.sort_values("_sort", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# "Group by" roll-up helpers (pure — no Streamlit, so unit-testable).
# ---------------------------------------------------------------------------
def _plural(n, noun):
    """"1 SKU" / "2 SKUs", "1 customer" / "3 customers" — a member-count label."""
    n = int(n)
    return f"{n:,} {noun}" if n == 1 else f"{n:,} {noun}s"


def _join_regions(s):
    """Comma-separated sorted unique regions in a group (for the SKU grain, where a
    SKU can span several regions)."""
    return ", ".join(sorted({str(r) for r in s.dropna()}))


def _dim_labels(group_by, key_index, n_sku, n_cust, region_first, region_join):
    """Display values for the SKU / Customer Grouping / Region columns at a
    non-default grain. The key column holds the real group value; the collapsed
    dimensions show a member-count, and a SKU spanning regions lists them all."""
    keys = key_index.astype(str)
    if group_by == GROUP_SKU:
        sku = keys
        cust = n_cust.map(lambda x: _plural(x, "customer")).to_numpy()
        region = region_join.astype(str).to_numpy()     # every region the SKU is in
    elif group_by == GROUP_CUSTOMER:
        sku = n_sku.map(lambda x: _plural(x, "SKU")).to_numpy()
        cust = keys
        region = region_first.astype(str).to_numpy()   # one region per customer
    else:  # GROUP_REGION
        sku = n_sku.map(lambda x: _plural(x, "SKU")).to_numpy()
        cust = n_cust.map(lambda x: _plural(x, "customer")).to_numpy()
        region = keys
    return sku, cust, region


def aggregate_exceptions(frame, group_by):
    """Roll the (SKU × Customer) exceptions frame up to one row per Customer /
    SKU / Region. Returns ``frame`` unchanged for ``GROUP_DETAIL`` (or empty).

    Additive metrics (8-week average, projection, revenue risk) are summed and the
    derived columns (% deviation, direction, Note, sort key) are recomputed from the
    summed totals using the same rules as ``compute_exceptions`` — so a group's net
    gap correctly nets its under- and over-projected SKUs. List Price is kept only at
    the SKU grain (a blended price across SKUs is meaningless)."""
    if group_by == GROUP_DETAIL or frame is None or frame.empty:
        return frame
    key = _GROUP_KEY[group_by]
    g = frame.groupby(key, dropna=False, sort=False)

    recent = g[RECENT_COL].sum(min_count=1).astype("float64")
    proj = g[PROJ_COL].sum(min_count=1).astype("float64")
    impact = g[IMPACT_COL].sum(min_count=1)
    weeks = g[WEEKS_COL].max()
    n_sku = g["SKU"].nunique()
    n_cust = g["Customer Grouping"].nunique()
    region_first = g["Region"].first()
    region_join = g["Region"].agg(_join_regions)
    has_pos = g["Data Source"].agg(lambda s: (s == "POS").any())
    has_ord = g["Data Source"].agg(lambda s: (s != "POS").any())
    idx = recent.index

    recent_f = recent.fillna(0.0).to_numpy()
    proj_f = proj.fillna(0.0).to_numpy()
    gap = np.rint(recent_f - proj_f)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(proj_f != 0, gap / proj_f, np.nan)
    # A group selling nothing against a real plan is a full over-projection (-100%).
    pct = np.where((recent_f == 0) & (proj_f > 0), -1.0, pct)
    flag = np.where(proj_f == 0, "No forecasts given",
                    np.where((recent_f == 0) & (proj_f > 0), "No recent sales", ""))
    direction = np.select([gap > 0, gap < 0], [UNDER, OVER], default=ON_PLAN)
    data_source = np.where(has_pos.to_numpy() & has_ord.to_numpy(), "Mixed",
                           np.where(has_pos.to_numpy(), "POS", "Orders"))

    sku_col, cust_col, region_col = _dim_labels(
        group_by, idx, n_sku, n_cust, region_first, region_join)

    if group_by == GROUP_SKU:
        desc = (frame.dropna(subset=["Description"]).groupby(key, sort=False)["Description"]
                .first().reindex(idx))
        price = g[PRICE_COL].first()
    else:
        counts = [_plural(c, "customer") for c in n_cust.to_numpy()] \
            if group_by == GROUP_REGION else None
        skus = [_plural(s, "SKU") for s in n_sku.to_numpy()]
        desc = pd.Series(
            [f"{c}, {s}" for c, s in zip(counts, skus)] if counts else skus, index=idx)
        price = pd.Series(np.nan, index=idx, dtype="float64")

    out = pd.DataFrame({
        "SKU": pd.Series(sku_col, index=idx).astype("string"),
        "Description": pd.Series(desc, index=idx),
        "Customer Grouping": pd.Series(cust_col, index=idx),
        "Region": pd.Series(region_col, index=idx),
        "Data Source": pd.Series(data_source, index=idx),
        RECENT_COL: np.rint(recent_f).astype("int64"),
        PROJ_COL: np.rint(proj_f).astype("int64"),
        WEEKS_COL: pd.to_numeric(weeks, errors="coerce").fillna(0).astype("int64"),
        GAP_COL: gap.astype("int64"),
        PCT_COL: (pd.Series(pct, index=idx) * 100).round(2),
        IMPACT_COL: impact.round(),
        FLAG_COL: pd.Series(flag, index=idx),
        PRICE_COL: pd.to_numeric(price, errors="coerce"),
        STATUS_COL: pd.Series(direction, index=idx).map(STATUS_SHORT).fillna(
            pd.Series(direction, index=idx)),
        DIRECTION_COL: pd.Series(direction, index=idx),
        "_sort": impact.abs().fillna(pd.Series(np.abs(gap), index=idx)),
    })
    for c in (RECENT_COL, PROJ_COL, WEEKS_COL, GAP_COL):
        out[c] = out[c].astype("Int64")
    return out.reset_index(drop=True)


def aggregate_spikes(frame, group_by):
    """Roll the (SKU × Customer) spikes frame up to one row per Customer / SKU /
    Region. Returns ``frame`` unchanged for ``GROUP_DETAIL`` (or empty).

    8-week run-rate sums; First Week Spike is the earliest onset and Weeks Since
    Spike the largest. Container Impact (SKU-level) is summed over the *distinct*
    SKUs in each group — containers of unplanned demand are additive. WOS Impact is
    a per-SKU supply ratio that does not sum, so it survives only the SKU grain."""
    if group_by == GROUP_DETAIL or frame is None or frame.empty:
        return frame
    key = _GROUP_KEY[group_by]
    g = frame.groupby(key, dropna=False, sort=False)

    recent = g[RECENT_COL].sum(min_count=1)
    first_wk = pd.to_datetime(g[SPIKE_FIRST_WEEK_COL].min())
    weeks_since = g[SPIKE_WEEKS_SINCE_COL].max()
    n_sku = g["SKU"].nunique()
    n_cust = g["Customer Grouping"].nunique()
    region_first = g["Region"].first()
    region_code_first = g["Region Code"].first()
    region_join = g["Region"].agg(_join_regions)
    active_first = g["Active in"].first()
    has_pos = g["Data Source"].agg(lambda s: (s == "POS").any())
    has_ord = g["Data Source"].agg(lambda s: (s != "POS").any())
    idx = recent.index

    # Container Impact is constant per SKU; de-dup on (group, SKU) before summing so
    # a SKU spanning multiple customers/regions is counted once per group.
    subset = ["SKU"] if key == "SKU" else [key, "SKU"]
    dd = frame.drop_duplicates(subset=subset)
    container = dd.groupby(key, sort=False)[CONTAINER_IMPACT_COL].sum(min_count=1).reindex(idx)
    wos = g[WOS_COL].first() if group_by == GROUP_SKU else pd.Series(np.nan, index=idx)

    data_source = np.where(has_pos.to_numpy() & has_ord.to_numpy(), "Mixed",
                           np.where(has_pos.to_numpy(), "POS", "Orders"))
    sku_col, cust_col, region_col = _dim_labels(
        group_by, idx, n_sku, n_cust, region_first, region_join)

    if group_by == GROUP_SKU:
        desc = (frame.dropna(subset=["Description"]).groupby(key, sort=False)["Description"]
                .first().reindex(idx))
        price = g[PRICE_COL].first()
        region_code = region_code_first
        active = active_first
    else:
        skus = [_plural(s, "SKU") for s in n_sku.to_numpy()]
        counts = [_plural(c, "customer") for c in n_cust.to_numpy()] \
            if group_by == GROUP_REGION else None
        desc = pd.Series(
            [f"{c}, {s}" for c, s in zip(counts, skus)] if counts else skus, index=idx)
        price = pd.Series(np.nan, index=idx, dtype="float64")
        region_code = region_code_first   # region is single per customer; the region key otherwise
        active = pd.Series(pd.NA, index=idx)

    out = pd.DataFrame({
        "SKU": pd.Series(sku_col, index=idx).astype("string"),
        "Description": pd.Series(desc, index=idx),
        "Region": pd.Series(region_col, index=idx),
        "Region Code": pd.Series(region_code, index=idx),
        "Active in": pd.Series(active, index=idx),
        "Customer Grouping": pd.Series(cust_col, index=idx),
        "Data Source": pd.Series(data_source, index=idx),
        SPIKE_FIRST_WEEK_COL: first_wk.dt.date,
        SPIKE_WEEKS_SINCE_COL: pd.to_numeric(weeks_since, errors="coerce").astype("Int64"),
        RECENT_COL: pd.to_numeric(recent, errors="coerce").round(1),
        PROJ_COL: 0,
        PRICE_COL: pd.to_numeric(price, errors="coerce"),
        CONTAINER_IMPACT_COL: container.round(1),
        WOS_COL: pd.to_numeric(wos, errors="coerce").round(1),
    })
    out["_sort"] = out[CONTAINER_IMPACT_COL].fillna(out[RECENT_COL]).abs()
    return out.sort_values("_sort", ascending=False).reset_index(drop=True)


def _exc_cols_for(group_by):
    """(condensed_cols, column_config, title_col) for the exceptions tables at a
    grain — the leftmost condensed column is always the group key."""
    if group_by == GROUP_SKU:
        cols = ["SKU", "Region", RECENT_COL, PROJ_COL, IMPACT_COL]
        return cols, {}, _TITLE_COL[group_by]
    if group_by == GROUP_CUSTOMER:
        cols = ["Customer Grouping", "Region", RECENT_COL, PROJ_COL, IMPACT_COL]
        return cols, {"Customer Grouping": st.column_config.Column("Customer")}, _TITLE_COL[group_by]
    if group_by == GROUP_REGION:
        cols = ["Region", RECENT_COL, PROJ_COL, IMPACT_COL]
        return cols, {}, _TITLE_COL[group_by]
    return CONDENSED_EXCEPTION_COLS, _CONDENSED_COLUMN_CONFIG, _TITLE_COL[GROUP_DETAIL]


def _spike_cols_for(group_by):
    """(condensed_cols, column_config, title_col) for the spikes table at a grain."""
    tail = [SPIKE_FIRST_WEEK_COL, SPIKE_WEEKS_SINCE_COL, RECENT_COL, CONTAINER_IMPACT_COL]
    if group_by == GROUP_DETAIL:
        return SPIKE_CONDENSED_COLS, _SPIKE_COLUMN_CONFIG, _TITLE_COL[GROUP_DETAIL]
    cfg = {
        CONTAINER_IMPACT_COL: st.column_config.NumberColumn(format="%.1f"),
        WOS_COL: st.column_config.NumberColumn(format="%.1f"),
    }
    if group_by == GROUP_SKU:
        cols = ["SKU", "Region"] + tail
    elif group_by == GROUP_CUSTOMER:
        cols = ["Customer Grouping", "Region"] + tail
        cfg["Customer Grouping"] = st.column_config.Column("Customer")
    else:  # GROUP_REGION
        cols = ["Region"] + tail
    return cols, cfg, _TITLE_COL[group_by]


def _render_aggregate_chart(agg, anchors, P, group_by, row, key_base):
    """Detail-card chart for a rolled-up row: sum the weekly POS/Orders/Projection
    across the group's members and plot the same actuals-vs-plan chart. No per-SKU
    "Calculate Optimal Projection" (that is inherently a single SKU/group).

    Signature matches the ``detail_chart(row, key_base)`` contract — bind the
    leading args via functools.partial (agg, anchors, P, group_by)."""
    if agg is None or agg.empty:
        return
    if group_by == GROUP_REGION:
        target = row.get("Region")
        groups = [gr for gr in agg["Customer Grouping"].dropna().unique()
                  if str(P.region_for_group(gr)) == str(target)]
        ag = agg[agg["Customer Grouping"].isin(groups)]
    elif group_by == GROUP_CUSTOMER:
        target = row.get("Customer Grouping")
        ag = agg[agg["Customer Grouping"].astype(str) == str(target)]
    else:  # GROUP_SKU
        target = row.get("SKU")
        ag = agg[agg["SKU"].astype(str) == str(target)]
    if ag.empty:
        st.caption("No weekly history to chart for this group.")
        return

    summed = (ag.groupby("WeekDate", as_index=False)[["POS", "Orders", "Projection"]]
              .sum(min_count=1))
    title = str(target)
    summed["SKU"] = title      # actuals_vs_plan_chart filters agg by this SKU value
    source = "POS" if ag["POS"].notna().any() else "Orders"
    _, lcw, ffw = anchors
    chart_lb = pd.to_datetime(summed["WeekDate"]).min()
    key = re.sub(r"[^0-9A-Za-z_]+", "_", f"{key_base}__aggchart__{group_by}__{title}")
    date_range = chart_range_control(summed, summed, lcw, key=key)
    fig = actuals_vs_plan_chart(title, row.get("Description"), source, summed,
                                (chart_lb, lcw, ffw), date_range=date_range)
    st.plotly_chart(fig, width="stretch", key=f"{key}_plot")


def _apply_thresholds(frame, min_pct, min_dollar):
    """Keep only material exceptions. A row passes the % gate if its |%| meets the
    threshold OR its % is undefined ("No forecasts given" — inherently extreme); it
    passes the $ gate if its |impact| meets the threshold OR is unknown (no price)."""
    pct_abs = frame[PCT_COL].abs()
    pct_pass = pct_abs.isna() | (pct_abs >= min_pct * 100)
    imp_abs = frame[IMPACT_COL].abs()
    dollar_pass = imp_abs.isna() | (imp_abs >= min_dollar)
    return frame[pct_pass & dollar_pass]


def _download_button(table, slug, label, today_str):
    """Excel download of an exceptions table, matching the data-quality tables'
    download design. ``label`` is the button text (names which table it is);
    ``slug`` names both the file and the widget key (unique per section); the
    full section is exported (unfiltered, like the other views)."""
    st.download_button(
        f"⬇️ Download {label}",
        data=summary_to_excel(table, sheet_name=slug[:31]),
        file_name=f"{slug}_{today_str}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"dl_{slug}",
    )


def _section(frame, direction, key, P, today_str, slug, label, cols=None,
             empty_msg=None, chart_cb=None, condensed_cols=None, column_config=None,
             title_col="SKU"):
    """Render one direction's ranked, filterable table (worst first). ``cols``
    selects the full column set (All-Exceptions vs Key SKUs); ``empty_msg`` overrides
    the placeholder caption when the section has no rows; ``slug``/``label`` name
    the download file and button; ``chart_cb`` draws the per-row detail chart.
    ``condensed_cols``/``column_config``/``title_col`` adapt the on-screen row to the
    current "Group by" grain (default: the SKU × Customer layout)."""
    cols = cols if cols is not None else _DISPLAY_COLS
    condensed_cols = condensed_cols if condensed_cols is not None else CONDENSED_EXCEPTION_COLS
    column_config = column_config if column_config is not None else _CONDENSED_COLUMN_CONFIG
    sub = frame[frame[DIRECTION_COL] == direction].sort_values(
        "_sort", ascending=False
    )
    st.markdown(f"#### {direction}")
    if sub.empty:
        st.caption(empty_msg or "No SKUs flagged in this section at the current thresholds.")
        return
    st.caption(f"{len(sub):,} rows flagged")
    render_selectable_table(sub[cols], key, P, condensed_cols=condensed_cols,
                            style=True, column_config=column_config,
                            detail_chart=chart_cb, detail_cols=EXCEPTION_CARD_COLS,
                            title_col=title_col)
    _download_button(sub[cols], slug, label, today_str)


def _render_spikes_section(agg, prices, sku_active_in, today_ts, P, today_str,
                           key_skus=None, key_suffix="", chart_cb=None,
                           container_load=None, onhand_by_sku=None,
                           group_by=GROUP_DETAIL, anchors=None):
    """The "Recent spikes in POS/Orders with no projections" table for one tab.

    Flags SKUs we project 0 for that have started selling in the last 8 weeks, with
    the onset week and the SKU-level Container Impact / WOS. A "Minimum container
    impact" threshold (keyed by ``key_suffix`` so the two tabs' widgets stay
    independent) focuses the table on the largest unplanned demand. ``key_skus`` (or
    None) restricts the table to the watchlist. ``container_load``/``onhand_by_sku``
    drive the SKU-level Container Impact / WOS columns."""
    st.markdown("#### Recent spikes in POS/Orders with no projections")
    st.caption(
        "SKUs we currently project **0** for that have **started selling** in the "
        "last 8 weeks — demand we aren't planning for yet. The **first selling week** "
        "is the spike onset; raise the minimum container impact to focus on the "
        "largest unplanned demand."
    )
    st.markdown(
        "- **Container Impact** = units sold since the spike ÷ the SKU's Container "
        "Load (how many containers' worth of unplanned demand).\n"
        "- **WOS Impact** (Weeks of Supply) = the SKU's **total** On Hand ÷ its "
        "**total** weekly projection across all customers.\n"
        "  - Both are **SKU-level**, so every row for the same SKU shows the same "
        "value; blank when Container Load / On Hand is unavailable. (Revenue Risk of "
        "moving to the model's forecast shows in the detail card via "
        "**Calculate Optimal Projection**.)"
    )
    c1, _ = st.columns([1, 3])
    min_container_impact = c1.number_input(
        "Minimum container impact", min_value=0.0, max_value=100_000.0, value=0.2,
        step=0.5, key=f"spike_ci{key_suffix}",
        help="Hide SKUs whose total unplanned demand is below this many containers "
             "(units sold since the spike ÷ Container Load). SKUs without a Container "
             "Load are always shown.",
    )

    frame = compute_spikes(agg, today_ts, prices, P, sku_active_in,
                           container_load=container_load, onhand_by_sku=onhand_by_sku,
                           min_container_impact=min_container_impact)
    if key_skus is not None:
        frame = frame[frame["SKU"].isin(key_skus)]
    if frame.empty:
        st.caption("No SKUs projected 0 with a recent spike at the current minimum "
                   "container impact.")
        return

    # Roll up to the chosen grain (identity for the default) and pick the matching
    # condensed columns + chart callback.
    frame = aggregate_spikes(frame, group_by)
    condensed, col_cfg, title_col = _spike_cols_for(group_by)
    section_chart = chart_cb if group_by == GROUP_DETAIL else \
        partial(_render_aggregate_chart, agg, anchors, P, group_by)
    st.caption(f"{len(frame):,} rows flagged")

    # Carry the hidden PROJ_COL/PRICE_COL so the detail-card chart callback can value
    # the optimized projection; they're excluded from the shown/condensed sets.
    show_cols = SPIKE_DISPLAY_COLS + [PROJ_COL, PRICE_COL]
    render_selectable_table(
        frame[show_cols], f"exc_spikes{key_suffix}", P,
        condensed_cols=condensed, style=True,
        column_config=col_cfg,
        detail_chart=section_chart, detail_cols=SPIKE_CARD_COLS,
        title_col=title_col,
    )
    _download_button(frame[SPIKE_DISPLAY_COLS], f"spikes_no_projection{key_suffix}",
                     "Recent Spikes table", today_str)


def _render_all_exceptions_tab(frame, P, today_str, chart_cb=None,
                               agg=None, prices=None, sku_active_in=None, today_ts=None,
                               container_load=None, onhand_by_sku=None, anchors=None):
    """The All-Exceptions tab: a Group-by selector, severity thresholds, and
    Under/Over sections over the diverging rows (on-plan rows are excluded here)."""
    group_by = st.segmented_control(
        "Group by", GROUP_BY_OPTIONS, default=GROUP_DETAIL, key="grp_all",
        help="Roll every table up to one row per Customer, SKU, or Region "
             "(summing the metrics), or keep the SKU × Customer detail.",
    ) or GROUP_DETAIL
    condensed, col_cfg, title_col = _exc_cols_for(group_by)
    section_chart = chart_cb if group_by == GROUP_DETAIL else \
        partial(_render_aggregate_chart, agg, anchors, P, group_by)

    # Roll up (identity for the default grain) BEFORE splitting by direction, so a
    # group's net gap nets its under- and over-projected SKUs.
    agg_frame = aggregate_exceptions(frame, group_by)
    diverging = agg_frame[agg_frame[DIRECTION_COL] != ON_PLAN]
    if diverging.empty:
        st.info("No exceptions found — every SKU's recent sell-through tracks its projection.")
        return

    # Severity thresholds (both filters; defaults hide sub-50% moves, $ off).
    c1, c2, _ = st.columns([1, 1, 2])
    min_pct = c1.number_input(
        "Min % deviation", min_value=0, max_value=1000, value=50, step=10,
        help="Hide rows whose recent run-rate is within this % of the projection.",
    ) / 100.0
    min_dollar = c2.number_input(
        "Min revenue risk / wk", min_value=0, max_value=1_000_000, value=0, step=100,
        help="Hide rows whose weekly revenue risk is below this (0 = off). "
             "Rows with no list price are always kept.",
    )

    flagged = _apply_thresholds(diverging, min_pct, min_dollar)
    st.caption(
        f"{len(flagged):,} rows flagged of {len(agg_frame):,} scanned "
        f"(≥{int(min_pct * 100)}% deviation"
        + (f" and ≥${min_dollar:,}/wk revenue risk" if min_dollar else "") + ")."
    )

    if flagged.empty:
        st.info("No exceptions at the current thresholds — try lowering them.")
        return

    _section(flagged, UNDER, "exc_under", P, today_str,
             slug="exceptions_under-projected",
             label="Understocked Exceptions table", chart_cb=section_chart,
             condensed_cols=condensed, column_config=col_cfg, title_col=title_col)
    st.divider()
    _section(flagged, OVER, "exc_over", P, today_str,
             slug="exceptions_over-projected",
             label="Overstocked Exceptions table", chart_cb=section_chart,
             condensed_cols=condensed, column_config=col_cfg, title_col=title_col)
    st.divider()
    _render_spikes_section(agg, prices, sku_active_in, today_ts, P, today_str,
                           key_skus=None, key_suffix="_all", chart_cb=chart_cb,
                           container_load=container_load, onhand_by_sku=onhand_by_sku,
                           group_by=group_by, anchors=anchors)


def _render_key_skus_fetch_prompt():
    """Empty-state prompt for the Key SKUs tab: a button that pulls the key-SKU
    list from the data warehouse in the background (extract_key_skus.py), so
    planners never have to run a terminal command. Mirrors the demand refresh
    button's running/idle states — once the pull's file lands, the tab
    re-discovers it and renders the watchlist on the next run."""
    running, started = key_skus_refresh_in_progress()
    if running:
        st.info(
            f"Fetching the key-SKU list from the data warehouse… "
            f"(started {started}). This usually takes under a minute."
        )
        if st.button("Check for the list", key="key_skus_check"):
            st.rerun()
        return

    st.info(
        "No key-SKU list yet. Fetch the current list of key items from the "
        "data warehouse to populate this watchlist."
    )
    if st.button("🔄 Fetch key-SKU list", key="key_skus_fetch"):
        ok, msg = start_key_skus_refresh()
        if ok:
            st.success("Fetching the key-SKU list — running in the background.")
            st.rerun()
        else:
            st.warning(msg)


def _render_key_skus_tab(frame, P, today_str, chart_cb=None,
                         agg=None, prices=None, sku_active_in=None, today_ts=None,
                         container_load=None, onhand_by_sku=None, anchors=None):
    """The Key SKUs watchlist tab: every key SKU (from extract_key_skus.py) with
    its status, no threshold filtering — a always-on watchlist of important items."""
    path = discover_key_skus_file()
    if not path:
        _render_key_skus_fetch_prompt()
        return
    key_skus = load_key_skus(path, os.path.getmtime(path))
    if not key_skus:
        st.info("The key-SKU list is empty.")
        return

    key_frame = frame[frame["SKU"].isin(key_skus)].copy()
    present = set(key_frame["SKU"])
    missing = sorted(key_skus - present)
    st.caption(
        f"Showing all {len(present):,} of {len(key_skus):,} key SKUs present in the "
        f"current demand data"
        + (f" ({len(missing):,} not found)" if missing else "") + "."
    )
    if key_frame.empty:
        st.info("None of the key SKUs appear in the current demand data.")
        return

    group_by = st.segmented_control(
        "Group by", GROUP_BY_OPTIONS, default=GROUP_DETAIL, key="grp_key",
        help="Roll every table up to one row per Customer, SKU, or Region "
             "(summing the metrics), or keep the SKU × Customer detail.",
    ) or GROUP_DETAIL
    condensed, col_cfg, title_col = _exc_cols_for(group_by)
    section_chart = chart_cb if group_by == GROUP_DETAIL else \
        partial(_render_aggregate_chart, agg, anchors, P, group_by)
    # Status is set centrally in compute_exceptions; roll the key-SKU frame up to
    # the chosen grain (identity for the default) before splitting by direction.
    key_frame = aggregate_exceptions(key_frame, group_by)

    # Split into the two planning actions, same layout as the All-Exceptions tab.
    _section(key_frame, UNDER, "exc_key_under", P, today_str,
             slug="key_skus_under-projected", label="Understocked Key SKUs table",
             cols=KEY_DISPLAY_COLS, empty_msg="No under-projected key SKUs.",
             chart_cb=section_chart, condensed_cols=condensed, column_config=col_cfg,
             title_col=title_col)
    st.divider()
    _section(key_frame, OVER, "exc_key_over", P, today_str,
             slug="key_skus_over-projected", label="Overstocked Key SKUs table",
             cols=KEY_DISPLAY_COLS, empty_msg="No over-projected key SKUs.",
             chart_cb=section_chart, condensed_cols=condensed, column_config=col_cfg,
             title_col=title_col)

    # On-plan key SKUs belong to neither table; keep them in a collapsed section
    # so the watchlist still accounts for every key SKU.
    on_plan = key_frame[key_frame[DIRECTION_COL] == ON_PLAN].sort_values(
        "_sort", ascending=False
    )
    if not on_plan.empty:
        with st.expander(f"On-plan key SKUs ({len(on_plan):,})"):
            render_selectable_table(on_plan[KEY_DISPLAY_COLS], "exc_key_onplan", P,
                                    condensed_cols=condensed,
                                    style=True, column_config=col_cfg,
                                    detail_chart=section_chart, detail_cols=EXCEPTION_CARD_COLS,
                                    title_col=title_col)
            _download_button(on_plan[KEY_DISPLAY_COLS], "key_skus_on-plan",
                             "On-plan Key SKUs table", today_str)

    if missing:
        with st.expander(f"Key SKUs not in current demand data ({len(missing)})"):
            st.markdown("\n".join(f"- {s}" for s in missing))

    st.divider()
    _render_spikes_section(agg, prices, sku_active_in, today_ts, P, today_str,
                           key_skus=key_skus, key_suffix="_key", chart_cb=chart_cb,
                           container_load=container_load, onhand_by_sku=onhand_by_sku,
                           group_by=group_by, anchors=anchors)


def _render_data_quality_expanders(
    *, view, region, today_str, P,
    warehouse_df, check_ran, inactive_df, excluded_counts_by_key, n_excluded_rows,
    disc_check_ran, discontinued_df, missing_df, missing_pos_df, cust_source,
    key_skus, key_suffix,
):
    """Render the four data-quality sections, each in a collapsed expander, for one
    Exceptions tab. ``key_skus`` (or None) filters every section to key SKUs;
    ``key_suffix`` keeps each tab's widget keys unique. The section renderers draw
    their own titles inside via ``show_header=False`` being unset — here we suppress
    the ``###`` header since the expander label carries the title."""
    with st.expander("SKUs with forecasts in locations they are not active in"):
        render_inactive_section(
            view, region, check_ran, inactive_df,
            excluded_counts_by_key, n_excluded_rows, today_str,
            key_skus=key_skus, key_suffix=key_suffix, show_header=False,
        )
    with st.expander("SKUs missing forecasts in locations they are active in"):
        render_missing_section(
            view, region, warehouse_df, check_ran, missing_df, today_str,
            cust_source, P,
            key_skus=key_skus, key_suffix=key_suffix, show_header=False,
        )
    with st.expander("SKUs missing POS/Orders data in locations they are active in"):
        render_missing_pos_section(
            view, region, missing_pos_df, today_str,
            key_skus=key_skus, key_suffix=key_suffix, show_header=False,
        )
    with st.expander("Inactive/discontinued SKUs with forecasts"):
        render_discontinued_section(
            view, region, disc_check_ran, discontinued_df, today_str,
            key_skus=key_skus, key_suffix=key_suffix, show_header=False,
        )


def render_exceptions(df, today_ts, today_str, prices, n_excluded_rows, anchors, P=None,
                      *, warehouse_df=None, plytix_df=None, check_ran=False,
                      inactive_df=None, excluded_counts_by_key=None,
                      disc_check_ran=False, discontinued_df=None,
                      allocation_pairs=None, onhand_by_sku=None):
    """Render the EXCEPTIONS_VIEW. Mirrors _render_best_model_combined's call
    signature so main() can dispatch it the same way; the page title is already
    drawn by main(), so we start at the subheader.

    The keyword-only args carry the inputs for the four data-quality sections
    (moved here from Quick Projections): they render below each tab's Under/Over
    tables — all rows in All Exceptions, key SKUs only in the Key SKUs tab."""
    st.subheader("Exceptions")
    st.caption(
        "SKUs where recent sales no longer match the plan. We compare each SKU's "
        "actual sell-through over the **last 8 weeks** (POS, or Orders where POS "
        "isn't available) against its **current system projection** — the official "
        "plan of record, not our model's forecast."
    )
    st.markdown(
        "- **Under-projected** — selling faster than planned → **stockout risk**\n"
        "- **Over-projected** — planned but not selling → **overstock risk**"
    )
    st.caption("How each column is calculated:")
    st.markdown(
        "- **Projection Difference** = (8-Week POS/Orders Average) − (Current Projection Average)\n"
        "- **% Deviation** = (Projection Difference / Current Projection Average) × 100\n"
        "  - Blank when there is no projection\n"
        "- **Revenue Risk (avg/wk)** = Projection Difference × List Price"
    )

    # Cache on a structural signature so filter/threshold reruns don't rebuild the
    # exceptions frame OR the (relatively expensive) data-quality tables. The
    # warehouse marker invalidates the cache when the warehouse grid is re-uploaded.
    price_marker = None if prices is None else int(len(prices))
    wh_marker = None if warehouse_df is None else int(len(warehouse_df))
    alloc_marker = None if allocation_pairs is None else len(allocation_pairs)
    sig = (EXCEPTIONS_VIEW_SIG, today_str, price_marker, n_excluded_rows,
           wh_marker, alloc_marker)
    if st.session_state.get("exceptions_structural") != sig:
        with st.spinner("Scanning for exceptions…"):
            frame = compute_exceptions(df, today_ts, prices, P)
            st.session_state["exceptions_frame"] = frame
            # Per-SKU-week frame for the detail-card charts (same aggregation the
            # table uses, so the plotted series tie out with the row's numbers).
            st.session_state["exceptions_agg"] = sku_week_by_group(df, P)
            # Data-quality tables for the four sections moved here from Quick
            # Projections. missing_df needs the warehouse grid; missing_pos_df
            # uses full demand history. cust_source rebuilds the POS/Orders label
            # from the exceptions frame (the model summary that feeds it in the
            # single-model views isn't computed in this model-agnostic view).
            st.session_state["exceptions_missing_df"] = compute_missing_projections(
                warehouse_df, plytix_df, df, P, allocation_pairs
            )
            st.session_state["exceptions_missing_pos_df"] = compute_missing_pos_orders(
                df, plytix_df, P, anchors=anchors
            )
            st.session_state["exceptions_cust_source"] = customer_source_map(frame)
            # SKU -> "Active in" regions (Plytix) for the spikes table; None when the
            # export lacks the columns (older list-price file) — the column stays blank.
            st.session_state["exceptions_active_in"] = compute_active_products(plytix_df)[1]
            # SKU -> Container Load (Plytix) for the spikes table's Container Impact.
            st.session_state["exceptions_container_load"] = container_load_from_plytix(plytix_df)
        st.session_state["exceptions_structural"] = sig
    frame = st.session_state.get("exceptions_frame")
    missing_df = st.session_state.get("exceptions_missing_df")
    missing_pos_df = st.session_state.get("exceptions_missing_pos_df")
    cust_source = st.session_state.get("exceptions_cust_source")
    sku_active_in = st.session_state.get("exceptions_active_in")
    container_load = st.session_state.get("exceptions_container_load")
    agg = st.session_state.get("exceptions_agg")

    if frame is None or frame.empty:
        st.info("No exceptions found — every SKU's recent sell-through tracks its projection.")
        return

    # The four data-quality sections render globally (all regions) in this view.
    dq_common = dict(
        view=ALL_CUSTOMERS_VIEW, region=None, today_str=today_str, P=P,
        warehouse_df=warehouse_df, check_ran=check_ran, inactive_df=inactive_df,
        excluded_counts_by_key=excluded_counts_by_key, n_excluded_rows=n_excluded_rows,
        disc_check_ran=disc_check_ran, discontinued_df=discontinued_df,
        missing_df=missing_df, missing_pos_df=missing_pos_df, cust_source=cust_source,
    )

    # Key SKUs to filter the Key-SKUs-tab sections by (None → sections skipped).
    path = discover_key_skus_file()
    key_skus = load_key_skus(path, os.path.getmtime(path)) if path else None

    # Per-row detail-card chart + "Calculate Optimal Projection". Binds the stashed
    # per-SKU-week frame, anchors, and the cleaned df / prices / today (needed for the
    # optimized forecast); called as (row, key_base) per the detail_chart contract.
    chart_cb = partial(_render_exception_chart, st.session_state.get("exceptions_agg"),
                       anchors, df, prices, today_ts)

    spike_kw = dict(agg=agg, prices=prices, sku_active_in=sku_active_in, today_ts=today_ts,
                    container_load=container_load, onhand_by_sku=onhand_by_sku,
                    anchors=anchors)
    tab_key, tab_all = st.tabs(["Key SKUs", "All Exceptions"])
    with tab_key:
        _render_key_skus_tab(frame, P, today_str, chart_cb=chart_cb, **spike_kw)
        if key_skus:
            st.divider()
            _render_data_quality_expanders(**dq_common, key_skus=key_skus, key_suffix="_key")
    with tab_all:
        _render_all_exceptions_tab(frame, P, today_str, chart_cb=chart_cb, **spike_kw)
        st.divider()
        _render_data_quality_expanders(**dq_common, key_skus=None, key_suffix="_all")
