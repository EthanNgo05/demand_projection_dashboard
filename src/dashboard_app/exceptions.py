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

import numpy as np
import pandas as pd
import streamlit as st

from dashboard_app.compute import EIGHT_WK_AVG_COL, _descriptive_averages, summary_to_excel
from dashboard_app.config import ALL_CUSTOMERS_VIEW, PRICE_COL, RISK_COL
from dashboard_app.dataquality import (
    render_discontinued_section,
    render_inactive_section,
    render_missing_pos_section,
    render_missing_section,
)
from dashboard_app.datasources import (
    compute_missing_pos_orders,
    compute_missing_projections,
    discover_key_skus_file,
    load_key_skus,
)
from dashboard_app.summaries import customer_source_map
from dashboard_app.tables import render_filtered_table

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

_DISPLAY_COLS = [
    "SKU", "Description", "Customer Grouping", "Region", "Data Source",
    RECENT_COL, PROJ_COL, GAP_COL, PCT_COL, PRICE_COL, IMPACT_COL, FLAG_COL,
]

# The Key SKUs watchlist table: the same columns as the All-Exceptions table
# (so names stay consistent across tabs) plus a Status column for the direction,
# minus the Flag column.
STATUS_COL = "Status"
KEY_DISPLAY_COLS = [
    "SKU", "Description", "Customer Grouping", "Region", STATUS_COL, "Data Source",
    RECENT_COL, PROJ_COL, PRICE_COL, IMPACT_COL, PCT_COL, FLAG_COL,
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
        GAP_COL: frame["_gap"].round().astype("Int64"),
        PCT_COL: (frame["_pct"] * 100).round(2),
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


def _section(frame, direction, key, P, today_str, slug, label, cols=None, empty_msg=None):
    """Render one direction's ranked, filterable table (worst first). ``cols``
    selects the column set (All-Exceptions vs Key SKUs); ``empty_msg`` overrides
    the placeholder caption when the section has no rows; ``slug``/``label`` name
    the download file and button."""
    cols = cols if cols is not None else _DISPLAY_COLS
    sub = frame[frame[DIRECTION_COL] == direction].sort_values(
        "_sort", ascending=False
    )
    st.markdown(f"#### {direction}")
    if sub.empty:
        st.caption(empty_msg or "No SKUs flagged in this section at the current thresholds.")
        return
    st.caption(f"{len(sub):,} SKUs flagged")
    render_filtered_table(sub[cols], key, P, style=True, column_config=_COLUMN_CONFIG)
    _download_button(sub[cols], slug, label, today_str)


def _render_all_exceptions_tab(frame, P, today_str):
    """The All-Exceptions tab: severity thresholds + Under/Over sections over the
    diverging rows (on-plan rows are excluded here)."""
    diverging = frame[frame[DIRECTION_COL] != ON_PLAN]
    if diverging.empty:
        st.info("No exceptions found — every SKU's recent sell-through tracks its projection.")
        return

    # Severity thresholds (both filters; defaults hide sub-50% moves, $ off).
    c1, c2, _ = st.columns([1, 1, 2])
    min_pct = c1.number_input(
        "Min % deviation", min_value=0, max_value=1000, value=50, step=10,
        help="Hide SKUs whose recent run-rate is within this % of the projection.",
    ) / 100.0
    min_dollar = c2.number_input(
        "Min revenue risk / wk", min_value=0, max_value=1_000_000, value=0, step=100,
        help="Hide SKUs whose weekly revenue risk is below this (0 = off). "
             "SKUs with no list price are always kept.",
    )

    flagged = _apply_thresholds(diverging, min_pct, min_dollar)
    total_active = frame["SKU"].nunique()
    st.caption(
        f"{flagged['SKU'].nunique():,} SKUs flagged of {total_active:,} scanned "
        f"(≥{int(min_pct * 100)}% deviation"
        + (f" and ≥${min_dollar:,}/wk revenue risk" if min_dollar else "") + ")."
    )

    if flagged.empty:
        st.info("No exceptions at the current thresholds — try lowering them.")
        return

    _section(flagged, UNDER, "exc_under", P, today_str,
             slug="exceptions_under-projected",
             label="Understocked Exceptions table")
    st.divider()
    _section(flagged, OVER, "exc_over", P, today_str,
             slug="exceptions_over-projected",
             label="Overstocked Exceptions table")


def _render_key_skus_tab(frame, P, today_str):
    """The Key SKUs watchlist tab: every key SKU (from extract_key_skus.py) with
    its status, no threshold filtering — a always-on watchlist of important items."""
    path = discover_key_skus_file()
    if not path:
        st.info(
            "No key-SKU list found yet. Run `python src/extract_key_skus.py` "
            "(or wait for the nightly refresh) to populate this tab."
        )
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

    key_frame[STATUS_COL] = key_frame[DIRECTION_COL].map(STATUS_SHORT).fillna(
        key_frame[DIRECTION_COL]
    )

    # Split into the two planning actions, same layout as the All-Exceptions tab.
    _section(key_frame, UNDER, "exc_key_under", P, today_str,
             slug="key_skus_under-projected", label="Understocked Key SKUs table",
             cols=KEY_DISPLAY_COLS, empty_msg="No under-projected key SKUs.")
    st.divider()
    _section(key_frame, OVER, "exc_key_over", P, today_str,
             slug="key_skus_over-projected", label="Overstocked Key SKUs table",
             cols=KEY_DISPLAY_COLS, empty_msg="No over-projected key SKUs.")

    # On-plan key SKUs belong to neither table; keep them in a collapsed section
    # so the watchlist still accounts for every key SKU.
    on_plan = key_frame[key_frame[DIRECTION_COL] == ON_PLAN].sort_values(
        "_sort", ascending=False
    )
    if not on_plan.empty:
        with st.expander(f"On-plan key SKUs ({on_plan['SKU'].nunique():,})"):
            render_filtered_table(on_plan[KEY_DISPLAY_COLS], "exc_key_onplan", P,
                                   style=True, column_config=_COLUMN_CONFIG)
            _download_button(on_plan[KEY_DISPLAY_COLS], "key_skus_on-plan",
                             "On-plan Key SKUs table", today_str)

    if missing:
        with st.expander(f"Key SKUs not in current demand data ({len(missing)})"):
            st.markdown("\n".join(f"- {s}" for s in missing))


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
                      disc_check_ran=False, discontinued_df=None):
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
    sig = (EXCEPTIONS_VIEW_SIG, today_str, price_marker, n_excluded_rows, wh_marker)
    if st.session_state.get("exceptions_structural") != sig:
        with st.spinner("Scanning for exceptions…"):
            frame = compute_exceptions(df, today_ts, prices, P)
            st.session_state["exceptions_frame"] = frame
            # Data-quality tables for the four sections moved here from Quick
            # Projections. missing_df needs the warehouse grid; missing_pos_df
            # uses full demand history. cust_source rebuilds the POS/Orders label
            # from the exceptions frame (the model summary that feeds it in the
            # single-model views isn't computed in this model-agnostic view).
            st.session_state["exceptions_missing_df"] = compute_missing_projections(
                warehouse_df, plytix_df, df, P
            )
            st.session_state["exceptions_missing_pos_df"] = compute_missing_pos_orders(
                df, plytix_df, P, anchors=anchors
            )
            st.session_state["exceptions_cust_source"] = customer_source_map(frame)
        st.session_state["exceptions_structural"] = sig
    frame = st.session_state.get("exceptions_frame")
    missing_df = st.session_state.get("exceptions_missing_df")
    missing_pos_df = st.session_state.get("exceptions_missing_pos_df")
    cust_source = st.session_state.get("exceptions_cust_source")

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

    tab_key, tab_all = st.tabs(["Key SKUs", "All Exceptions"])
    with tab_key:
        _render_key_skus_tab(frame, P, today_str)
        if key_skus:
            st.divider()
            _render_data_quality_expanders(**dq_common, key_skus=key_skus, key_suffix="_key")
    with tab_all:
        _render_all_exceptions_tab(frame, P, today_str)
        st.divider()
        _render_data_quality_expanders(**dq_common, key_skus=None, key_suffix="_all")
