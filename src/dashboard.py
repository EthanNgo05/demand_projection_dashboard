"""
Demand Projection Dashboard
===========================

An interactive Streamlit + Plotly front-end for the 15-week demand forecasts
produced by the pipeline files in ``models/`` (regression, exponential
smoothing, XGBoost).

Rather than reading the saved Excel files (which only contain the 15 forecast
weeks), this dashboard reads the *same raw data file* the pipeline uses and
recomputes the forecast live by importing the pipeline's own functions
(``week_anchors``, ``aggregate_to_sku_week``, ``fit_regression``,
``region_for_group``). That keeps a single source of truth for the forecasting
logic and unlocks the 8 weeks of historical actuals so they can be charted
flowing straight into the forecast.

Each SKU is forecast from POS where it has any in the 8-week window, otherwise
from the Orders signal (the pipeline's POS-then-Orders fallback). The dashboard
mirrors that: the historical line for a SKU shows whichever signal drove its
forecast, and the "Data Source" is surfaced throughout.

Run it locally with two terminals:

    Terminal 1: streamlit run dashboard.py --server.headless true
    Terminal 2: ngrok http 12589

    Use link like: https://reissue-ninetieth-deeply.ngrok-free.dev 

Also hosted on Streamlit Community Cloud

    https://sh-demand-projections.streamlit.app/ 

By default it discovers the raw folder from the pipeline's own
``RAW_INPUTS_FOLDER`` (currently ``raw_inputs/demand_projections``), resolved next
to this file. Override paths with the DEMAND_PIPELINE / DEMAND_RAW_DIR env vars.
"""

import os
import re
import sys
import time
import base64
import datetime
import functools
import glob
import html
import json
import inspect
import logging
import tempfile
import threading
import subprocess
import traceback
import importlib.util
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Shared, Streamlit-free I/O (Phase 2 of the agentic workflow): file discovery
# and raw-frame cleaning live in agent/data_io.py so the dashboard and the
# LangGraph agent share one source of truth. The @st.cache_data wrappers below
# stay here, wrapping thin calls into the shared module.
from agent import data_io

# Date-organized logging (logs/<date>/...), Streamlit-free so the agent can
# share it. See log_config.py.
from log_config import DateFolderHandler, dated_log_path

# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
# Developer-facing log, organized by day under ``logs/<date>/app.log`` at the
# repo root so issues can be inspected after the fact (on Streamlit Cloud, also
# visible via Manage app → logs). Configured once per process; Streamlit reruns
# import the module only once, so the handler isn't attached repeatedly. The
# DateFolderHandler rolls to a new day's folder on its own, so a dashboard left
# running for days still files each line under the date it was written.
LOG_FILENAME = "app.log"

logger = logging.getLogger("demand_dashboard")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    # File output is best-effort (read-only hosts): the handler swallows OSError
    # internally, and the StreamHandler below still logs to the console.
    _fh = DateFolderHandler(LOG_FILENAME)
    _fh.setFormatter(_fmt)
    logger.addHandler(_fh)
    _sh = logging.StreamHandler()
    _sh.setFormatter(_fmt)
    logger.addHandler(_sh)
    logger.propagate = False

# --------------------------------------------------------------------------- #
# Facade re-exports: keep every helper reachable as dashboard.<name> so the   #
# tests and main()/_run() below resolve them unchanged. Implementation lives  #
# in the dashboard_app/ package; this file stays the Streamlit entrypoint.    #
# --------------------------------------------------------------------------- #
from dashboard_app.config import (  # noqa: F401
    ALL_CUSTOMERS_VIEW, ALL_REGIONS, BEST_MODEL_COMBINED_VIEW, C_ACTUAL, C_GRID,
    C_ORIGINAL, C_UPDATED,
    DEFAULT_MODEL, EXCEPTIONS_VIEW, HERE, HISTORICAL_VIEW, KPI_HELP, LOGO_PATH,
    MODEL_DISPLAY, MODEL_OPTIONS, MODEL_USED_COL,
    PRICE_COL, QUICK_VIEW, REGION_ALL_PREFIX, REPO_ROOT, RISK_COL, SCOPE_CAPTIONS,
    SCOPE_LABELS, WATCHLIST_VIEW,
    _ENV_PIPELINE,
    bounded_put, fmt_dollar, fmt_when, model_display, quick_group_label,
    region_all_view, region_from_view,
)
from dashboard_app.pipeline import (  # noqa: F401
    _load_pipeline_cached, _supports_autofit, _supports_min_weeks, _supports_prices,
    _supports_smoothing, load_pipeline, pipeline_path,
)
from dashboard_app.summaries import (  # noqa: F401
    _format_generated_at, avg_window_phrase, historical_window,
    historical_window_label, resolve_avg_col, resolve_demand, source_map,
)
from dashboard_app.charts import (  # noqa: F401
    _base_layout, _clip_to_range, aggregate_chart, chart_range_control,
    customer_share_donut, sku_chart,
)
from dashboard_app.tables import (  # noqa: F401
    FIXED_FILTER_LABELS, render_filtered_table, render_selectable_table,
    style_summary,
)
from dashboard_app.keyskus import (  # noqa: F401
    current_key_skus, key_sku_mask, mark_key_sku, with_key_sku_column,
    sku_chip_column_config,
)
from dashboard_app.datasources import (  # noqa: F401
    DISCONTINUED_COLS, INACTIVE_COLS, MISSING_COLS, MISSING_POS_COLS, WAREHOUSE_REGIONS,
    _active_in_list, _clean, _date_from_name, _raw_dir, _region_code, _this_week_start,
    compute_active_products, compute_discontinued_products,
    compute_discontinued_projections, compute_inactive_projections,
    discover_key_skus_file,
    discover_price_file, discover_raw_files, fetch_plytix_from_url, load_key_skus,
    load_allocation_pairs_from_bytes, load_allocation_pairs_from_path,
    load_onhand_by_sku_from_bytes, load_onhand_by_sku_from_path,
    load_prices_from_bytes, load_prices_from_path, load_raw_from_bytes,
    load_raw_from_path, load_warehouse_from_paths, load_warehouse_from_uploads,
    price_glob, raw_glob, read_plytix_from_bytes, read_plytix_from_path,
)
from dashboard_app.compute import (  # noqa: F401
    ALL_TIME_AVG_COL, EIGHT_WK_AVG_COL, ONHAND_COL, TREND_COL, WOS_COL,
    _agent_summaries_generated_at, _agent_summaries_mtime, _agent_summary_path,
    _best_model_for_group, _forecast_one_group, _load_agent_summary, _region_frame,
    attach_current_projection, attach_descriptive_averages, attach_supply_columns,
    attach_top_volume, compute_by_customer, compute_by_customer_best,
    compute_by_customer_frames, compute_view, list_views, roll_up_summary,
    roll_up_to_sku_week, run_autofit, single_group_frames, sku_grain_demand_frame,
    summary_to_excel, view_to_excel, with_export_flags,
)
from dashboard_app.refresh import (  # noqa: F401
    BATCH_STALE_SECONDS, EXTRACT_SCRIPT, REFRESH_STALE_SECONDS, WAREHOUSE_EXTRACT_SCRIPT,
    _batch_lock_path, _batch_log_path, _batch_result_line, _clear_lock, _launch_refresh,
    _refresh_lock_path, _refresh_log_path, _refresh_state, _wh_refresh_lock_path,
    _wh_snapshot_complete_since, batch_elapsed_suffix, batch_failures,
    batch_in_progress, batch_progress,
    batch_result_message, clear_sync_failures, refresh_in_progress,
    start_agent_batch, start_key_skus_refresh, start_refresh,
    start_warehouse_refresh, sync_failures, warehouse_refresh_in_progress,
)
from dashboard_app.agent_summary import (  # noqa: F401
    LLM_PROVIDERS, _AGENT_NODE_PROGRESS, _agent_progress_fragment, _agent_scores,
    _confirm_run_all_dialog, _model_fit_callout, _render_agent_summary, _run_agent_job,
)
from dashboard_app.kpis import (  # noqa: F401
    BEST_MIX_CARD_COLS, BEST_MIX_CONDENSED_COLS,
    _render_best_model_combined, _render_kpis, render_sku_detail_card,
    render_sku_detail_section, projection_difference_delta, projection_kpi_extras,
)
from dashboard_app.exceptions import (  # noqa: F401
    compute_exceptions, render_exceptions,
)
from dashboard_app import forecast_cache  # noqa: F401
from dashboard_app.watchlist_view import render_watchlist  # noqa: F401
from dashboard_app.historical_summary import render_historical_summary  # noqa: F401


# Bounds for the per-view session caches below. Entries are keyed per
# (view, model, snapshot, …); the caps keep memory flat across a long session of
# view-hopping while leaving plenty of headroom for a normal set of visited
# views. Forecast results carry DataFrames (kept smaller), autofit params are a
# few floats (kept larger).
#
# FC_CACHE_MAX is deliberately small. Each entry now holds the per-group SKU-week
# aggregate that the Customer-detail chart and the summary table's detail cards
# draw from — ~500k rows / ~34 MB for the all-customers scope — so 16 slots would
# reach ~700 MB of session_state per browser session. Four keeps it near the
# ~100 MB the 4-frame tuples used to cost. An eviction is cheap: the fits
# themselves stay memoised in _forecast_one_group (in process AND on disk), so a
# revisited view pays only ~1.5s of re-stitching, never a refit. (If four proves
# tight, the per-group aggregate is model- and parameter-independent, so it could
# move to its own small cache with these tuples holding a shared reference.)
FC_CACHE_MAX = 4
AUTOFIT_CACHE_MAX = 64

# Quick Projections' main table mirrors Optimized Projections: the same five
# scannable columns per row, with everything else one click away in the detail
# card. There is no "Model Used" column here (one chosen model fits the whole
# view, named in the Forecasting model selector).
QUICK_CONDENSED_COLS = ["SKU", "Customer Grouping", EIGHT_WK_AVG_COL,
                        "Current Projection Average", RISK_COL]
# The card's KPI tiles. A SET, not a sequence — config.kpi_sort orders them, so a
# field sits in the same place as on Optimized / Exceptions / Watchlist cards.
# Mirrors kpis.BEST_MIX_CARD_COLS minus "Model Used" (see above). The forecast and
# money fields used to be st.metric calls in a column beside the chart inside
# render_sku_detail_card; they are tiles here now, so the card has ONE KPI zone
# instead of two — and "Data Source" appears once instead of in both.
QUICK_CARD_COLS = [
    "Customer Grouping", "Data Source",
    "Weeks with data", ALL_TIME_AVG_COL, EIGHT_WK_AVG_COL, TREND_COL,
    "Current Projection Average", "Updated Projection Average",
    "Projection Difference",
    PRICE_COL, RISK_COL,
    ONHAND_COL, WOS_COL,
]


def _coverage_note(src, by_cust, agg_by_group, horizon):
    """What the view's totals do NOT cover, as counts — or None when they cover all.

    Now that every figure on the page is the sum of the per-(SKU, customer) rows, a
    row that never got forecast is a unit of demand missing from the total. The
    per-group loop skips such groups silently (``compute._by_customer_frames``), which
    is precisely how a wrong total hides, so the three ways coverage can fall short
    are counted here and reported on the page:

    * ``groups`` — customer groups in this view that produced no forecast at all
      (no POS and no Orders anywhere in the fit window).
    * ``ungrouped_rows`` — rows whose Customer Grouping is blank. The per-group loop
      enumerates non-null groups only, so these are outside the totals entirely.
    * ``planned_pairs`` / ``planned_units`` — (SKU, customer) pairs carrying a forward
      plan but no forecast, and the weekly units that plan represents. These are
      genuine over-projection candidates rather than a defect in the total; the
      Exceptions view is where they are meant to be actioned.
    """
    note = {}
    if src is not None and "Customer Grouping" in src.columns:
        offered = set(src["Customer Grouping"].dropna().astype(str))
        covered = set(by_cust["Customer Grouping"].astype(str))
        note["groups"] = sorted(offered - covered)
        note["ungrouped_rows"] = int(src["Customer Grouping"].isna().sum())
    if agg_by_group is not None and "Projection" in agg_by_group.columns:
        horizon = pd.to_datetime(pd.Series(list(horizon))).unique()
        fwd = agg_by_group[pd.to_datetime(agg_by_group["WeekDate"]).isin(horizon)]
        planned = (
            pd.to_numeric(fwd["Projection"], errors="coerce")
            .groupby([fwd["Customer Grouping"].astype(str), fwd["SKU"].astype(str)])
            .sum(min_count=1)
        )
        planned = planned[planned.notna() & (planned != 0)]
        have = set(zip(by_cust["Customer Grouping"].astype(str),
                       by_cust["SKU"].astype(str)))
        orphan = planned[[k not in have for k in planned.index]]
        note["planned_pairs"] = int(len(orphan))
        note["planned_units"] = float(orphan.sum()) / max(len(horizon), 1)
    if not any(note.get(k) for k in
               ("groups", "ungrouped_rows", "planned_pairs")):
        return None
    return note


def _render_coverage_note(coverage):
    """Render ``_coverage_note``'s counts as one caption under the KPI row."""
    if not coverage:
        return
    bits = []
    groups = coverage.get("groups") or []
    if groups:
        shown = ", ".join(groups[:4])
        more = f" +{len(groups) - 4} more" if len(groups) > 4 else ""
        bits.append(
            f"**{len(groups)}** customer group(s) had no POS or Orders in the window "
            f"and are not in these totals ({shown}{more})."
        )
    if coverage.get("ungrouped_rows"):
        bits.append(
            f"**{coverage['ungrouped_rows']:,}** rows have no Customer Grouping and "
            "are outside every per-customer forecast."
        )
    if coverage.get("planned_pairs"):
        bits.append(
            f"**{coverage['planned_pairs']:,}** SKU-customer pairs carry a forward "
            f"plan of ~{coverage['planned_units']:,.0f} units/wk but no recent demand "
            "to forecast from, so they are not in the Updated figures — see "
            "Exceptions for those."
        )
    if bits:
        st.caption("⚑ " + " ".join(bits))


@st.cache_data(show_spinner=False)
def _logo_data_uri():
    """Return the simplehuman logo as a ``data:`` URI for inline <img> use, or
    None if the bundled asset is missing/unreadable (the header then degrades to
    a plain title). Cached so the small file is read+encoded once per session."""
    try:
        with open(LOGO_PATH, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except OSError:
        return None


# Moved to dashboard_app.config so exceptions.py can bound its own session cache
# with the same semantics; kept as a module-level alias because dashboard.py is a
# facade and callers/tests resolve it as dashboard._bounded_put.
_bounded_put = bounded_put


# The by-customer table used to carry a SKU dropdown ABOVE its filter chips, which
# needed two pieces of machinery that are now gone: a callback to clear the table's
# positional row selection (tables._clear_row_selection does that for every control
# on the fixed filter bar), and a one-shot scroll-into-view (the app's only
# JavaScript). The scroll existed because the picker sat outside the table's
# fragment, far enough above the results that a pick left the reader looking at the
# wrong part of the page. The bar renders INSIDE the fragment, directly above the
# rows it narrows, and a fragment rerun does not move the page — so there is nothing
# left to scroll to.


def main():
    # Browser-tab favicon: the simplehuman logo when the bundled asset is present,
    # otherwise a plain fallback so a missing file never breaks app startup.
    page_icon = LOGO_PATH if os.path.exists(LOGO_PATH) else "◾"
    st.set_page_config(
        page_title="Demand Projections", page_icon=page_icon, layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Keep the reasoning-LLM choice across reruns where its radio isn't
    # re-rendered — a refresh button st.rerun()s before the script reaches the
    # radio, and the Exceptions view never renders it at all. Streamlit
    # garbage-collects an unrendered keyed widget's state, which would snap the
    # radio back to its first option (Anthropic) and spuriously surface the
    # "No ANTHROPIC_API_KEY" warning even when the user picked Local.
    # Re-registering the key here preserves the actual selection.
    if "agent_llm_provider" in st.session_state:
        st.session_state["agent_llm_provider"] = st.session_state["agent_llm_provider"]

    st.markdown(
        f"""
        <style>
        /* Render the top-of-page view segmented control as a tab strip: the four
           options read as left-aligned folder-style tabs sitting on a shared
           baseline, with the active tab marked by a blue underline + bold label.
           Scoped to the segmented-control widget (stButtonGroup) so ordinary
           buttons and st.columns button rows are unaffected. Harmless no-op if a
           future Streamlit renames the test id (the active/inactive suffixes are
           stBaseButton-segmented_control[Active]). */
        div[data-testid="stButtonGroup"] {{
            width: 100%;
            border-bottom: 1px solid rgba(148,163,184,0.35);  /* shared tab baseline */
            margin-bottom: 0.75rem;
        }}
        div[data-testid="stButtonGroup"] > div {{ display: flex; gap: 0.25rem; }}
        div[data-testid="stButtonGroup"] button {{
            background: transparent !important;
            border: none !important;
            border-bottom: 2px solid transparent !important;  /* reserve underline space */
            border-radius: 0 !important;
            margin-bottom: -1px;              /* overlap the baseline */
            padding: 0.4rem 1rem;
            color: rgba(148,163,184,1);       /* muted inactive label */
            font-weight: 500;
            font-size: 1.15rem;               /* larger, more prominent tab titles */
        }}
        /* The button label text inherits size from the button, but Streamlit
           wraps it in a <p>/markdown span with its own size — bump that too so
           the enlarged font actually takes effect. */
        div[data-testid="stButtonGroup"] button p {{
            font-size: 1.15rem;
        }}
        div[data-testid="stButtonGroup"] button:hover {{
            color: inherit !important;
            border-bottom-color: rgba(148,163,184,0.5) !important;
        }}
        /* Active tab: accent underline + emphasis. Follows the theme's primaryColor
           (graphite in light, near-white in dark) via the CSS variable so it stays
           on-brand in both modes and matches every other accent in the app. */
        div[data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_controlActive"] {{
            color: inherit !important;
            font-weight: 700;
            border-bottom-color: var(--primary-color, #000000) !important;
        }}

        /* Replace Streamlit's top-right "running" status graphic — which cycles
           through animated sport figures (runner, cyclist, swimmer…) — with a
           plain spinning loader. We hide the icon wrapper's contents and draw a
           CSS spinner in its place; the "Running..." text and Stop button are
           separate elements and stay intact. */
        [data-testid="stStatusWidgetRunningIcon"] > * {{
            display: none !important;
        }}
        [data-testid="stStatusWidgetRunningIcon"] {{
            display: inline-flex !important;
            align-items: center;
            justify-content: center;
        }}
        [data-testid="stStatusWidgetRunningIcon"]::after {{
            content: "";
            width: 0.9rem;
            height: 0.9rem;
            border: 2px solid currentColor;
            border-top-color: transparent;
            border-radius: 50%;
            opacity: 0.55;
            animation: sh-status-spin 0.7s linear infinite;
        }}
        @keyframes sh-status-spin {{
            to {{ transform: rotate(360deg); }}
        }}

        /* Tighten the top of the page. In wide layout Streamlit reserves ~6rem
           of padding above the main container plus a header spacer, which leaves
           a large empty band above the title. Trim the container padding, zero
           the (transparent) header spacer, and drop the title's own top margin so
           the header sits near the top edge and the control row follows without a
           big gap. Deploy menu / status widget stay reachable (header kept, just
           collapsed — not display:none). */
        div[data-testid="stMainBlockContainer"],
        .block-container {{
            padding-top: 2.5rem;
        }}
        [data-testid="stHeader"] {{
            height: 0;
            background: transparent;
        }}
        div[data-testid="stMainBlockContainer"] h1 {{
            padding-top: 0;
            margin-top: 0;
        }}

        /* ---- KPI metrics as stat-tile cards ------------------------------- */
        /* Turn the flat st.metric widgets into bordered cards: a soft surface
           fill (theme's secondary background), a hairline border, rounded
           corners and padding. Uses theme CSS variables + semi-transparent grey
           so it adapts cleanly to both light and dark. Applies to the 7-KPI row
           and the stacked per-SKU metric column alike. */
        [data-testid="stMetric"] {{
            /* Translucent grey fill instead of a theme variable: it lifts off a
               white surface as a soft grey card and off a dark surface as a raised
               panel, so it is correct in BOTH modes without depending on a
               Streamlit CSS variable that may not exist (a light fallback would
               make dark-mode cards light-grey with unreadable light text). */
            background: rgba(128,128,128,0.10);
            border: 1px solid rgba(128,128,128,0.22);
            border-radius: 0.6rem;
            padding: 0.85rem 1rem 0.9rem 1rem;
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }}
        /* Uniform top-row KPI bubbles: scoped to the wide KPI row (keyed container
           in kpis.py) so the side / stacked metrics keep their natural size. Each
           card fills its column's full height — the columns already stretch to the
           tallest, so every bubble matches — with a floor so the row still looks
           even. Content stays top-aligned; extra space pads the shorter cards. */
        .st-key-kpi_bubble_row [data-testid="stColumn"] {{
            align-self: stretch;
        }}
        /* Carry height:100% down through Streamlit's wrapper divs so the card can
           actually fill the stretched column (an auto-height wrapper in between
           would otherwise collapse the chain). */
        .st-key-kpi_bubble_row [data-testid="stColumn"] > div,
        .st-key-kpi_bubble_row [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-kpi_bubble_row [data-testid="stVerticalBlock"],
        .st-key-kpi_bubble_row [data-testid="stElementContainer"] {{
            height: 100%;
        }}
        .st-key-kpi_bubble_row [data-testid="stMetric"] {{
            height: 100%;
            min-height: 7.25rem;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }}
        /* ---- Historical Summary: clickable KPI tiles ---------------------- */
        /* Three labelled sections of four. Streamlit rejects a repeated container
           key, so these rows cannot reuse kpi_bubble_row's exact class; a prefix
           selector gives them the identical equal-height treatment. Additive: the
           rules above are untouched and keep applying to every existing view. */
        [class*="st-key-histkpi-row-"] [data-testid="stColumn"] {{
            align-self: stretch;
        }}
        /* stLayoutWrapper, NOT stVerticalBlockBorderWrapper: that testid does not
           exist in Streamlit 1.58 (zero occurrences in its bundle), so the selector
           it replaces was dead. 1.58 inserts an unkeyed stLayoutWrapper twice in
           this chain — between the row container and the columns, and between each
           column and its tile container — and height:100% has to be carried through
           every level or the chain collapses to auto. */
        [class*="st-key-histkpi-row-"] [data-testid="stColumn"] > div,
        [class*="st-key-histkpi-row-"] [data-testid="stLayoutWrapper"],
        [class*="st-key-histkpi-row-"] [data-testid="stVerticalBlock"],
        [class*="st-key-histkpi-row-"] [data-testid="stElementContainer"] {{
            height: 100%;
        }}
        [class*="st-key-histkpi-tile-"] [data-testid="stMetric"] {{
            height: 100%;
            min-height: 7.25rem;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }}
        /* Fixed label height — two lines at the 0.74rem label size. This is what
           makes the VALUES line up: "Top-10 Revenue Share" wraps to two lines while
           "Active SKUs" fits on one, so without a floor the numbers beside each
           other start at different heights and twelve tidy figures read as a ragged
           list. The vertical half of the consistency fix; fmt_compact is the
           horizontal half.
           (Card height itself is handled by min-height on the metric above. Only 5
           of the 12 tiles carry a year-over-year delta, and Streamlit omits the
           delta element entirely when there is none — so nothing here can reserve
           its space; top-aligned content plus the card floor is what keeps a
           delta-less tile the same size as its neighbour.) */
        [class*="st-key-histkpi-tile-"] [data-testid="stMetricLabel"] {{
            min-height: 2.2rem;
            display: flex;
            align-items: flex-start;
        }}
        /* Whole-tile click target. st.metric has no click event, so each tile pairs
           it with a real keyed button that CSS stretches over the card.
           DEGRADATION: the button is an ordinary button that CSS merely MOVES — if a
           Streamlit release changes this DOM and the selector stops matching, every
           tile falls back to a plainly visible working button rather than a dead
           card. opacity (not display:none) keeps it in tab order, and focus-within
           reveals it so keyboard users can see where they are. */
        [class*="st-key-histkpi-tile-"] {{
            position: relative;
            cursor: pointer;
        }}
        [class*="st-key-histkpi-go-"] {{
            position: absolute;
            inset: 0;
            z-index: 3;
            opacity: 0;
        }}
        /* THE FIX for "only the top of the tile was clickable". Streamlit's
           div[data-testid="stButton"] is emitted with NO height — its styled-component
           takes width and height as props and Button.tsx passes only width — so it
           computes to auto. height:100% on the <button> then resolved against an
           auto parent, collapsed to auto, and the button fell back to its min-height
           (~2.5rem) pinned at the top of a 7.25rem card, leaving the lower two thirds
           of the tile dead. Giving stButton a definite height repairs the chain.
           This is also why the tile button carries NO help=: a tooltip inserts three
           further auto-height wrappers AND makes Streamlit emit two <button>s
           (desktop + mobile), which would break the chain again. */
        [class*="st-key-histkpi-go-"] [data-testid="stButton"] {{
            height: 100%;
        }}
        [class*="st-key-histkpi-go-"] button {{
            width: 100%;
            height: 100%;
        }}
        [class*="st-key-histkpi-go-"]:focus-within {{
            opacity: 1;
        }}
        [data-testid="stMetric"]:hover {{
            border-color: rgba(128,128,128,0.40);
            box-shadow: 0 1px 6px rgba(0,0,0,0.06);
        }}
        /* Label: smaller, muted, subtly tracked — reads as a caption above the number. */
        [data-testid="stMetricLabel"] p {{
            font-size: 0.74rem !important;
            font-weight: 600;
            letter-spacing: 0.02em;
            opacity: 0.72;
        }}
        /* Value: tabular figures so digits align across the row; sized to fit the
           narrow 1/7 columns without wrapping long dollar amounts. */
        [data-testid="stMetricValue"] {{
            font-size: 1.55rem !important;
            font-weight: 600;
            font-variant-numeric: tabular-nums;
            line-height: 1.15;
        }}
        [data-testid="stMetricDelta"] {{
            font-variant-numeric: tabular-nums;
        }}

        /* ---- Detail-card KPI tiles --------------------------------------- */
        /* The same st.metric card, re-proportioned for a detail card: a tile there
           is 1/4 of the card, not 1/7 of the page, and its label ("All-Time
           POS/Orders Average") is longer than the page row's. So the value steps
           down and is allowed to wrap — without this, a long dollar amount or a
           model name overflows its tile. Deliberately re-uses the tile chrome above
           (fill, border, radius, hover) rather than restating it, so the cards and
           the page KPI row can never drift apart. */
        [class*="st-key-detailcard-"] [data-testid="stMetricValue"] {{
            font-size: 1.2rem !important;
            white-space: normal;
        }}
        /* Identity fields (Customer, Region, Model Used, Status, ...) are short
           strings, not measurements. Sized as a caption and allowed to break mid-word
           so "Holt-Winters (triple) exponential smoothing" reads as a label instead
           of a headline, and tabular-nums is dropped — it only helps digits. */
        [class*="st-key-kpitile-text-"] [data-testid="stMetricValue"] {{
            font-size: 0.95rem !important;
            font-weight: 500;
            font-variant-numeric: normal;
            line-height: 1.3;
            overflow-wrap: anywhere;
        }}
        /* Equal-height tiles within each row of the grid. Same height cascade as the
           page KPI row above (Streamlit's wrapper divs are auto-height, so height:100%
           has to be carried down every level or the chain collapses). Without it a
           value that wraps to three lines leaves its neighbours short and the grid
           reads ragged. No min-height: card tiles size to their own content, unlike
           the page row where a fixed floor keeps the seven bubbles uniform. */
        [class*="st-key-kpitiles-"] [data-testid="stColumn"] {{
            align-self: stretch;
        }}
        [class*="st-key-kpitiles-"] [data-testid="stColumn"] > div,
        [class*="st-key-kpitiles-"] [data-testid="stVerticalBlockBorderWrapper"],
        [class*="st-key-kpitiles-"] [data-testid="stVerticalBlock"],
        [class*="st-key-kpitiles-"] [data-testid="stElementContainer"] {{
            height: 100%;
        }}
        [class*="st-key-kpitiles-"] [data-testid="stMetric"] {{
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }}

        /* ---- Heading rhythm ---------------------------------------------- */
        /* Give section headers (st.subheader / ### markdown -> h2/h3) consistent
           breathing room so sections separate evenly, with a hair of negative
           tracking for a tighter, more designed look. */
        div[data-testid="stMainBlockContainer"] h2,
        div[data-testid="stMainBlockContainer"] h3 {{
            margin-top: 1.4rem;
            margin-bottom: 0.4rem;
            letter-spacing: -0.01em;
        }}
        div[data-testid="stMainBlockContainer"] h4 {{
            margin-top: 1.0rem;
            margin-bottom: 0.3rem;
            letter-spacing: -0.005em;
        }}

        /* ---- General polish ---------------------------------------------- */
        /* Softer, rounded expander & dataframe frames and a bit more air around
           dividers. Colors come from theme vars / translucent grey so both modes
           stay correct. */
        [data-testid="stExpander"] details {{
            border: 1px solid rgba(128,128,128,0.22);
            border-radius: 0.6rem;
        }}
        [data-testid="stDataFrame"] {{
            border-radius: 0.5rem;
            overflow: hidden;
        }}
        [data-testid="stCaptionContainer"] {{
            opacity: 0.8;
        }}
        hr {{
            margin: 1.4rem 0 1.0rem 0;
            opacity: 0.5;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ----- Model selection logic (the widget renders later, in the top panel) -
    # This runs BEFORE load_pipeline() below so the chosen pipeline loads on the
    # same run: pipeline_path() reads st.session_state["model_choice"], and the
    # pending-model switch must write that key before the selectbox is
    # instantiated (Streamlit forbids setting a widget-keyed value once its
    # widget exists this run). The selectbox itself is rendered into the top
    # control panel further down, and only for the Quick Projections scope.
    if not MODEL_OPTIONS:
        st.error(
            "No forecasting pipeline found — expected "
            "models/exponential_smoothing.py, models/holt_winters.py, "
            "models/xgboost.py, models/tsb.py or models/regression.py next "
            "to dashboard.py (or set DEMAND_PIPELINE)."
        )
        st.stop()

    def _on_model_change():
        # Autofitted parameters belong to the previous model; drop them so
        # the new pipeline re-autofits (or falls back to its file defaults).
        # A structural change also recomputes automatically via the compute
        # gate.
        #
        # Drop the "autofit_tried" marker too: it and autofit_params are one
        # logical fact ("we have a backtest result for this model/view/
        # snapshot"). Clearing only the params leaves the marker asserting we
        # already tried, so returning to a smoothing model would SKIP the
        # backtest and silently fall back to file-default α/β/φ — changing
        # the forecast for an unchanged view. Keep the two in lock-step.
        st.session_state.pop("autofit_params", None)
        st.session_state.pop("autofit_tried", None)

    # After "Recommend best model" picks a winner, switch the model selector to
    # it so the screen shows that model. The switch is stashed as a pending key
    # (the button handler runs *after* the widget) and applied here, before the
    # selectbox is instantiated. We replicate _on_model_change's side effects
    # since applying it programmatically doesn't fire on_change.
    pending_model = st.session_state.pop("_pending_model_choice", None)
    if pending_model in MODEL_OPTIONS and pending_model != st.session_state.get(
        "model_choice"
    ):
        st.session_state["model_choice"] = pending_model
        _on_model_change()

    # Help text for the forecasting-model selector (rendered later in the panel).
    # Must be flush-left: Markdown treats lines indented 4+ spaces as a code
    # block, which would show the literal ** and never wrap (horizontal scroll).
    _MODEL_HELP = (
        "**Forecasting models**\n\n"
        "- **8-Week Moving Average** – Simple baseline that forecasts using the "
        "average demand over the previous 8 weeks.\n"
        "- **Holt's Exponential Smoothing** – Captures both level and trend.\n"
        "- **Holt-Winters Exponential Smoothing** – Extends Holt's method with "
        "seasonality; well suited for recurring demand patterns.\n"
        "- **XGBoost** – Machine-learning model that captures complex, nonlinear "
        "patterns. Best with ample history and predictive features.\n"
        "- **TSB (Teunter-Syntetos-Babai)** – For intermittent demand, where "
        "products have many zero-demand weeks with occasional sales."
    )

    P = load_pipeline(pipeline_path())
    # Brand header: the simplehuman logo mark left of the H1 title, replacing the
    # old 📦 emoji. Rendered as one flex row (logo <img> + <h1>) rather than
    # st.logo() — the CSS above zeroes the Streamlit header band where st.logo()
    # would render, clipping it. The <h1> still picks up the theme's heading
    # styling. Degrades to a plain title if the bundled logo asset is missing.
    _logo_uri = _logo_data_uri()
    if _logo_uri:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:0.65rem; margin:0 0 0.25rem 0;">
              <img src="{_logo_uri}" alt="simplehuman"
                   style="height:44px; width:auto; display:block; border-radius:0.25rem;"/>
              <h1 style="margin:0; padding:0; font-size:2.25rem; font-weight:700; line-height:1.1;">
                Demand Projection Dashboard
              </h1>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.title("Demand Projection Dashboard")
    # Header caption: the pipeline can supply its own (DASHBOARD_CAPTION, e.g.
    # the XGBoost pipeline); otherwise fall back to the smoothing-aware blurbs.
    # It describes the *selected* model, so it's rendered next to the Forecasting
    # model selector further down and only for the single-model views (the
    # combined best-model view uses a different model per group, and Exceptions is
    # model-agnostic — both supply their own captions).
    caption = getattr(P, "DASHBOARD_CAPTION", None)
    if caption:
        header_caption = caption
    elif _supports_smoothing(P):
        header_caption = (
            "15-week Holt damped-trend forecast from the historical demand "
            "window (POS where available, else Orders). Smoothing (α/β/φ) is "
            "autofitted per view by backtesting."
        )
    else:
        tw = getattr(P, "TREND_WEIGHT", None)
        header_caption = (
            "15-week forecasts from the historical demand window "
            "(POS where available, else Orders"
            + (f"; trend weight = {tw})." if tw is not None else ").")
        )

    # ----- Top-of-page control panel ---------------------------------------
    # No sidebar: every control lives here. Placeholder containers pin the visual
    # order — the data source sits at the very top (it decides which data feeds
    # the forecast), then the view tabs, then the By-Region sub-selectors, then
    # the model + analysis panel — even though several are populated later (they
    # need `df`, which the data-source block loads, and `view`/`today_ts`,
    # resolved after that). Containers decouple render position from execution
    # order, so the data-source block still runs before the region/model panels
    # while rendering above them.
    data_source_slot = st.container()
    view_slot = st.container()
    region_slot = st.container()
    panel = st.container()
    with panel:
        col_model = st.container()

    # Defaults so the recommend-button handlers below are always well-defined,
    # even for views that hide their button (each view renders at most one).
    run_agent = False
    run_all = False
    provider_label = None
    anthropic_no_key = False

    with view_slot:
        # The four top-level views as a button-bar segmented control. Keeps
        # key="scope" and the same internal view IDs the rest of the app reads
        # (the model-selection logic above resolves the pipeline off it).
        scope = st.segmented_control(
            "View",
            [QUICK_VIEW, BEST_MODEL_COMBINED_VIEW, EXCEPTIONS_VIEW,
             HISTORICAL_VIEW, WATCHLIST_VIEW],
            default=QUICK_VIEW,
            key="scope",
            format_func=lambda s: SCOPE_LABELS.get(s, s),
            label_visibility="collapsed",
        )
        # segmented_control returns None if the user deselects the active pill;
        # fall back to the persisted choice (or the default) so a view is always
        # resolved.
        if scope is None:
            scope = st.session_state.get("scope") or QUICK_VIEW

        # Contextual help: one line describing the active view, in place of the
        # old "About these views" expander that listed everything at once.
        st.caption(SCOPE_CAPTIONS.get(scope, ""))

    # Optimized Projections / Exceptions / Watchlist are their own view IDs, so
    # `scope` IS the view. Quick Projections is a container tab: its Region /
    # Customer-group dropdowns need list_views(df), so it resolves to a real view
    # ID only once the Data source block has loaded the frame — see the
    # region_slot fill below, which is also where `view` gets set for it.
    view = None if scope == QUICK_VIEW else scope

    # ----- Data source (very top of the page) ------------------------------
    # Promoted above the view tabs because the chosen snapshot/warehouse/Plytix
    # data determines every projection. Renders into data_source_slot (declared
    # first) though it executes here — before the region selectors and model
    # panel that depend on the `df` it loads.
    with data_source_slot:
        # Breathing room between the page title/brand header and the data
        # controls below, so the "Sync from Data Warehouse" button isn't crowded
        # up against the title. Kept here (top of the first-rendered block) so the
        # gap is consistent whether the header shows the logo row or the plain
        # title fallback.
        st.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)
        files = discover_raw_files()
        df = None
        today_str = None
        # (SKU, Customer) combos in the current allocation, used to drop phantom
        # rows from the missing-projections table. Keyed on the current week so
        # its trailing-3-month window rolls forward without stale caching.
        allocation_pairs = None
        # SKU -> current total On Hand, for the Exceptions spikes table's WOS column.
        # Same raw-frame source as allocation_pairs; also week-keyed so the "current"
        # on-hand week rolls forward.
        onhand_by_sku = None
        # Path of the on-disk snapshot in play, or None when the user has
        # overridden the data with an upload. It identifies the demand input for
        # the persistent forecast cache's key (see the data_sig block below);
        # uploaded bytes have no stable file identity, so that path deliberately
        # leaves the disk cache out of the picture and only the in-session caches
        # apply.
        snapshot_path = None
        _week_key = data_io._this_week_start().isoformat()

        # Background pulls are coordinated through lock files, so their state
        # is known before the snapshot dropdowns are drawn (needed to auto-select
        # the fresh files the instant a pull finishes — see below).
        running, started = refresh_in_progress()
        wh_running, wh_started = warehouse_refresh_in_progress()

        # ----- Pull fresh data straight from the warehouse ------------------
        # One button refreshes everything: the demand snapshot and the five
        # regional warehouse-projection files are pulled in the background
        # (see start_refresh / start_warehouse_refresh); the Plytix feed is
        # re-fetched immediately via a cache-busting nonce. The page keeps
        # serving the current snapshots and switches to the new ones once
        # they land. The demand pull is a fast INCREMENTAL one (last few
        # weeks + projections merged into the newest snapshot); the nightly
        # scheduled task still does the full 36-month pull as the
        # self-healing baseline.
        if running or wh_running:
            st.info(
                f"⏳ Syncing from data warehouse… started at "
                f"{fmt_when(started or wh_started)}. "
                "You can keep working on the current snapshot; the page "
                "switches to the fresh data automatically when it finishes "
                "(usually a few minutes)."
            )

        # A pull that FAILS must say so. It used to just release its lock and
        # re-enable the button, so six days of failed syncs looked identical to
        # "nobody clicked sync" and the dashboard quietly served stale data. The
        # marker is written next to each lock, so this banner survives a rerun
        # and is visible from every session and both hosts.
        for _label, _when, _host, _detail in sync_failures():
            st.error(
                # Label first, verbatim — lowercasing it to fit a sentence turned
                # "Key-SKU list" into "key-sku list".
                f"❌ **{_label} — couldn't sync.** The last attempt "
                f"{fmt_when(_when)} didn't finish. You're still seeing the "
                "previous data, which is safe to keep working with."
            )
            # The raw error line and the host that produced it are what a
            # developer needs and what everyone else scrolls past, so they go
            # behind the same "Technical details" disclosure the top-level
            # error handler uses (see the except block at the end of _run).
            with st.expander(f"Technical details — {_label} sync"):
                st.code(_detail)
                st.caption(
                    f"Reported by `{_host}`. Full logs are in "
                    f"{os.path.dirname(dated_log_path(LOG_FILENAME))}."
                )
        if sync_failures() and st.button("Dismiss sync errors", key="clear_sync_err"):
            clear_sync_failures()
            st.rerun()

        # The refresh button, the "manually override" toggle, and the reasoning-
        # LLM selector sit side by side at the top. When the toggle is off
        # (default) every file picker below is hidden and the app just loads the
        # newest files / Plytix feed; flip it on to reveal the snapshot
        # selectboxes and upload boxes. The LLM selector picks which model powers
        # the "Model analysis" recommend button (rendered below the view tabs); it
        # lives up here so it reads at the same level as the sync/override
        # controls. It's irrelevant to the model-agnostic Exceptions scan, so its
        # column is dropped there (keeping the button/toggle widths stable).
        # Data controls (sync button, snapshot status, manual-override toggle)
        # group on the LEFT; the reasoning-LLM selector sits on the RIGHT. The
        # override toggle is a data control, so keeping it beside the sync button
        # — rather than stranded in a middle column — closes the wide gaps the old
        # three-across layout left and reads as one coherent group. The data
        # column keeps its width regardless of scope; col_llm is simply dropped
        # for the model-agnostic Exceptions scope, leaving the right side blank.
        col_data, col_llm = st.columns([2, 1], gap="large", vertical_alignment="top")
        # Both of these scopes are model-agnostic — Exceptions is a pure
        # actuals-vs-plan scan and Historical Summary runs no forecast at all — so
        # the reasoning-LLM selector has nothing to act on in either.
        if scope in (EXCEPTIONS_VIEW, HISTORICAL_VIEW):
            col_llm = None
        with col_data:
            # Sync writes fresh data to a new dated snapshot on disk. While the
            # "Manually override data" toggle is on the user is analyzing chosen /
            # uploaded files that take precedence over anything on disk, so a pull
            # would be invisible and pointless — disable Sync in that case. The
            # toggle renders further down (after this button), so read its
            # persisted state from session_state; toggling it always triggers a
            # rerun, so the value is current by the time this matters. Default
            # mirrors the toggle's own default (on only when no snapshot on disk).
            override_on = st.session_state.get("data_override", not files)
            do_refresh = False
            if running or wh_running:
                if st.button("Check for new data", key="check_refresh"):
                    st.rerun()
            else:
                do_refresh = st.button(
                    "🔄 Sync from Data Warehouse",
                    key="refresh_all",
                    disabled=override_on,
                    help="Pull the demand snapshot (last few weeks + current "
                         "projections) and the five regional warehouse-projection "
                         "files from the data warehouse now, in the background, and "
                         "re-fetch list prices from the Plytix feed. The page stays "
                         "usable and switches to the new snapshots when they're "
                         "ready. A nightly job does the full pull. Unavailable while "
                         "\"Manually override data\" is on, since the pull writes to "
                         "a new snapshot on disk that your chosen/uploaded files "
                         "override.",
                )
                if override_on:
                    st.caption(
                        "Sync is paused while \"Manually override data\" is on — "
                        "turn it off to sync."
                    )
            # A compact timestamp of the last data-warehouse pull, so users know
            # how fresh the auto-loaded data is without opening the manual
            # pickers. Sits right under the sync button as its status line.
            if files:
                _d0, _p0 = files[0]
                st.caption(f"Latest snapshot: {fmt_when(os.path.getmtime(_p0))}")
            override = st.toggle(
                "Manually override data",
                # Default on only when there's no snapshot on disk yet, so a
                # first-time user still gets the upload box; otherwise off so
                # the manual pickers stay hidden until explicitly requested.
                value=not files,
                key="data_override",
                help="""
        **Off (default)**:
        Automatically loads the latest data snapshot, Plytix feed, and warehouse files.

        **On**:
        Lets you choose specific files from previous snapshots or upload your own files for analysis.
        """,
            )
        # Reasoning-LLM selector (drives the "Model analysis" recommend button
        # below the tabs). Rendered here so it sits level with the sync/override
        # controls; sets provider_label / anthropic_no_key before col_model reads
        # them. Skipped for the model-agnostic Exceptions scope (col_llm is None).
        if col_llm is not None:
            with col_llm:
                provider_label = st.radio(
                    "Reasoning LLM",
                    list(LLM_PROVIDERS.keys()),
                    key="agent_llm_provider",
                    help="""
    Select which large language model (LLM) generates the forecast summary and anomaly analysis.

    **Anthropic (Claude):** uses Anthropic's Claude API and requires an `ANTHROPIC_API_KEY`.

    **Local (Gemma):** runs Google's Gemma model locally and does not require an external API.
    """,
                )
                # Anthropic needs a key; without one, block the run and steer the
                # user to Local rather than silently degrading to it.
                anthropic_no_key = LLM_PROVIDERS[provider_label] == "anthropic" and not (
                    os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                )
                if anthropic_no_key:
                    st.caption(
                        "⚠️ Claude isn't set up on this machine — select "
                        "**Local LLM** to run the analysis."
                    )

        if do_refresh:
            ok_dw, msg_dw = start_refresh()
            if ok_dw:
                # Remember the newest mtime NOW so we can tell, on completion,
                # whether the pull actually produced a newer file.
                st.session_state["_refresh_active"] = True
                st.session_state["_refresh_baseline"] = max(
                    (os.path.getmtime(p) for _, p in files), default=0.0
                )
            _wh_paths_now = [
                p for ps in data_io.discover_warehouse_files().values() for p in ps
            ]
            ok_wh, msg_wh = start_warehouse_refresh()
            if ok_wh:
                st.session_state["_wh_refresh_active"] = True
                st.session_state["_wh_refresh_baseline"] = max(
                    (os.path.getmtime(p) for p in _wh_paths_now), default=0.0
                )
            # Ride the key-SKU pull along so the Key SKUs watchlist stays current.
            # Fire-and-forget: the watchlist tab re-discovers the new file on its
            # own, so no baseline/auto-select bookkeeping is needed here.
            start_key_skus_refresh()
            st.session_state["plytix_nonce"] = (
                st.session_state.get("plytix_nonce", 0) + 1
            )
            if ok_dw or ok_wh:
                st.success(
                    "Sync started — it's running in the background, so you can "
                    "keep working. The page picks up the new data on its own."
                )
                st.rerun()
            else:
                # Neither pull started, so both carry a reason; the demand one is
                # the primary and `or` keeps the warehouse reason as a fallback
                # rather than dropping it on the floor.
                st.warning(msg_dw or msg_wh)

        # If a refresh we launched this session just finished AND actually wrote
        # a newer file than existed when it started, jump the snapshot selection
        # to that newest file so the page shows the fresh pull without a manual
        # pick. Done BEFORE the selectbox is instantiated (Streamlit forbids
        # setting a widget-keyed value once its widget exists this run).
        if st.session_state.get("_refresh_active") and not running:
            st.session_state.pop("_refresh_active", None)
            baseline = st.session_state.pop("_refresh_baseline", 0.0)
            newest_mtime = max((os.path.getmtime(p) for _, p in files), default=0.0)
            if files and newest_mtime > baseline:
                d0, p0 = files[0]
                st.session_state["snapshot_choice"] = f"{d0}  ({os.path.basename(p0)})"
                st.toast("Fresh snapshot loaded from the data warehouse.")
            else:
                st.warning(
                    "The sync finished but didn't bring back a new snapshot, so "
                    "you're still seeing the previous data. Logs are in "
                    f"{os.path.dirname(dated_log_path(LOG_FILENAME))} if this "
                    "keeps happening."
                )

        # Manual data-file pickers live in ONE collapsible section at the top so
        # the sync button + toggle stay uncluttered. A single expander object is
        # reused across the snapshot / prices / warehouse blocks below: Streamlit
        # forbids *nesting* expanders, but re-entering the same expander via
        # `with data_exp:` just appends to it, which is allowed.
        data_exp = None
        if override:
            data_exp = st.expander(
                "Data files (snapshot / prices / warehouse)", expanded=not files
            )

        if files:
            labels = {f"{d}  ({os.path.basename(p)})": (d, p) for d, p in files}
            if override:
                with data_exp:
                    choice = st.selectbox(
                        "Snapshot (raw file)", list(labels.keys()), key="snapshot_choice"
                    )
            else:
                # Toggle off: always the newest snapshot (== the refresh
                # auto-select target), no widget shown.
                choice = list(labels.keys())[0]
            today_str, path = labels[choice]
            snapshot_path = path
            df = load_raw_from_path(path, os.path.getmtime(path), pipeline_path())
            allocation_pairs = load_allocation_pairs_from_path(
                path, os.path.getmtime(path), _week_key
            )
            onhand_by_sku = load_onhand_by_sku_from_path(
                path, os.path.getmtime(path), _week_key
            )
        elif override:
            with data_exp:
                st.info("Upload the Demand Planning Details and Plytix files below.")

        # Show the upload box only when overriding. The toggle defaults on when
        # there's no on-disk snapshot yet, so a first-time user still lands here.
        if override:
            with data_exp:
                st.markdown("**Demand Planning Details — PowerBI export**")
                up = st.file_uploader("all_demand_projections_*.xlsx", type=["xlsx"])
                if up is not None:
                    data = up.getvalue()
                    df = load_raw_from_bytes(data, up.name, pipeline_path())
                    allocation_pairs = load_allocation_pairs_from_bytes(
                        data, up.name, _week_key
                    )
                    onhand_by_sku = load_onhand_by_sku_from_bytes(
                        data, up.name, _week_key
                    )
                    today_str = _date_from_name(up.name)

        # ----- List prices (drive revenue risk) ---------------------------
        # The Plytix export doubles as the source of each SKU's list price AND
        # its 'Active in' regions (used by the active-in check below), so we read
        # both from whichever Plytix source is in play. Precedence: a manually
        # uploaded workbook wins; otherwise pull the public Plytix channel feed
        # (the default, so no file has to be dragged); otherwise fall back to the
        # newest local list_prices_*.xlsx on disk.
        prices = None
        plytix_df = None
        up_price = None
        price_file = discover_price_file()
        if override:
            with data_exp:
                st.markdown("**Revenue risk — list prices (Plytix override)**")
                up_price = st.file_uploader(
                    "list_prices_*.xlsx", type=["xlsx"], key="price_upload",
                    help="SKU + List Price USD, plus SKU Status / SKU Type / "
                        "'Active in'. Drives revenue risk (projection difference × "
                        "list price) and the active-in check. Overrides the Plytix "
                        "feed when set.",
                )

        if up_price is not None:
            prices = load_prices_from_bytes(
                up_price.getvalue(), up_price.name, pipeline_path()
            )
            plytix_df = read_plytix_from_bytes(up_price.getvalue(), up_price.name)
            if prices is not None and override:
                st.success(f"{len(prices):,} list prices (uploaded)")
        elif data_io.PLYTIX_FEED_URL:
            nonce = st.session_state.setdefault("plytix_nonce", 0)
            try:
                plytix_df = fetch_plytix_from_url(data_io.PLYTIX_FEED_URL, nonce)
                prices = data_io.prices_from_plytix(plytix_df)
                if prices is not None and override:
                    st.success(f"{len(prices):,} list prices (Plytix feed)")
            except Exception as e:  # network/parse failure -> fall back to disk
                plytix_df = None
                prices = None
                if override:
                    st.warning(
                        "Couldn't reach the Plytix feed, so list prices are "
                        "coming from the newest local price file instead."
                    )
                    with st.expander("Technical details — Plytix feed"):
                        st.code(str(e))

        # Fall back to the newest local xlsx when neither an upload nor the feed
        # produced prices (feed disabled/unreachable and nothing uploaded).
        if prices is None and up_price is None and price_file is not None:
            prices = load_prices_from_path(
                price_file, os.path.getmtime(price_file), pipeline_path()
            )
            plytix_df = read_plytix_from_path(
                price_file, os.path.getmtime(price_file)
            )
            if prices is not None and override:
                st.success(
                    f"{len(prices):,} list prices "
                    f"({os.path.basename(price_file)})"
                )

        # ----- Warehouse projections (drive the "missing projections" table) --
        # A DIFFERENT data source than the demand file above: the warehouse
        # projection exports (one per region: AU/CA/EU/JP/US) list which
        # SKU×customer×week cells carry a projection — a missing cell is
        # exactly what the "missing future projections" table finds, so it
        # needs these files, not the demand file. The nightly SQL pull (or the
        # refresh button under "Data source") writes them; a manual PowerBI
        # export (wide matrix or long table layout — the reader sniffs which)
        # still works.
        warehouse_df = None

        # If a warehouse refresh we launched just finished and actually wrote a
        # newer snapshot, jump the snapshot selection to it (before the widget
        # is instantiated — same dance as the demand snapshot above).
        wh_snapshots = data_io.discover_warehouse_files()
        _wh_all_paths = [p for ps in wh_snapshots.values() for p in ps]
        if st.session_state.get("_wh_refresh_active") and not wh_running:
            st.session_state.pop("_wh_refresh_active", None)
            baseline = st.session_state.pop("_wh_refresh_baseline", 0.0)
            newest_mtime = max(
                (os.path.getmtime(p) for p in _wh_all_paths), default=0.0
            )
            if wh_snapshots and newest_mtime > baseline:
                st.session_state["warehouse_snapshot"] = next(iter(wh_snapshots))
                st.toast("Fresh warehouse projections loaded from the data warehouse.")
            else:
                st.warning(
                    "The warehouse projections synced but didn't bring back a new "
                    "file, so you're still seeing the previous ones. Logs are in "
                    f"{os.path.dirname(dated_log_path(LOG_FILENAME))} if this "
                    "keeps happening."
                )

        if override:
            with data_exp:
                st.markdown("**Warehouse projections — AU/CA/EU/JP/US**")
                if wh_snapshots:
                    wh_choice = st.selectbox(
                        "Warehouse snapshot",
                        list(wh_snapshots.keys()),
                        key="warehouse_snapshot",
                        help="Each snapshot is the set of regional warehouse exports "
                             "sharing that date.",
                    )
                    wh_paths = tuple(wh_snapshots[wh_choice])
                    warehouse_df = load_warehouse_from_paths(
                        wh_paths, tuple(os.path.getmtime(p) for p in wh_paths)
                    )
                    st.caption(
                        f"{len(wh_paths)} file(s): "
                        + ", ".join(os.path.basename(p) for p in wh_paths)
                    )
                up_wh = st.file_uploader(
                    "AU/CA/EU/JP/US_warehouse_projections_*.xlsx",
                    type=["xlsx"], accept_multiple_files=True, key="warehouse_upload",
                    help="One wide export per region. The region is read from the "
                         "filename prefix (AU/CA/EU/JP/US).",
                )
                if up_wh:
                    warehouse_df = load_warehouse_from_uploads(
                        tuple((f.name, f.getvalue()) for f in up_wh)
                    )
                if warehouse_df is not None and not warehouse_df.empty:
                    locs = ", ".join(sorted(warehouse_df["Region Code"].unique()))
                    st.success(f"{len(warehouse_df):,} projection rows ({locs})")
        elif wh_snapshots:
            # Toggle off: silently load the newest warehouse snapshot so the
            # missing-projections table still works without the picker showing.
            wh_choice = next(iter(wh_snapshots))
            wh_paths = tuple(wh_snapshots[wh_choice])
            warehouse_df = load_warehouse_from_paths(
                wh_paths, tuple(os.path.getmtime(p) for p in wh_paths)
            )

    if df is None:
        st.warning("Upload the Demand Planning Details Projections file to get started.")
        st.stop()

    # A snapshot date anchors the entire 8-week history / 15-week forecast
    # window, so we don't silently fall back to "today" — a wrong anchor
    # produces plausible-looking but wrong numbers. If the filename carried no
    # date, ask the user to confirm one explicitly before computing anything.
    if not today_str:
        st.warning(
            "No snapshot date was found in the filename. The date sets the "
            "8-week history and 15-week forecast windows, so please confirm "
            "it before continuing."
        )
        picked = st.date_input(
            "Snapshot date (as-of date for this data)",
            value=pd.Timestamp.today().normalize(),
            help="Usually the date the raw file was exported. Everything is "
                 "computed relative to this date.",
            key="manual_snapshot_date",
        )
        confirmed = st.checkbox(
            "Use this date", key="confirm_snapshot_date",
            help="Tick to compute the forecast with the date above.",
        )
        if not confirmed:
            st.info("Confirm a snapshot date above to continue.")
            st.stop()
        today_str = pd.Timestamp(picked).strftime("%Y-%m-%d")
        logger.info("Snapshot date manually confirmed by user: %s", today_str)

    today_ts = pd.Timestamp(today_str)
    lb, lcw, ffw = P.week_anchors(today_ts)

    # ----- Exclusions: never forecast a SKU that shouldn't be projected -----
    # Two Plytix-driven filters, applied to the demand frame BEFORE forecasting
    # so a discontinued/inactive SKU — or an active SKU in a region it is not
    # 'Active in' (e.g. ST1082, active in US/CA/UK/SG/EU/AU, appearing under JP
    # (NETDEPOT)) — is never projected, flagged, or counted in revenue, and is
    # surfaced in its own table below. The identical logic runs in the agent's
    # ingest node (agent/data_io.py is the single source of truth), so the
    # dashboard and the agent agree on which SKUs are in scope.
    #
    # The exclusion filters run full-frame groupbys/region-maps but their result
    # is view-INDEPENDENT — identical for every customer view on the same
    # snapshot/prices/model. Memoise it in session_state keyed on a lightweight
    # signature (same len()-marker style used for price_marker below) so
    # switching views doesn't re-run the groupbys every time. today_str pins the
    # anchors (lb/lcw/ffw derive from it), so it isn't needed in the key.
    excl_sig = (
        today_str,
        pipeline_path(),
        None if df is None else len(df),
        None if plytix_df is None else len(plytix_df),
    )
    if (
        st.session_state.get("_excl_sig") == excl_sig
        and "_excl_result" in st.session_state
    ):
        excl = st.session_state["_excl_result"]
    else:
        excl = data_io.apply_exclusions(df, plytix_df, P, anchors=(lb, lcw, ffw))
        st.session_state["_excl_result"] = excl
        st.session_state["_excl_sig"] = excl_sig
    df = excl.df
    # Same frame minus the discontinued drop, for the Historical Summary only. That
    # filter removes a retired SKU's ENTIRE history, which is right for forecasting
    # and wrong for a backward-looking view — see agent/data_io.ExclusionResult.
    # Every forecast path below keeps using `df`.
    df_hist = excl.df_with_discontinued
    check_ran = excl.active_check_ran
    inactive_df = excl.inactive_df
    disc_check_ran = excl.disc_check_ran
    discontinued_df = excl.discontinued_df
    n_excluded_rows = excl.n_excluded_rows
    excluded_counts_by_key = excl.excluded_counts_by_key
    if n_excluded_rows:
        logger.info(
            "Active-in check: dropped %d raw rows across %d SKU×customer×"
            "region combos not in the SKU's 'Active in' list.",
            n_excluded_rows, len(inactive_df),
        )
    if excl.n_disc_rows:
        logger.info(
            "Discontinued check: dropped %d raw rows across %d "
            "discontinued/inactive SKUs (trailing '*' or Plytix status).",
            excl.n_disc_rows, excl.n_disc_skus,
        )

    # ----- Persistent forecast-cache signature -----------------------------
    # Identifies the inputs every forecast on this page derives from, so results
    # can be cached to disk (outputs/.cache) and survive a restart, a browser
    # refresh, or a different planner opening the app — and so the nightly
    # agent.batch can warm them. Built HERE, after exclusions, because `df` from
    # this point on is the post-exclusion frame: the excluded row count and the
    # final row count are what make the cleaned frame reproducible, on top of the
    # file identities (mtime + size, so an in-place incremental refresh of the
    # same filename invalidates too).
    #
    # Threaded explicitly into the compute functions rather than held in module
    # state: Streamlit runs each browser session's script on its own thread
    # against the same module objects, so a shared global could let one session's
    # snapshot key another session's forecast. None (an upload override, or no
    # snapshot on disk) simply disables the disk tier.
    data_sig = None
    if snapshot_path is not None:
        data_sig = forecast_cache.snapshot_signature(
            snapshot_path,
            n_excluded_rows=n_excluded_rows,
            n_rows=len(df) if df is not None else None,
            # Content-hashed rather than file-stat'd: prices arrive from an
            # upload, the Plytix feed, or a local xlsx, so no path identifies
            # them, and they land in the revenue-risk columns.
            prices=forecast_cache.content_signature(prices),
        )

    # ----- View selector (Quick Projections Region / Customer group) --------
    # The scope buttons rendered at the top of the page (into view_slot) already
    # set `view` for the three scopes that ARE view IDs. Quick Projections needs
    # list_views(df), so its two dropdowns are filled here — into region_slot,
    # which reserved its spot directly under the buttons.
    #
    # Between them they reach every scope a planner wants, with no nested toggle:
    # everything (ALL_REGIONS + "All customers"), one region combined, or one
    # customer group. Each resolves to one of the same three view-string shapes
    # the app has always used, so nothing downstream of `view` changes.
    if scope == QUICK_VIEW:
        with region_slot:
            by_region = list_views(df)
            c1, c2 = st.columns(2)
            # key=str: a custom pipeline's region_for_group may return non-string
            # labels; sorting by their string form keeps the selectbox from
            # crashing on mixed types (see logs.txt, 2026-07-06). The RAW key
            # indexes by_region below — region_all_view()'s f-string does the str
            # coercion _region_frame expects.
            region = c1.selectbox(
                "Region", [ALL_REGIONS] + sorted(by_region.keys(), key=str),
                key="quick_region",
            )
            if region == ALL_REGIONS:
                # Every group across every region, plus the all-customers combined
                # fit. The flattened list is exactly the group list
                # compute_by_customer iterates, so the table's groups match.
                options = [ALL_CUSTOMERS_VIEW] + sorted(
                    {g for groups in by_region.values() for g in groups}, key=str
                )
            else:
                # First entry is the synthetic per-region rollup, every group in
                # this region combined. Its stored value embeds the region so
                # caches/keys stay unique across regions.
                options = [region_all_view(region)] + by_region[region]
            # Changing Region re-options this selectbox. A keyed selectbox keeps
            # the options list out of its element identity, and Streamlit resets a
            # no-longer-offered value to the first option (writing it back in the
            # same run), so switching region lands on that region's rollup without
            # an exception or an extra rerun. Pinned by test_phase5_dashboard.
            view = c2.selectbox(
                "Customer group", options, key="quick_group",
                format_func=quick_group_label, help="Type to search",
            )

    # ----- Agent summary (LangGraph pipeline) ------------------------------
    # Button-triggered only: invoking the graph backtests all three models AND
    # calls an LLM, which is far too slow/expensive to run on every rerun. The
    # provider selector switches the reasoning nodes between the Claude API and
    # a local OpenAI-compatible server; agent/llm.py re-reads LLM_PROVIDER from
    # the env at call time, so setting it here just before invoke() is enough.
    # The Exceptions view is model-agnostic (pure actuals-vs-plan; no forecast is
    # fit), so the whole model-analysis apparatus — reasoning-LLM selector, the
    # all-views recommendation run, and any "No ANTHROPIC_API_KEY" warning — is
    # irrelevant there and would only confuse. Skip it entirely for that view.
    # ----- Forecasting model + Model analysis (left column of the panel) ----
    with col_model:
        # Forecasting model: only Quick Projections uses a chosen model (Optimized
        # Projections picks one per group; Exceptions is model-agnostic).
        if scope == QUICK_VIEW:
            st.subheader("Forecasting model", help=_MODEL_HELP)
            # The model dropdown and the "Recommend best model" button sit side by
            # side — the button is short, so it takes a narrow column. The no-key
            # warning is folded into the (disabled) button's hover tooltip instead
            # of a standalone caption; provider_label / anthropic_no_key come from
            # the top-row reasoning-LLM selector.
            m_col, b_col = st.columns([3, 1], vertical_alignment="bottom")
            with m_col:
                st.selectbox(
                    "Forecasting model", list(MODEL_OPTIONS.keys()),
                    key="model_choice", on_change=_on_model_change,
                    format_func=model_display, label_visibility="collapsed",
                )
            with b_col:
                run_agent = st.button(
                    "Recommend best model",
                    key="run_agent_summary",
                    disabled=anthropic_no_key,
                    help=(
                        "⚠️ Claude isn't set up on this machine (no "
                        "ANTHROPIC_API_KEY) — pick Local LLM above to enable this."
                        if anthropic_no_key else
                        "Backtests all models for this view, recommends the most "
                        "accurate one, and writes an AI summary + flagged anomalies. "
                        "Slow — runs only when you click, never on a normal rerun."
                    ),
                )
            # Blurb describing the selected model (computed near the title). Only
            # Quick Projections reaches here, which matches its old suppression
            # (combined/best-model and Exceptions supply their own captions).
            st.caption(header_caption)

        # Model analysis: only Optimized Projections keeps its own section — the
        # global all-views recommendation run. Quick Projections renders its
        # "Recommend best model" button inline beside the model dropdown above;
        # Exceptions is model-agnostic (no analysis apparatus).
        if scope == BEST_MODEL_COMBINED_VIEW:
            st.subheader("Model analysis")
            # Optimized Projections is the only place the global all-views run
            # lives — the combined table is built from every group's own best
            # model. Same work as `python -m agent.batch`; runs hidden in the
            # background, and while it runs the button becomes a status check.
            batch_running, batch_started = batch_in_progress()
            if batch_running:
                elapsed = batch_elapsed_suffix(batch_started)
                prog = batch_progress()
                if prog:
                    done, total = prog
                    st.info(f"⏳ Recommending the best model for every view — "
                            f"{done} of {total} done.{elapsed} This runs in the "
                            "background, so you can keep using the dashboard.")
                else:
                    st.info(f"⏳ Recommending the best model for every view — getting "
                            f"started.{elapsed} This runs in the background, so you "
                            "can keep using the dashboard.")
                if st.button("Check progress", key="check_agent_batch"):
                    st.rerun()
            else:
                # A just-finished run (this session): surface its outcome once.
                proc = st.session_state.get("agent_batch_proc")
                if proc is not None and proc.poll() is not None:
                    st.session_state["_batch_toast"] = (
                        batch_result_message() or "Recommendations finished."
                    )
                    st.session_state.pop("agent_batch_proc", None)
                run_all = st.button(
                    "Recommend models (all views)",
                    key="run_agent_all",
                    disabled=anthropic_no_key,
                    help="Recommends the most accurate model for EVERY view and "
                         "writes each recommendation to disk. Runs ~114 views — can "
                         "take up to 1 hour. Asks for confirmation first.",
                )
                if run_all:
                    _confirm_run_all_dialog(LLM_PROVIDERS[provider_label])

                # If the last run left any views un-updated, name them and offer
                # a targeted retry (re-runs ONLY those, not the whole batch).
                failures = batch_failures()
                if failures:
                    names = [v for v, _ in failures]
                    st.warning(
                        "These views couldn't be updated last time:\n"
                        + "\n".join(f"- {n}" for n in names)
                    )
                    if st.button("Retry failed views", key="retry_agent_failed",
                                 disabled=anthropic_no_key):
                        ok, msg = start_agent_batch(
                            LLM_PROVIDERS[provider_label], views=names
                        )
                        st.session_state["_batch_toast"] = (
                            f"Retrying {len(names)} view(s)…" if ok else f"⚠️ {msg}"
                        )
                        st.rerun()

    # Surface batch start/finish toasts once (set from the dialog / poll above).
    if "_batch_toast" in st.session_state:
        st.toast(st.session_state.pop("_batch_toast"))

    if run_agent:
        # Kick off the pipeline on a background thread and rerun immediately, so
        # the (minutes-long) run never blocks the script. Progress is polled by
        # _agent_progress_fragment; completion is finalized further below.
        os.environ["LLM_PROVIDER"] = LLM_PROVIDERS[provider_label]  # llm.py reads env at call time
        # Remember that the agent was run for this view this session, so the
        # summary expander below appears only after an explicit click — never a
        # stale persisted summary surfacing on page load.
        st.session_state.setdefault("agent_ran_views", set()).add(view)
        shared = {
            "status": "running",
            "progress": 0.0,
            "step": "Starting…",
            "view": view,
            "started_at": time.time(),  # so the progress panel can show elapsed time
            "result": {},
            "error": None,
        }
        thread = threading.Thread(
            target=_run_agent_job, args=(view, today_ts, shared), daemon=True
        )
        st.session_state["agent_job"] = shared
        st.session_state["agent_job_thread"] = thread
        thread.start()
        st.rerun()

    job = st.session_state.get("agent_job")
    if job is not None and job.get("view") == view:
        status = job.get("status")
        if status == "running":
            # Live, non-blocking progress. Only the fragment reruns on its timer;
            # everything else on the page stays interactive.
            _agent_progress_fragment()
        elif status in ("done", "error"):
            # A full rerun (fired by the fragment) lands here once the run ends.
            result = job.get("result") or {}
            if status == "error" or result.get("errors"):
                st.error(job.get("error") or "\n".join(result.get("errors", [])) or "Model analysis failed.")
                job["status"] = "shown"  # consume so the error isn't re-raised on later reruns
            else:
                best = result.get("best_model") or (_load_agent_summary(view) or {}).get("best_model")
                started = job.get("started_at")
                dur = f" in {int(time.time() - started)}s" if started else ""
                st.toast(f"Recommended model{dur}: {best or 'n/a'}")
                job["status"] = "shown"  # consume before any rerun below
                # Switch the model toggle to the agent's winner so the screen
                # shows the best model. Stash it as a pending key and rerun: the
                # toggle widget already rendered above, so it can't be written
                # here — the pending value is applied before the widget rebuilds.
                if best in MODEL_OPTIONS and best != st.session_state.get("model_choice"):
                    st.session_state["_pending_model_choice"] = best
                    st.rerun()

    # Show the cached run (from the JSON publish wrote) only for views the user
    # has run the agent on this session — clicking is what reveals it.
    if view in st.session_state.get("agent_ran_views", set()):
        _render_agent_summary(view)

    # ----- Historical Summary view -----------------------------------------
    # Backward-looking analytics with no forecast in it at all, so it comes first
    # in the branch chain and stops before every compute path below. Note the
    # frame: df_hist keeps discontinued SKUs' active years, which the projection
    # views correctly drop (see the exclusions block above).
    if view == HISTORICAL_VIEW:
        render_historical_summary(
            df_hist, today_ts, today_str, prices, n_excluded_rows, (lb, lcw, ffw),
            P, plytix_df=plytix_df, data_sig=data_sig, onhand_by_sku=onhand_by_sku,
        )
        st.stop()

    # ----- Combined best-model-per-group view ------------------------------
    # This view has no single model, so it skips the smoothing/autofit step, the
    # single-model compute, and the charts/KPIs below entirely: it renders the
    # stitched per-group best-model table and stops.
    if view == BEST_MODEL_COMBINED_VIEW:
        _render_best_model_combined(
            df, today_ts, today_str, prices, n_excluded_rows, (lb, lcw, ffw), P,
            data_sig=data_sig, onhand_by_sku=onhand_by_sku,
        )
        st.stop()

    # ----- Exceptions view -------------------------------------------------
    # Model-agnostic actuals-vs-plan scan; like the best-model view it renders its
    # own table and stops before the single-model compute/charts/KPIs below.
    if view == EXCEPTIONS_VIEW:
        render_exceptions(
            df, today_ts, today_str, prices, n_excluded_rows, (lb, lcw, ffw), P,
            warehouse_df=warehouse_df, plytix_df=plytix_df, check_ran=check_ran,
            inactive_df=inactive_df, excluded_counts_by_key=excluded_counts_by_key,
            disc_check_ran=disc_check_ran, discontinued_df=discontinued_df,
            allocation_pairs=allocation_pairs, onhand_by_sku=onhand_by_sku,
            data_sig=data_sig,
        )
        st.stop()

    # ----- Watchlist view --------------------------------------------------
    # Pin-board of starred (SKU, Customer Grouping) pairs. Reuses the best-model
    # table for its detail numbers; like the two views above it renders its own
    # table and stops before the single-model compute/charts/KPIs below.
    if view == WATCHLIST_VIEW:
        render_watchlist(
            df, today_ts, today_str, prices, n_excluded_rows, (lb, lcw, ffw), P,
            data_sig=data_sig, onhand_by_sku=onhand_by_sku,
        )
        st.stop()

    # ----- Model parameters (Holt damped-trend smoothing) ------------------
    # Parameters are hidden from the UI entirely: Holt always uses autofitted
    # α/β/φ (backtested per view/snapshot), falling back to the pipeline's file
    # defaults when the backtest can't run. min-weeks uses the file default.
    # This runs inline (no sidebar); its only visible output is a transient
    # tuning spinner and a toast, so it doesn't inject controls into the page.
    min_weeks = None
    alpha = beta = phi = None
    smoothing_ok = _supports_smoothing(P)
    min_weeks_ok = _supports_min_weeks(P)

    if smoothing_ok or min_weeks_ok:
        # The pipeline's own constants are the "file defaults".
        a0 = float(getattr(P, "ALPHA", 0.5))
        b0 = float(getattr(P, "BETA", 0.3))
        p0 = float(getattr(P, "PHI", 0.85))
        mw0 = int(getattr(P, "MIN_WEEKS_FOR_TREND", 4))

        if min_weeks_ok:
            min_weeks = mw0

        # Autofit results are keyed per (model, view, snapshot) so revisiting an
        # already-fitted view restores its params instantly — no re-backtest and
        # no extra st.rerun(). Both maps are insertion-ordered dicts bounded by
        # AUTOFIT_CACHE_MAX; autofit_tried also records failed attempts (best is
        # None) so a backtest that can't score isn't retried on every rerun.
        autofit_key = (pipeline_path(), view, today_str)
        autofit_map = st.session_state.setdefault("autofit_params", {})
        autofit_tried_map = st.session_state.setdefault("autofit_tried", {})
        autofit = autofit_map.get(autofit_key)
        autofit_active = autofit is not None

        # ----- Always autofit -------------------------------------------
        # Selecting a smoothing model (or a new view / snapshot) runs the
        # backtest once per (model, view, snapshot) and uses the winning α/β/φ.
        autofit_tried = autofit_key in autofit_tried_map
        if (
            smoothing_ok
            and _supports_autofit(P)
            and not autofit_active
            and not autofit_tried
        ):
            _bounded_put(autofit_tried_map, autofit_key, True, AUTOFIT_CACHE_MAX)
            with st.spinner("Tuning the forecast for this view…"):
                best = run_autofit(df, view, today_ts, pipeline_path(), mw0,
                                   data_sig)
            if best is not None:
                logger.info(
                    "Autofit [%s]: alpha=%.2f beta=%.2f phi=%.2f "
                    "(MAE %.2f vs %.2f with file defaults)",
                    view, best["alpha"], best["beta"], best["phi"],
                    best["mae"], best["baseline_mae"],
                )
                _bounded_put(autofit_map, autofit_key, dict(best), AUTOFIT_CACHE_MAX)
                autofit = autofit_map[autofit_key]
                autofit_active = True
                # Recompute the forecast with the fitted values.
                st.session_state["_do_recompute"] = True
                st.rerun()

        if smoothing_ok:
            if autofit_active:
                alpha, beta, phi = (
                    autofit["alpha"], autofit["beta"], autofit["phi"]
                )
            else:
                alpha, beta, phi = a0, b0, p0

        if smoothing_ok and _supports_autofit(P) and autofit_active:
            improve = autofit["baseline_mae"] - autofit["mae"]
            pct = (
                f" ({improve / autofit['baseline_mae'] * 100:.0f}% better "
                "than the default settings)"
                if autofit["baseline_mae"] > 0 and improve > 0 else ""
            )
            st.toast(f"Forecast auto-tuned for this view{pct}.")

    # ----- Compute (with a progress bar) -----------------------------------
    # Forecasts are cached PER VIEW in session_state (an insertion-ordered dict
    # bounded by FC_CACHE_MAX), so switching back to an already-visited view is an
    # instant lookup — no recompute, no full-DataFrame re-hashing by
    # compute_view's @st.cache_data, and no re-running the compute_by_customer
    # loop/progress bar. The key carries every structural input (view / model /
    # snapshot / prices / data) PLUS the smoothing params, so it self-invalidates
    # whenever any of them change (a new snapshot, a model swap, or autofit
    # landing new α/β/φ all produce a fresh key -> a genuine recompute).
    price_marker = None if prices is None else int(len(prices))
    cache_key = (
        view, pipeline_path(), today_str, price_marker, n_excluded_rows,
        alpha, beta, phi, min_weeks,
    )

    do_recompute = st.session_state.pop("_do_recompute", False)
    fc_cache = st.session_state.setdefault("fc_cache", {})
    if do_recompute:
        fc_cache.pop(cache_key, None)

    if cache_key in fc_cache:
        # Move-to-end so a revisited view isn't the next eviction victim.
        stored = fc_cache.pop(cache_key)
        fc_cache[cache_key] = stored
        (summary, weekly, agg, by_cust, weekly_by_group, agg_by_group,
         coverage) = stored
    else:
        prog = st.progress(0.0, text="Preparing…")
        try:
            summary = weekly = agg = None
            by_cust = weekly_by_group = agg_by_group = None
            coverage = None
            region_all = region_from_view(view)
            is_combined = view == ALL_CUSTOMERS_VIEW or region_all is not None
            if is_combined:
                # ONE forecast per (SKU, customer group), and the view's totals are
                # the SUM of those. There is deliberately no second fit on the
                # customer-summed history here: it produced a different number for
                # the same SKU than the table below it (+19.9% apart on the live
                # snapshot, 90% of which was the Orders-only customers that a
                # single per-SKU source selection silently dropped), and totals
                # that disagree with their own parts are not usable for ordering.
                # Optimized Projections has always worked this way; this is Quick
                # Projections being brought in line with it, through the same
                # roll_up_to_sku_week helper so the two provably agree.
                def _bump(done, total, group):
                    frac = 0.05 + 0.9 * (done / max(total, 1))
                    prog.progress(
                        min(frac, 0.98),
                        text=f"Per-customer forecast… ({done}/{total})",
                    )
                # A region rollup breaks out only its own region's groups.
                src = df if region_all is None else _region_frame(df, P, region_all)
                by_cust, weekly_by_group, agg_by_group = (
                    compute_by_customer_frames(
                        src, today_ts, pipeline_path(),
                        prices, alpha, beta, phi, min_weeks, progress_cb=_bump,
                        data_sig=data_sig,
                    )
                )
                if by_cust is not None and not by_cust.empty:
                    # Resolve each (SKU, customer)'s own POS/Orders signal BEFORE
                    # summing, so the actuals total covers every customer the same
                    # way the forecast total does. Charts and KPIs read the
                    # resulting `demand` column via summaries.historical_window.
                    agg_by_group = resolve_demand(agg_by_group, by_cust)
                    weekly = roll_up_to_sku_week(
                        weekly_by_group, ["projected_pos"], min_count=0)
                    agg = roll_up_to_sku_week(
                        agg_by_group, ["POS", "Orders", "Projection", "demand"])
                    # Make the existing-plan column summable BEFORE rolling up: the
                    # models' per-series mean divides by however many horizon weeks
                    # that series has a plan for, so the rows could not be added.
                    by_cust = attach_current_projection(
                        by_cust, agg_by_group, weekly["WeekDate"],
                        ["Customer Grouping", "SKU"],
                    )
                    summary = roll_up_summary(by_cust, agg, (lb, lcw, ffw), view)
                    # Descriptive averages at SKU grain, from the resolved demand.
                    summary = attach_descriptive_averages(
                        summary, sku_grain_demand_frame(agg, view), today_ts)
                    summary = attach_top_volume(summary, P, src, today_ts)
                    coverage = _coverage_note(src, by_cust, agg_by_group,
                                              weekly["WeekDate"])
            else:
                # The view IS one customer group: one fit, and the total already IS
                # the part. compute_view's single-group branch does exactly what the
                # per-group loop would (same slice, same aggregate, same label), so
                # single_group_frames reuses it rather than paying a second fit.
                prog.progress(0.15, text="Building forecast for this view…")
                summary, weekly, agg = compute_view(
                    df, view, today_ts, pipeline_path(),
                    prices, alpha, beta, phi, min_weeks, data_sig,
                )
                if summary is not None and not summary.empty:
                    by_cust, weekly_by_group, agg_by_group = single_group_frames(
                        summary, weekly, agg, view, today_ts,
                    )
                    agg_by_group = resolve_demand(agg_by_group, by_cust)
                    agg = roll_up_to_sku_week(
                        agg_by_group, ["POS", "Orders", "Projection", "demand"])
                    # Same fixed-denominator existing-plan figure the combined views
                    # use, so the column means ONE thing whichever view you open. A
                    # single group has one row per SKU, so summary and by_cust get
                    # the identical correction.
                    horizon = weekly["WeekDate"]
                    by_cust = attach_current_projection(
                        by_cust, agg_by_group, horizon, ["Customer Grouping", "SKU"])
                    summary = attach_current_projection(
                        summary, agg, horizon, ["SKU"])
            prog.progress(1.0, text="Done")
        finally:
            prog.empty()

        # A 7th element, but not a 7th DataFrame: `coverage` is a handful of ints and
        # short group-name lists, so it costs nothing against the ~34 MB the
        # per-group aggregate in slot 6 dominates (see FC_CACHE_MAX above). It has to
        # ride along rather than be recomputed on a cache hit because it is derived
        # from `src`, which only the miss branch builds.
        _bounded_put(
            fc_cache, cache_key,
            (summary, weekly, agg, by_cust, weekly_by_group, agg_by_group, coverage),
            FC_CACHE_MAX,
        )

    if summary is None or summary.empty:
        st.error(
            f"No POS or Orders in the 8-week window for **{view}** — "
            "nothing to forecast."
        )
        st.stop()

    # On Hand / Weeks of Supply for the detail cards. Attached HERE, after the cache
    # read, rather than inside compute_by_customer_frames: On Hand comes from a
    # separately loaded map, not from the fit, so it must not become part of a
    # forecast-cache key that only tracks forecast inputs.
    by_cust = attach_supply_columns(by_cust, onhand_by_sku)

    # ----- Header / windows -------------------------------------------------
    st.subheader(quick_group_label(view))
    w1, w2 = st.columns(2)
    # The window's nominal lower bound (lb) can sit earlier than the first week
    # the data actually reaches — e.g. the all-history pipelines anchor lb a few
    # years before the run date but the raw file's earliest week is later. Show
    # the first week that is genuinely used in the fit and the chart (earliest
    # WeekDate within [lb, lcw] carrying a POS/Orders signal) rather than the
    # nominal lb, so the displayed start matches what the graph plots.
    win = agg[(agg["WeekDate"] >= lb) & (agg["WeekDate"] <= lcw)]
    win_sig = win[win["POS"].notna() | win["Orders"].notna()]
    hist_start = win_sig["WeekDate"].min() if not win_sig.empty else lb
    # Count the completed weeks actually used — distinct weeks within the window
    # that carry a POS/Orders signal. The regression pipeline's window is a fixed
    # 8 weeks; the all-history pipelines (Holt/XGBoost) span however many weeks of
    # data exist between hist_start and lcw, so the count is derived, not fixed.
    n_hist_weeks = win_sig["WeekDate"].nunique()
    week_word = "week" if n_hist_weeks == 1 else "weeks"
    # Muted parenthetical uses Streamlit's :gray[...] colored-text directive, which
    # the frontend recolors per active theme (readable on both light and dark).
    hist_span = (
        f"**Historical window** &nbsp; {hist_start.date()} → {lcw.date()} "
        f":gray[({n_hist_weeks} completed {week_word})]"
    )
    w1.markdown(hist_span, unsafe_allow_html=True)
    fc_weeks = pd.to_datetime(weekly["WeekDate"])
    w2.markdown(
        f"**Forecast window** &nbsp; {ffw.date()} → "
        f"{fc_weeks.max().date()} "
        f":gray[({fc_weeks.nunique()} weeks)]",
        unsafe_allow_html=True,
    )

    # ----- KPIs -------------------------------------------------------------
    # Window label for the historical-demand KPI's help text: taken from
    # compute_view's summary, which carries exactly the selected model's own
    # average column, so it always describes the (lb, lcw) span the metric uses.
    # by_cust carries BOTH averages, so its stacked KPIs must be told explicitly.
    # Window label for the total-weekly-demand KPI. It has to describe `anchors`, so
    # it comes from the SIDEBAR model's own AVG_COL_LABEL — not from `summary`, which
    # now carries BOTH descriptive averages (attach_descriptive_averages puts All-Time
    # first, so resolve_avg_col's fallback would mislabel an 8-week window). Same
    # reasoning, and the same line, as _render_best_model_combined.
    anchors_avg_col = (
        getattr(P, "AVG_COL_LABEL", EIGHT_WK_AVG_COL) if P is not None
        else resolve_avg_col(summary)
    )
    _render_kpis(summary, agg, (lb, lcw, ffw), avg_col=anchors_avg_col)
    # Every figure above is the sum of the per-customer rows in the table below, so
    # anything those rows don't cover is missing from the total. Say what, and how much.
    _render_coverage_note(coverage)

    # ----- Aggregate chart --------------------------------------------------
    # Per-chart date-range picker (own key => independent from the other charts).
    agg_ctrl, _ = st.columns([1, 2])
    with agg_ctrl:
        agg_range = chart_range_control(agg, weekly, lcw, key="range_agg")
    st.plotly_chart(
        aggregate_chart(agg, summary, weekly, (lb, lcw, ffw), quick_group_label(view),
                        date_range=agg_range, prices=prices),
        width="stretch",
    )

    has_by_cust = by_cust is not None and not by_cust.empty
    groups = (
        sorted(by_cust["Customer Grouping"].astype(str).unique())
        if has_by_cust else []
    )
    # Chart-only anchors for the per-group charts below, mirroring Optimized
    # Projections: `lb` is as short as 8 weeks with the 8-Week Moving Average
    # model, which would trap their date-range pickers inside that window. The
    # KPIs keep the model's own anchors so no number shifts — only how much
    # history the charts are ALLOWED to show changes.
    chart_anchors = (
        (pd.to_datetime(agg_by_group["WeekDate"]).min(), lcw, ffw)
        if has_by_cust else (lb, lcw, ffw)
    )

    # ----- Per-customer detail ----------------------------------------------
    # One customer group's total weekly demand, drawn from that group's un-summed
    # frames. Skipped when the view resolves to a single group: the KPI row and the
    # total-demand chart above already ARE that group.
    if len(groups) > 1:
        st.markdown("### Customer detail")
        customer = st.selectbox(
            "Customer", groups, help="Type to search", key="quick_customer"
        )
        agg_c = agg_by_group[
            agg_by_group["Customer Grouping"].astype(str) == customer
        ]
        wk_c = weekly_by_group[
            weekly_by_group["Customer Grouping"].astype(str) == customer
        ]
        summary_c = by_cust[by_cust["Customer Grouping"].astype(str) == customer]
        ccL, ccR = st.columns([3, 1])
        with ccL:
            cust_range = chart_range_control(agg_c, wk_c, lcw, key="range_cust_quick")
            st.plotly_chart(
                aggregate_chart(
                    agg_c, summary_c, wk_c,
                    (pd.to_datetime(agg_c["WeekDate"]).min(), lcw, ffw),
                    customer, date_range=cust_range, prices=prices,
                ),
                width="stretch",
            )
        with ccR:
            # Same seven metrics as the top of the view, scoped to this group and
            # stacked to fit the side column. Uses the section's original anchors
            # (not the widened chart range) so the historical-demand window lines
            # up with the KPI row above.
            _render_kpis(summary_c, agg_c, (lb, lcw, ffw), stacked=True,
                         avg_col=anchors_avg_col)

    # ----- Per-SKU detail (view total) --------------------------------------
    # The mirror of Customer detail, drilled the other way. The section itself lives
    # in kpis.render_sku_detail_section, shared with Optimized Projections so the two
    # views read identically (the same reason render_sku_detail_card is shared).
    #
    # Gated to the all-customers scopes (combined or a region rollup), the only ones
    # where "total across all customers" is a number the page doesn't already show:
    # for a single bare customer group the KPI row and the total-demand chart above
    # ARE that group, so the section would only restate the table's detail cards.
    #
    # `summary` (SKU grain) drives the tiles, `by_cust` the customer breakdown. Both
    # are the ROLL-UP of that SKU's per-customer forecasts, the same figures behind
    # the KPI row, the chart above and the by-SKU expander below.
    is_view_total = view == ALL_CUSTOMERS_VIEW or region_from_view(view) is not None
    if is_view_total:
        render_sku_detail_section(summary, agg, weekly, by_cust, (lb, lcw, ffw),
                                  prices, avg_col=anchors_avg_col, key="quick")

    # ----- Summary table by SKU (view total), collapsed ---------------------
    # One row per SKU: the SUM of that SKU's customer rows in the table below, and the
    # per-SKU grain the KPI row and the chart are built from. It used to be a
    # separately fit combined series — a different number under identical column
    # names, which is what this whole change removed. It is kept (rather than deleted
    # as a duplicate) because per-SKU IS the grain an order is placed at, and because
    # it is where the top-volume breakdown appears as a column.
    #
    # Directly under SKU detail because it is the table form of exactly what that
    # section just charted — one row per SKU at the view total — so the two read
    # together; collapsed so it still doesn't compete with the main table beneath it.
    # Skipped for a single customer group, where one row per SKU and one row per
    # (SKU, customer) are the same table: it shares `is_view_total` with the
    # SKU-detail section above so the two gates can't drift.
    if is_view_total:
        with st.expander("Summary table by SKU (view total)"):
            st.caption(
                "One row per SKU for the whole view — the sum of that SKU's customer "
                "rows below, so the two tables tie out exactly. Both carry the same "
                "observed demand averages (model-independent) plus the recent 8-week "
                "run-rate; the only column unique to this one is the top-volume "
                "customer breakdown."
            )
            # No "(model fit)" rename any more: there is no separate fitted average
            # here to distinguish. attach_descriptive_averages gives this frame the
            # same central observed figures every other table shows, so the column
            # means exactly what its name says.
            summary_table = summary
            if RISK_COL in summary_table.columns \
                    and summary_table[RISK_COL].notna().any():
                # Largest revenue risk first, by magnitude (a big drop is as much a
                # "risk" as a big gain); SKUs with no price (blank risk) sort last.
                summary_table = (
                    summary_table.assign(_abs_risk=summary_table[RISK_COL].abs())
                    .sort_values("_abs_risk", ascending=False, na_position="last")
                    .drop(columns="_abs_risk")
                    .reset_index(drop=True)
                )
                st.caption(
                    "Ordered by largest revenue risk (by magnitude); blanks last."
                )
            render_filtered_table(summary_table, "filter_by_sku", P)
            st.download_button(
                "⬇️ Download the summary table by SKU",
                data=view_to_excel(with_export_flags(summary_table),
                                   with_export_flags(weekly)),
                file_name=f"{view.replace('/', '-').replace(' ', '_')}"
                          f"_dashboard_by_sku_demand_projections_{today_str}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_by_sku",
            )

    # ----- Summary table by SKU and customer --------------------------------
    # The page's main table: every SKU broken out by customer group, mirroring the
    # pipeline's ALL_CUSTOMERS_demand_projections file. Computed alongside the main
    # forecast above (and cached in session_state) so it stays on the same
    # snapshot / prices / parameters as the KPIs and charts.
    #
    # Deliberately NOT collapsible: it is the page's main content and stays on
    # screen. Only the view-total roll-up above it folds away — the two are not a
    # matched pair, and putting this one behind a click as well would leave the page
    # opening on nothing but the drill-downs.
    st.markdown("### Summary table by SKU and customer")
    if not has_by_cust:
        st.info("No per-customer forecasts to show for this snapshot.")
    else:
        if RISK_COL in by_cust.columns and by_cust[RISK_COL].notna().any():
            # Keep each SKU's customers together; within a SKU show the largest
            # revenue risk (by magnitude) first, blanks last.
            by_cust_table = (
                by_cust.assign(_abs_risk=by_cust[RISK_COL].abs())
                .sort_values(
                    ["SKU", "_abs_risk"],
                    ascending=[True, False],
                    na_position="last",
                )
                .drop(columns="_abs_risk")
                .reset_index(drop=True)
            )
            sort_note = (
                "Each SKU broken out by customer group; within a SKU, largest "
                "revenue risk first (by magnitude)."
            )
        else:
            by_cust_table = (
                by_cust.sort_values(["SKU", "Customer Grouping"])
                .reset_index(drop=True)
            )
            sort_note = "Each SKU broken out by customer group."

        st.caption(f"{sort_note} Click a row to open its chart and metrics.")

        # Top-volume breakdown is attached by attach_top_volume, so it exists on
        # the combined / region-rollup views only (a single-group view has
        # nothing to break down). It is a whole-view figure keyed by SKU; the
        # card labels it as such.
        top_groups = (
            dict(zip(summary["SKU"].astype(str),
                     summary["Top Volume Customer Groups"]))
            if "Top Volume Customer Groups" in summary.columns else None
        )
        # Condensed rows (five scannable columns); clicking one reveals that
        # exact (SKU, customer group) combination's chart, date range and
        # metrics — one row of what the SKU detail section above totals.
        #
        # `fixed`: SKU / Customer / Region / Key SKU, on screen from the first paint
        # and cross-filtered against each other, same as Optimized Projections. This
        # table used to carry its own SKU dropdown above the filter chips; the bar's
        # SKU filter is that control, so there is one instead of two, and
        # render_selectable_table derives the old `focus_single` from it — pick one
        # SKU, and if one row is left its card opens outright with no table.
        sku_card = functools.partial(
            render_sku_detail_card, agg_by_group, weekly_by_group,
            (lb, lcw, ffw), chart_anchors, prices,
            model_label=model_display(st.session_state.get("model_choice")),
            top_groups=top_groups,
        )
        render_selectable_table(
            by_cust_table, "filter_by_customer", P,
            condensed_cols=QUICK_CONDENSED_COLS, style=True,
            detail_chart=sku_card, detail_cols=QUICK_CARD_COLS,
            extra_kpis=projection_kpi_extras,
            kpi_deltas={"Projection Difference": projection_difference_delta},
            fixed=FIXED_FILTER_LABELS,
        )

        # `by_cust_table`, NOT the filtered frame: the bar narrows what is on
        # SCREEN, and a planner who drilled to one SKU before hitting download
        # still expects the whole workbook — every SKU × customer combination in
        # the view. An export that silently inherited the filters would be
        # indistinguishable from a complete one once it is off the page.
        #
        # Both sheets are the same grain's numbers: the `summary` sheet is the
        # per-customer rows and `weekly_forecast` is their roll-up, so pivoting the
        # weekly sheet reproduces the summary sheet's totals. It used to pair these
        # bottom-up rows with the combined fit's weekly sheet, which meant one
        # workbook disagreed with itself.
        #
        # "dashboard" in the name because the batch pipeline writes its own
        # ALL_CUSTOMERS_demand_projections_<date>.xlsx to outputs/ — same old
        # filename, a separate combined fit inside. Two files with one name and
        # different numbers is how a spreadsheet ends up in the wrong meeting.
        st.download_button(
            "⬇️ Download the summary table by SKU and Customer (all SKUs)",
            data=view_to_excel(with_export_flags(by_cust_table),
                               with_export_flags(weekly)),
            file_name=(
                f"{'ALL_CUSTOMERS' if view == ALL_CUSTOMERS_VIEW else view.replace('/', '-').replace(' ', '_')}"
                f"_dashboard_demand_projections_{today_str}.xlsx"
            ),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_by_customer",
        )

    # The four data-quality sections (inactive-region / missing forecasts /
    # missing POS-Orders / discontinued-with-forecasts) used to render here for
    # every Quick Projections view. They now live in the Exceptions view (both
    # tabs), rendered by render_exceptions above.


def _run():
    """Run the app, turning any uncaught exception into a friendly message.

    Non-engineer users shouldn't see a raw traceback (and it can leak column
    names / paths). We log the full traceback to logs.txt for developers and
    show a calm, actionable message instead. ``st.stop()`` raises internally to
    halt a run and must be allowed to propagate untouched.
    """
    try:
        main()
    except Exception:  # noqa: BLE001 -- deliberately broad: last line of defence
        # RerunException / StopException are Streamlit control-flow signals, not
        # errors; let Streamlit handle them normally.
        try:
            from streamlit.runtime.scriptrunner import StopException, RerunException
            _control_flow = (StopException, RerunException)
        except Exception:
            _control_flow = ()
        exc = sys.exc_info()[1]
        if _control_flow and isinstance(exc, _control_flow):
            raise

        tb = traceback.format_exc()
        logger.error("Unhandled exception in dashboard:\n%s", tb)
        st.error(
            "Something went wrong while building this view. The error has been "
            "logged for the developers. A common cause is an unexpected file "
            "format — check that the raw file is a standard "
            "`all_demand_projections_*.xlsx` export (headers on row 3) and the "
            "list-price file is a `list_prices_*.xlsx`. If it keeps happening, "
            "share the details below with the team."
        )
        with st.expander("Technical details (for developers)"):
            st.exception(exc)
            st.caption(f"Full traceback is also recorded in {dated_log_path(LOG_FILENAME)}.")
        st.stop()


if __name__ == "__main__":
    _run()
