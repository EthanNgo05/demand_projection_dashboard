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
    Terminal 2: ngrok http 8501

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
import datetime
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
    ALL_CUSTOMERS_VIEW, BEST_MODEL_COMBINED_VIEW, C_ACTUAL, C_GRID, C_ORIGINAL, C_UPDATED,
    DEFAULT_MODEL, EXCEPTIONS_VIEW, HERE, MODEL_DISPLAY, MODEL_OPTIONS, MODEL_USED_COL,
    PRICE_COL, REGION_ALL_PREFIX, REPO_ROOT, RISK_COL, SCOPE_LABELS, _ENV_PIPELINE,
    fmt_dollar, model_display, region_all_view, region_from_view,
)
from dashboard_app.pipeline import (  # noqa: F401
    _load_pipeline_cached, _supports_autofit, _supports_min_weeks, _supports_prices,
    _supports_smoothing, load_pipeline, pipeline_path,
)
from dashboard_app.summaries import (  # noqa: F401
    _format_generated_at, avg_window_phrase, customer_source_map, historical_window,
    resolve_avg_col, source_map,
)
from dashboard_app.charts import (  # noqa: F401
    _base_layout, _clip_to_range, aggregate_chart, chart_range_control, sku_chart,
)
from dashboard_app.tables import (  # noqa: F401
    render_filtered_table, style_summary,
)
from dashboard_app.datasources import (  # noqa: F401
    DISCONTINUED_COLS, INACTIVE_COLS, MISSING_COLS, MISSING_POS_COLS, WAREHOUSE_REGIONS,
    _active_in_list, _clean, _date_from_name, _raw_dir, _region_code, _this_week_start,
    compute_active_products, compute_discontinued_products,
    compute_discontinued_projections, compute_inactive_projections,
    compute_missing_pos_orders, compute_missing_projections, discover_key_skus_file,
    discover_price_file, discover_raw_files, fetch_plytix_from_url, load_key_skus,
    load_prices_from_bytes, load_prices_from_path, load_raw_from_bytes,
    load_raw_from_path, load_warehouse_from_paths, load_warehouse_from_uploads,
    price_glob, raw_glob, read_plytix_from_bytes, read_plytix_from_path,
)
from dashboard_app.compute import (  # noqa: F401
    _agent_summaries_generated_at, _agent_summaries_mtime, _agent_summary_path,
    _best_model_for_group, _forecast_one_group, _load_agent_summary, _region_frame,
    compute_by_customer, compute_by_customer_best, compute_view, list_views, run_autofit,
    summary_to_excel, view_to_excel,
)
from dashboard_app.refresh import (  # noqa: F401
    BATCH_STALE_SECONDS, EXTRACT_SCRIPT, REFRESH_STALE_SECONDS, WAREHOUSE_EXTRACT_SCRIPT,
    _batch_lock_path, _batch_log_path, _batch_result_line, _clear_lock, _launch_refresh,
    _refresh_lock_path, _refresh_log_path, _refresh_state, _wh_refresh_lock_path,
    _wh_snapshot_complete_since, batch_elapsed_suffix, batch_failures,
    batch_in_progress, batch_progress,
    batch_result_message, refresh_in_progress, start_agent_batch,
    start_refresh, start_warehouse_refresh, warehouse_refresh_in_progress,
)
from dashboard_app.agent_summary import (  # noqa: F401
    LLM_PROVIDERS, _AGENT_NODE_PROGRESS, _agent_progress_fragment, _agent_scores,
    _confirm_run_all_dialog, _model_fit_callout, _render_agent_summary, _run_agent_job,
)
from dashboard_app.kpis import (  # noqa: F401
    _render_best_model_combined, _render_kpis,
)
from dashboard_app.exceptions import (  # noqa: F401
    compute_exceptions, render_exceptions,
)
from dashboard_app.dataquality import (  # noqa: F401
    render_discontinued_section, render_inactive_section, render_missing_pos_section,
    render_missing_section,
)


def main():
    st.set_page_config(
        page_title="Demand Projections", page_icon="📦", layout="wide"
    )

    # Keep the reasoning-LLM choice across reruns where the sidebar radio isn't
    # re-rendered — the refresh buttons st.rerun() before the script reaches the
    # radio (see the "Sync" / "Check for new data" buttons above the model
    # analysis section). Streamlit garbage-collects an unrendered keyed widget's
    # state, which would snap the radio back to its first option (Anthropic) and
    # spuriously surface the "No ANTHROPIC_API_KEY" warning even when the user
    # picked Local. Re-registering the key here preserves the actual selection.
    if "agent_llm_provider" in st.session_state:
        st.session_state["agent_llm_provider"] = st.session_state["agent_llm_provider"]

    # Widen the sidebar a touch (Streamlit's default is ~244-260px). Adjust
    # SIDEBAR_WIDTH_PX to taste; users can still drag the divider to resize.
    SIDEBAR_WIDTH_PX = 340
    st.markdown(
        f"""
        <style>
        /* Only widen the sidebar while it's expanded. Scoping to
           aria-expanded="true" lets Streamlit's collapse animation drive the
           width to 0 when hidden, so the main content reclaims the full width
           instead of the min-width pinning it open. */
        section[data-testid="stSidebar"][aria-expanded="true"] {{
            width: {SIDEBAR_WIDTH_PX}px !important;
            min-width: {SIDEBAR_WIDTH_PX}px !important;
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
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ----- Model toggle ------------------------------------------------------
    # Rendered first so every downstream helper (raw-file discovery, cleaning,
    # forecasting) sees the chosen pipeline on this same run.
    with st.sidebar:
        st.header("Model")
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

        # After "Run Agent Summary" picks a best model, switch the toggle to it
        # so the screen shows that model. The switch is stashed as a pending key
        # (the button handler runs *after* this widget) and applied here, before
        # the radio is instantiated — Streamlit forbids writing a widget-keyed
        # value once its widget exists this run. We replicate _on_model_change's
        # side effects since applying it programmatically doesn't fire on_change.
        pending_model = st.session_state.pop("_pending_model_choice", None)
        if pending_model in MODEL_OPTIONS and pending_model != st.session_state.get(
            "model_choice"
        ):
            st.session_state["model_choice"] = pending_model
            _on_model_change()

        # The Optimal Projections view forecasts each group with its own
        # best model, so the sidebar model choice does nothing there. Grey the
        # radio out (it keeps its stored value) so users aren't misled. Scope is
        # read from the persisted key set by the Scope radio further down; on the
        # first run the key is absent (-> enabled), which matches the default
        # Executive Overview scope.
        _combined_view = st.session_state.get("scope") == BEST_MODEL_COMBINED_VIEW
        st.radio(
            "Forecasting model", list(MODEL_OPTIONS.keys()),
            key="model_choice", on_change=_on_model_change,
            format_func=model_display, disabled=_combined_view,
            help="""
        **Forecasting models**

        - **8-Week Moving Average** – Simple baseline model that forecasts using the average demand over the previous 8 weeks.
        - **Holt's Exponential Smoothing** – Standard time series forecasting model that captures both level and trend.
        - **Holt-Winters Exponential Smoothing** – Extends Holt's method by also modeling seasonality, making it well suited for recurring demand patterns.
        - **XGBoost** – Machine learning model that can capture complex relationships and nonlinear patterns in demand data. Best when sufficient historical data and predictive features are available.
        - **TSB (Teunter-Syntetos-Babai)** – Designed for intermittent demand, where products have many zero-demand periods with occasional sales.
        """,
        )
        if _combined_view:
            st.caption(
                "Not used in Optimal Projections — each customer group uses its "
                "own best model."
            )

    P = load_pipeline(pipeline_path())
    st.title("📦 Demand Projection Dashboard")
    # Header caption: the pipeline can supply its own (DASHBOARD_CAPTION, e.g.
    # the XGBoost pipeline); otherwise fall back to the smoothing-aware blurbs.
    # It describes the *selected* model, so render it into a slot we fill only
    # once the view is known — the combined best-model view uses a different model
    # per group, so it suppresses this caption (and supplies its own).
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
    _header_caption_slot = st.empty()

    # ----- Data source -----------------------------------------------------
    with st.sidebar:
        st.header("Data source")
        files = discover_raw_files()
        df = None
        today_str = None

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
                f"⏳ Syncing from data warehouse… started {started or wh_started}. "
                "You can keep working on the current snapshot; the page "
                "switches to the fresh data automatically when it finishes "
                "(usually a few minutes)."
            )

        # The refresh button and the "manually override" toggle sit side by
        # side. When the toggle is off (default) every file picker below is
        # hidden and the app just loads the newest files / Plytix feed; flip
        # it on to reveal the snapshot selectboxes and upload boxes.
        col_btn, col_toggle = st.columns([1, 1])
        with col_toggle:
            override = st.toggle(
                "Manually override data",
                value=False,
                key="data_override",
                help="""
        **Off (recommended)**:
        Automatically loads the latest data snapshot, Plytix feed, and warehouse files.

        **On**:
        Lets you choose specific files from previous snapshots or upload your own files for analysis.
        """,
            )
        with col_btn:
            do_refresh = False
            if running or wh_running:
                if st.button("Check for new data", key="check_refresh"):
                    st.rerun()
            else:
                do_refresh = st.button(
                    "🔄 Sync from Data Warehouse",
                    key="refresh_all",
                    help="Pull the demand snapshot (last few weeks + current "
                         "projections) and the five regional warehouse-projection "
                         "files from the data warehouse now, in the background, and "
                         "re-fetch list prices from the Plytix feed. The page stays "
                         "usable and switches to the new snapshots when they're "
                         "ready. A nightly job does the full pull.",
                )

        # A compact timestamp of the last data-warehouse pull, so users know how
        # fresh the auto-loaded data is without opening the manual pickers.
        if files:
            _d0, _p0 = files[0]
            st.caption(
                f"Latest snapshot: {_d0} — pulled "
                f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(_p0)))}"
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
            st.session_state["plytix_nonce"] = (
                st.session_state.get("plytix_nonce", 0) + 1
            )
            if ok_dw or ok_wh:
                st.success(
                    f"Refresh started ({msg_dw if ok_dw else msg_wh}) — "
                    "running in the background."
                )
                st.rerun()
            else:
                st.warning(msg_dw)

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
                    "The data-warehouse refresh didn't produce a new snapshot — "
                    "see logs/<date>/logs_refresh.txt for details."
                )

        if files:
            labels = {f"{d}  ({os.path.basename(p)})": (d, p) for d, p in files}
            if override:
                choice = st.selectbox(
                    "Snapshot (raw file)", list(labels.keys()), key="snapshot_choice"
                )
            else:
                # Toggle off: always the newest snapshot (== the refresh
                # auto-select target), no widget shown.
                choice = list(labels.keys())[0]
            today_str, path = labels[choice]
            df = load_raw_from_path(path, os.path.getmtime(path), pipeline_path())
        elif override:
            st.info("Upload the Demand Planning Details and Plytix files below.")

        # Show the upload box when overriding, and always when there's no
        # on-disk snapshot yet (otherwise a first-time user can't get started).
        if override or not files:
            with st.expander("Upload the Demand Planning Details Projections from PowerBI", expanded=not files):
                up = st.file_uploader("all_demand_projections_*.xlsx", type=["xlsx"])
                if up is not None:
                    data = up.getvalue()
                    df = load_raw_from_bytes(data, up.name, pipeline_path())
                    today_str = _date_from_name(up.name)

        # ----- List prices (drive revenue risk) ---------------------------
        # The Plytix export doubles as the source of each SKU's list price AND
        # its 'Active in' regions (used by the active-in check below), so we read
        # both from whichever Plytix source is in play. Precedence: a manually
        # uploaded workbook wins; otherwise pull the public Plytix channel feed
        # (the default, so no file has to be dragged); otherwise fall back to the
        # newest local list_prices_*.xlsx on disk.
        if override:
            st.header("Revenue risk")
        prices = None
        plytix_df = None
        up_price = None
        price_file = discover_price_file()
        if override:
            with st.expander("Override: upload a Plytix list-price file", expanded=False):
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
                        f"Couldn't fetch the Plytix feed ({e}); "
                        "falling back to the newest local list-price file."
                    )

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
        if override:
            st.header("Warehouse projections")
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
                    "The warehouse refresh didn't produce a new snapshot — "
                    "see logs/<date>/logs_refresh.txt for details."
                )

        if override:
            with st.expander(
                "Upload warehouse projection files (AU/CA/EU/JP/US)",
                expanded=not wh_snapshots,
            ):
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
    excl = data_io.apply_exclusions(df, plytix_df, P, anchors=(lb, lcw, ffw))
    df = excl.df
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

    # ----- View selector ---------------------------------------------------
    with st.sidebar:
        st.header("View")
        by_region = list_views(df)
        scope = st.radio(
            "Scope",
            [ALL_CUSTOMERS_VIEW, "By region", BEST_MODEL_COMBINED_VIEW, EXCEPTIONS_VIEW],
            index=0, key="scope",
            format_func=lambda s: SCOPE_LABELS.get(s, s),
            help="""
            Choose how forecasts are grouped before modeling.

            **Executive Overview:**
            Forecasts all customer groups as one combined demand series using the forecasting model selected in the sidebar.

            **Optimized Projections:**
            Forecasts each customer group with its own most-accurate model (determined by model analysis), then combines the results into a single table. Requires the model analysis pipeline to have been run for all customer groups.

            **By Region:**
            Forecasts only the selected fulfillment region (or customer group within that region) using the forecasting model selected in the sidebar.

            **Exceptions:**
            Scans every customer group for SKUs whose recent actual sell-through has diverged sharply from the existing system projection (the plan of record). Model-agnostic — no forecast is run.

            **Summary:**

            • Executive Overview: One model across all customers.

            • Optimized Projections: Best model for each customer group, combined into one view.

            • By Region: One model applied only to the selected customer group.

            • Exceptions: Recent actuals vs. the system projection, worst offenders first.
            """,
        )
        if scope == ALL_CUSTOMERS_VIEW:
            view = ALL_CUSTOMERS_VIEW
            region = None
        elif scope == BEST_MODEL_COMBINED_VIEW:
            view = BEST_MODEL_COMBINED_VIEW
            region = None
        elif scope == EXCEPTIONS_VIEW:
            view = EXCEPTIONS_VIEW
            region = None
        else:
            # key=str: a custom pipeline's region_for_group may return non-string
            # labels; sorting by their string form keeps the selectbox from
            # crashing on mixed types (see logs.txt, 2026-07-06).
            region = st.selectbox("Region", sorted(by_region.keys(), key=str))
            # First entry is the synthetic per-region rollup ("All Customers"),
            # every group in this region combined. Its stored value embeds the
            # region so caches/keys stay unique across regions; format_func
            # shows the short label the user expects.
            all_view = region_all_view(region)
            view = st.selectbox(
                "Customer group", [all_view] + by_region[region],
                format_func=lambda v: f"All Customers - {region}" if v == all_view else v,
            )

    # Now that the view is known, fill the header caption — except for views that
    # aren't tied to the sidebar model: the combined best-model view (a different
    # model per group) and the model-agnostic Exceptions view both render their
    # own caption instead, so the selected-model blurb would mislead.
    if view not in (BEST_MODEL_COMBINED_VIEW, EXCEPTIONS_VIEW):
        _header_caption_slot.caption(header_caption)

    # ----- Agent summary (LangGraph pipeline) ------------------------------
    # Button-triggered only: invoking the graph backtests all three models AND
    # calls an LLM, which is far too slow/expensive to run on every rerun. The
    # provider selector switches the reasoning nodes between the Claude API and
    # a local OpenAI-compatible server; agent/llm.py re-reads LLM_PROVIDER from
    # the env at call time, so setting it here just before invoke() is enough.
    with st.sidebar:
        st.header("Model analysis")
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
        # Anthropic needs a key; without one, block the run and steer the user
        # to Local rather than silently degrading to it behind the scenes.
        anthropic_no_key = LLM_PROVIDERS[provider_label] == "anthropic" and not (
            os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        )
        if anthropic_no_key:
            st.caption("⚠️ No ANTHROPIC_API_KEY found — select **Local LLM** to run the analysis.")
        # The per-view "Recommend best model" button lives on the page (next to
        # the recommendation it produces); only the global all-views run and the
        # reasoning-LLM selector stay here in the sidebar.

        # All-views run (same work as `python -m agent.batch`). Runs hidden in
        # the background; while it runs the button becomes a status check.
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
                     "writes each recommendation to disk. Runs ~60 views — can "
                     "take up to 1 hour. Asks for confirmation first.",
            )
            if run_all:
                _confirm_run_all_dialog(LLM_PROVIDERS[provider_label])

            # If the last run left any views un-updated, name them and offer a
            # targeted retry (re-runs ONLY those, not the whole batch).
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

    # ----- Model recommendation (this view) --------------------------------
    # The per-view analysis button lives on the page, right above the live
    # progress and the recommendation panel it produces (rendered below), so the
    # trigger and its output sit together. The best-model-per-group view isn't a
    # single-model view, so it has no button here — it reads every group's
    # recommendation and points to the sidebar's all-views run instead.
    run_agent = False
    if view != BEST_MODEL_COMBINED_VIEW:
        run_agent = st.button(
            "Recommend best model",
            key="run_agent_summary",
            disabled=anthropic_no_key,
            help="Backtests all models for this view, recommends the most "
                 "accurate one, and writes an AI summary + flagged anomalies. "
                 "Slow — runs only when you click, never on a normal rerun.",
        )
        if anthropic_no_key:
            st.caption("⚠️ No ANTHROPIC_API_KEY found — pick **Local LLM** in the "
                       "sidebar to enable this.")

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

    # ----- Combined best-model-per-group view ------------------------------
    # This view has no single model, so it skips the smoothing sidebar, the
    # single-model compute, and the charts/KPIs below entirely: it renders the
    # stitched per-group best-model table and stops.
    if view == BEST_MODEL_COMBINED_VIEW:
        _render_best_model_combined(
            df, today_ts, today_str, prices, n_excluded_rows, (lb, lcw, ffw), P
        )
        st.stop()

    # ----- Exceptions view -------------------------------------------------
    # Model-agnostic actuals-vs-plan scan; like the best-model view it renders its
    # own table and stops before the single-model compute/charts/KPIs below.
    if view == EXCEPTIONS_VIEW:
        render_exceptions(
            df, today_ts, today_str, prices, n_excluded_rows, (lb, lcw, ffw), P
        )
        st.stop()

    # ----- Model parameters (Holt damped-trend smoothing) ------------------
    # Parameters are hidden from the UI entirely: Holt always uses autofitted
    # α/β/φ (backtested per view/snapshot), falling back to the pipeline's file
    # defaults when the backtest can't run. min-weeks uses the file default.
    min_weeks = None
    alpha = beta = phi = None
    with st.sidebar:
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

            # Autofit results are keyed to the model/view/snapshot they were
            # fitted on; anything else falls back to the file defaults.
            autofit = st.session_state.get("autofit_params")
            autofit_active = bool(
                autofit
                and autofit.get("model") == pipeline_path()
                and autofit.get("view") == view
                and autofit.get("today") == today_str
            )

            # ----- Always autofit -------------------------------------------
            # Selecting a smoothing model (or a new view / snapshot) runs the
            # backtest once per (model, view, snapshot) and uses the winning
            # α/β/φ. The "autofit_tried" marker records that we've attempted
            # it, so a failed backtest isn't retried on every rerun and a good
            # fit isn't re-run needlessly.
            autofit_key = (pipeline_path(), view, today_str)
            autofit_tried = st.session_state.get("autofit_tried") == autofit_key
            if (
                smoothing_ok
                and _supports_autofit(P)
                and not autofit_active
                and not autofit_tried
            ):
                st.session_state["autofit_tried"] = autofit_key
                with st.spinner("Tuning the forecast for this view…"):
                    best = run_autofit(df, view, today_ts, pipeline_path(), mw0)
                if best is not None:
                    logger.info(
                        "Autofit [%s]: alpha=%.2f beta=%.2f phi=%.2f "
                        "(MAE %.2f vs %.2f with file defaults)",
                        view, best["alpha"], best["beta"], best["phi"],
                        best["mae"], best["baseline_mae"],
                    )
                    st.session_state["autofit_params"] = {
                        **best, "model": pipeline_path(),
                        "view": view, "today": today_str,
                    }
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
                st.success(f"Forecast auto-tuned for this view{pct}.")

    # ----- Compute (with a progress bar) -----------------------------------
    # The forecast is cached in session_state and only (re)built when:
    #   * there is no result yet (first load), or
    #   * a structural input changed (view / model / snapshot / data / prices), or
    #   * autofit produced new parameters (it sets _do_recompute).
    price_marker = None if prices is None else int(len(prices))
    structural_sig = (
        view, pipeline_path(), today_str, price_marker, n_excluded_rows
    )

    do_recompute = st.session_state.pop("_do_recompute", False)
    stored = st.session_state.get("fc_result")
    need_compute = (
        stored is None
        or st.session_state.get("fc_structural") != structural_sig
        or do_recompute
    )

    if need_compute:
        prog = st.progress(0.0, text="Preparing…")
        try:
            prog.progress(0.15, text="Building forecast for this view…")
            summary, weekly, agg = compute_view(
                df, view, today_ts, pipeline_path(),
                prices, alpha, beta, phi, min_weeks,
            )

            by_cust = None
            region_all = region_from_view(view)
            is_combined = view == ALL_CUSTOMERS_VIEW or region_all is not None
            if is_combined and summary is not None and not summary.empty:
                def _bump(done, total, group):
                    frac = 0.4 + 0.55 * (done / max(total, 1))
                    prog.progress(
                        min(frac, 0.98),
                        text=f"Per-customer forecast… ({done}/{total})",
                    )
                # A region rollup breaks out only its own region's groups.
                src = df if region_all is None else _region_frame(df, P, region_all)
                by_cust = compute_by_customer(
                    src, today_ts, pipeline_path(),
                    prices, alpha, beta, phi, min_weeks, progress_cb=_bump,
                )
            prog.progress(1.0, text="Done")
        finally:
            prog.empty()

        st.session_state["fc_result"] = (summary, weekly, agg, by_cust)
        st.session_state["fc_structural"] = structural_sig
    else:
        summary, weekly, agg, by_cust = stored

    if summary is None or summary.empty:
        st.error(
            f"No POS or Orders in the 8-week window for **{view}** — "
            "nothing to forecast."
        )
        st.stop()

    # ----- Header / windows -------------------------------------------------
    st.subheader(view)
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
    hist_span = (
        f"**Historical window** &nbsp; {hist_start.date()} → {lcw.date()} "
        f"<span style='color:#64748b'>({n_hist_weeks} completed {week_word})</span>"
    )
    w1.markdown(hist_span, unsafe_allow_html=True)
    fc_weeks = pd.to_datetime(weekly["WeekDate"])
    w2.markdown(
        f"**Forecast window** &nbsp; {ffw.date()} → "
        f"{fc_weeks.max().date()} "
        f"<span style='color:#64748b'>({fc_weeks.nunique()} weeks)</span>",
        unsafe_allow_html=True,
    )

    # ----- KPIs -------------------------------------------------------------
    _render_kpis(summary, agg, (lb, lcw, ffw))

    # ----- Aggregate chart --------------------------------------------------
    # Per-chart date-range picker (own key => independent from the SKU chart).
    agg_ctrl, _ = st.columns([1, 2])
    with agg_ctrl:
        agg_range = chart_range_control(agg, weekly, lcw, key="range_agg")
    st.plotly_chart(
        aggregate_chart(agg, summary, weekly, (lb, lcw, ffw), view, date_range=agg_range),
        width="stretch",
    )

    # ----- Per-SKU detail ---------------------------------------------------
    st.markdown("### SKU detail")
    skus = summary["SKU"].astype(str).tolist()
    sku = st.selectbox("SKU", skus, help="Type to search")
    row = summary.loc[summary["SKU"].astype(str) == sku].iloc[0]
    desc = row["Description"] if isinstance(row["Description"], str) else ""
    source = row["Data Source"] if "Data Source" in summary.columns else "POS"

    cL, cR = st.columns([3, 1])
    with cL:
        # Per-chart date-range picker (own key => independent from the aggregate chart).
        sku_range = chart_range_control(agg, weekly, lcw, key="range_sku")
        st.plotly_chart(
            sku_chart(sku, desc, source, agg, weekly, (lb, lcw, ffw), date_range=sku_range),
            width="stretch",
        )
    with cR:
        st.metric("Data Source", source)
        avg_col = resolve_avg_col(summary)
        phrase = avg_window_phrase(avg_col)
        window_label = "All-Time" if phrase == "All-History" \
            else phrase.replace(" Week", "-Week")
        st.metric(
            f"{window_label} Historical Demand (avg/wk)",
            f"{row[avg_col]:,.1f}",
        )
        sysv = row.get("Initial Projection Average")
        st.metric(
            "Initial Forecast (avg/wk)",
            "—" if pd.isna(sysv) else f"{sysv:,.0f}",
        )
        st.metric(
            "Updated Forecast (avg/wk)",
            f"{row['Updated Projection Average']:,.0f}",
        )
        st.metric(
            "Projection Difference (avg/wk)",
            f"{row['Projection Difference']:+,.0f}"
            if pd.notna(row["Projection Difference"]) else "—",
        )
        if RISK_COL in summary.columns:
            pv = row.get(PRICE_COL)
            rv = row.get(RISK_COL)
            st.metric("List Price", fmt_dollar(pv, decimals=2))
            st.metric(
                "Revenue Risk (avg/wk)",
                fmt_dollar(rv, signed=True),
                help="Projection difference × list price.",
            )
            prv = pv * row["Updated Projection Average"] if pd.notna(pv) else None
            st.metric(
                "Projected Revenue (avg/wk)",
                fmt_dollar(prv),
                help="List price × updated weekly-avg forecast — the gross value "
                     "at list price of this SKU's forecasted weekly demand.",
            )
        if "Top Volume Customer Groups" in summary.columns:
            st.markdown("**Top Volume Groups**")
            st.caption(row["Top Volume Customer Groups"])

    # ----- Summary table ----------------------------------------------------
    st.markdown("### Summary table by SKU")
    summary_table = summary
    if RISK_COL in summary.columns and summary[RISK_COL].notna().any():
        # Largest revenue risk first, by magnitude (a big drop is as much a
        # "risk" as a big gain); SKUs with no price (blank risk) sort to the end.
        summary_table = (
            summary.assign(_abs_risk=summary[RISK_COL].abs())
            .sort_values("_abs_risk", ascending=False, na_position="last")
            .drop(columns="_abs_risk")
            .reset_index(drop=True)
        )
        st.caption("Ordered by largest revenue risk (by magnitude); blanks last.")
    render_filtered_table(summary_table, "filter_by_sku", P)
    st.download_button(
        "⬇️ Download the summary table by SKU",
        data=view_to_excel(summary_table, weekly),
        file_name=f"{view.replace('/', '-').replace(' ', '_')}"
                  f"_demand_projections_{today_str}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_by_sku",
    )

    # ----- Summary table by SKU and Customer (ALL CUSTOMERS view only) ------
    # Mirrors the pipeline's ALL_CUSTOMERS_demand_projections file: every SKU
    # broken out by customer group. Computed alongside the main forecast in the
    # recompute block above (and cached in session_state) so it stays on the
    # same snapshot / prices / parameters as the SKU table.
    if view == ALL_CUSTOMERS_VIEW or region_from_view(view) is not None:
        st.markdown("### Summary table by SKU and Customer")
        if by_cust is None or by_cust.empty:
            st.info("No per-customer forecasts to show for this snapshot.")
        else:
            if RISK_COL in by_cust.columns and by_cust[RISK_COL].notna().any():
                # Keep each SKU's customers together; within a SKU show the
                # largest revenue risk (by magnitude) first, blanks last.
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
                st.caption(
                    "Each SKU broken out by customer group; within a SKU, "
                    "largest revenue risk first (by magnitude)."
                )
            else:
                by_cust_table = (
                    by_cust.sort_values(["SKU", "Customer Grouping"])
                    .reset_index(drop=True)
                )
                st.caption("Each SKU broken out by customer group.")
            render_filtered_table(by_cust_table, "filter_by_customer", P)
            st.download_button(
                "⬇️ Download the summary table by SKU and Customer",
                data=summary_to_excel(by_cust_table),
                file_name=(
                    f"{view.replace('/', '-').replace(' ', '_')}"
                    f"_demand_projections_{today_str}.xlsx"
                    if view != ALL_CUSTOMERS_VIEW
                    else f"ALL_CUSTOMERS_demand_projections_{today_str}.xlsx"
                ),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_by_customer",
            )

    # ----- Excluded: active products projected in non-active regions --------
    render_inactive_section(
        view, region, check_ran, inactive_df,
        excluded_counts_by_key, n_excluded_rows, today_str,
    )

    # ----- Active products MISSING projections in regions they ARE active in --
    # Uses the warehouse projection grid (sidebar), not the demand file.
    missing_df = compute_missing_projections(warehouse_df, plytix_df, df, P)
    # Per-(customer group, SKU) data source to label the missing table. The
    # by-customer table (ALL CUSTOMERS view) carries every group; otherwise the
    # single-group summary does.
    cust_source = customer_source_map(by_cust) or customer_source_map(summary)
    render_missing_section(
        view, region, warehouse_df, check_ran, missing_df, today_str,
        cust_source, P,
    )

    # ----- Active SKUs (incl. Parts) MISSING POS/Orders data where active ----
    # Uses the demand file's full history (not the warehouse grid), so gone-silent
    # channels and prolonged stockouts surface even with no recent data.
    missing_pos_df = compute_missing_pos_orders(df, plytix_df, P, anchors=(lb, lcw, ffw))
    render_missing_pos_section(view, region, missing_pos_df, today_str)

    # ----- Discontinued/inactive products with projections ------------------
    render_discontinued_section(
        view, region, disc_check_ran, discontinued_df, today_str,
    )


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
