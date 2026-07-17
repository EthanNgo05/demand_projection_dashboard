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
# Configuration                                                               #
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
# Repo root (parent of src/) — where the data folders raw_inputs/, outputs/ and
# logs/ live. HERE is used for sibling CODE (models/, extract script); REPO_ROOT
# for DATA that stays at the repo root.
REPO_ROOT = os.path.dirname(HERE)

# The forecasting models on offer. Each entry maps the label shown in the
# sidebar toggle to the pipeline file implementing it. When a DEMAND_PIPELINE
# env var is set, that file is offered as an extra option and is the default.
MODEL_OPTIONS = {
    "8-Week Moving Average": os.path.join(HERE, "models/regression.py"),
    "Holt's (double) exponential smoothing": os.path.join(HERE, "models/exponential_smoothing.py"),
    "Holt-Winters (triple) exponential smoothing": os.path.join(HERE, "models/holt_winters.py"),
    "XGBoost": os.path.join(HERE, "models/xgboost.py"),
    "TSB (intermittent demand)": os.path.join(HERE, "models/tsb.py"),
}
_ENV_PIPELINE = os.environ.get("DEMAND_PIPELINE")
if _ENV_PIPELINE:
    MODEL_OPTIONS = {"Custom (DEMAND_PIPELINE)": _ENV_PIPELINE, **MODEL_OPTIONS}

# Only offer models whose file is actually present in this deployment.
MODEL_OPTIONS = {k: v for k, v in MODEL_OPTIONS.items() if os.path.exists(v)}
DEFAULT_MODEL = next(iter(MODEL_OPTIONS), None)


def pipeline_path():
    """Path of the currently selected pipeline (the sidebar model toggle).

    Falls back to the first available model before the toggle has rendered
    (or if session state holds a label that no longer exists).
    """
    choice = st.session_state.get("model_choice", DEFAULT_MODEL)
    if choice not in MODEL_OPTIONS:
        choice = DEFAULT_MODEL
    if choice is None:
        raise FileNotFoundError(
            "No forecasting pipeline found — expected "
            "models/exponential_smoothing.py, models/holt_winters.py, "
            "models/xgboost.py, models/tsb.py or models/regression.py next to "
            "dashboard.py (or set the DEMAND_PIPELINE env var)."
        )
    return MODEL_OPTIONS[choice]

ALL_CUSTOMERS_VIEW = "All customers (combined)"

# A combined view that, unlike ALL_CUSTOMERS_VIEW (one model over all SKUs
# summed), forecasts each customer group with ITS OWN backtest-winning model —
# the model published in that group's agent_summary_<group>.json — and stitches
# every group's per-SKU rows into one table with a "Model Used" column. It is the
# "best model per group, combined" table, so it depends on the agent batch having
# run for every group (the "Agent Summary (all views)" button / `agent.batch`).
BEST_MODEL_COMBINED_VIEW = "Combined (best model per group)"
MODEL_USED_COL = "Model Used"

# Per-region rollup views: "All Customers - <region label>" combines every
# customer group in one region into a single forecast (e.g.
# "All Customers - AU (ACR)" = Web Sales - AU + Others - AU). The region is
# embedded in the view string so everything keyed on `view` — cache keys,
# session signatures, headers, filename mangling, the agent's summary path —
# works unchanged. Mirrored in agent/config.py (must match exactly).
REGION_ALL_PREFIX = "All Customers - "


def region_all_view(region):
    """The synthetic per-region combined view string for ``region``."""
    return f"{REGION_ALL_PREFIX}{region}"


def region_from_view(view):
    """Region label if ``view`` is a per-region rollup, else None."""
    if isinstance(view, str) and view.startswith(REGION_ALL_PREFIX):
        return view[len(REGION_ALL_PREFIX):]
    return None


def _region_frame(df, P, region):
    """Rows of ``df`` whose customer group belongs to ``region``.

    str() on region_for_group: a custom pipeline may return non-string labels
    (see the key=str note in the sidebar), and the view string the region was
    parsed from was built from the str form.
    """
    groups = df["Customer Grouping"].map(lambda g: str(P.region_for_group(g)))
    return df[groups == region]

# The warehouse regions we check "Active in" against. A SKU should only be
# projected in a region it is "Active in" (per the Plytix export); a projection
# in any other region is flagged and excluded from the forecast (see the
# inactive-projections logic below, ported from inactive_projections.ipynb).
WAREHOUSE_REGIONS = ["AU", "CA", "EU", "JP", "US"]

# Column names produced by the pipeline's fit_regression when list prices are
# supplied (see DISPLAY_NAMES in the pipeline). Kept here so the dashboard can
# format / sort on them.
PRICE_COL = "List Price (USD)"
RISK_COL = "Revenue Risk (avg/wk)"


def fmt_dollar(v, decimals=0, signed=False):
    """Format a dollar amount with the sign OUTSIDE the $ (e.g. -$500, +$500).

    Python's ``{:+,.0f}`` puts the sign after the ``$`` (``$-500``); this keeps
    it in front so negatives read like ``-$500``.
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    sign = "-" if v < 0 else ("+" if signed else "")
    return f"{sign}${abs(v):,.{decimals}f}"

# Chart palette -- actuals are the anchor (solid), the two projections are
# de-emphasised dashed/dotted lines so the eye reads "history -> forecast".
C_ACTUAL = "#2563eb"   # blue   - historical actual demand (POS or Orders)
C_UPDATED = "#ea580c"  # orange - our recomputed 15-week forecast
C_ORIGINAL = "#9ca3af"   # grey   - the existing projection
C_GRID = "rgba(148,163,184,0.18)"


# --------------------------------------------------------------------------- #
# Pipeline loading + data layer (pure cores + cached wrappers)                #
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _load_pipeline_cached(path, mtime):
    """Import the forecasting pipeline module by file path.

    ``mtime`` is part of the cache key so that pushing an updated pipeline
    file invalidates the cached module instead of serving a stale copy.
    """
    spec = importlib.util.spec_from_file_location("demand_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_pipeline(path):
    """Load the pipeline module, re-importing whenever the file changes."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pipeline not found at {path}")
    return _load_pipeline_cached(path, os.path.getmtime(path))


def _supports_prices(P):
    """True if this pipeline's fit_regression accepts a list_prices argument.

    Lets the dashboard stay compatible with an older pipeline that predates the
    revenue-risk columns: if the argument isn't supported we simply skip it.
    """
    try:
        return "list_prices" in inspect.signature(P.fit_regression).parameters
    except (TypeError, ValueError):
        return False


def _supports_smoothing(P):
    """True if this pipeline's fit_regression accepts alpha/beta/phi arguments.

    Lets the sidebar smoothing sliders stay compatible with an older pipeline
    whose fit_regression predates the per-call ALPHA/BETA/PHI override: if the
    arguments aren't supported we skip them and the pipeline's own module-level
    constants apply instead.
    """
    try:
        params = inspect.signature(P.fit_regression).parameters
    except (TypeError, ValueError):
        return False
    return {"alpha", "beta", "phi"} <= set(params)


def _supports_min_weeks(P):
    """True if this pipeline's fit_regression accepts a min_weeks_for_trend arg.

    Guards the sidebar's min-weeks slider so the dashboard still runs against a
    pipeline that predates the short-history flat-forecast guard: if the argument
    isn't supported we skip it and the pipeline's own MIN_WEEKS_FOR_TREND applies.
    """
    try:
        params = inspect.signature(P.fit_regression).parameters
    except (TypeError, ValueError):
        return False
    return "min_weeks_for_trend" in params


def _supports_autofit(P):
    """True if this pipeline can grid-search its own smoothing parameters.

    The Holt pipeline exposes ``autofit_smoothing`` (a backtest over an
    alpha/beta/phi grid). Pipelines without it simply don't get the button.
    """
    return callable(getattr(P, "autofit_smoothing", None))


def _raw_dir():
    """Resolve the folder holding the raw + price files (agent/data_io.py)."""
    return data_io._raw_dir(load_pipeline(pipeline_path()))


def raw_glob():
    """Build the raw-file glob, tracking the pipeline's RAW_INPUTS_FOLDER."""
    return data_io.raw_glob(load_pipeline(pipeline_path()))


def price_glob():
    """Build the list-price glob, mirroring the pipeline's LIST_PRICE_GLOB."""
    return data_io.price_glob(load_pipeline(pipeline_path()))


def discover_price_file():
    """Newest list-price file in the raw folder, or None if there isn't one."""
    return data_io.discover_price_file(load_pipeline(pipeline_path()))


_date_from_name = data_io._date_from_name


def discover_raw_files():
    """Return [(date_str, path)] newest first, mirroring resolve_input_file()."""
    return data_io.discover_raw_files(load_pipeline(pipeline_path()))


# --------------------------------------------------------------------------- #
# Manual data-warehouse refresh                                               #
# --------------------------------------------------------------------------- #
# The demand snapshot is normally refreshed by a nightly scheduled task that
# runs extract_demand_details.py (the ~10-minute SQL pull) OUT of the request
# path, so the dashboard always serves a recent file instantly. This button lets
# a user force a fresh pull on demand WITHOUT blocking the page: it launches the
# extract as a detached background process and drops a lock file in the snapshot
# folder. While the lock is live the page stays fully usable on the current
# snapshot; when the child writes the new (atomic) workbook, the snapshot
# dropdown auto-selects it. The lock also stops a manual click and the nightly
# task from overlapping into two concurrent 10-minute queries.
EXTRACT_SCRIPT = os.path.join(HERE, "extract_demand_details.py")


def _refresh_log_path():
    """Today's refresh log: ``logs/<date>/logs_refresh.txt``. Computed per call
    (not at import) so a long-running dashboard files each refresh under the day
    it ran, and shares the exact file the scheduled task writes."""
    return dated_log_path("logs_refresh.txt")
# A pull older than this with no new file is treated as crashed, so the button
# re-enables instead of wedging the UI forever. Comfortably above the ~10-minute
# typical runtime and the extract's own 900s SQL_QUERY_TIMEOUT default.
REFRESH_STALE_SECONDS = 30 * 60


def _refresh_lock_path():
    """Lock file marking an in-flight DW pull, kept in the snapshot folder.

    Lives inside the raw folder (not matched by the ``all_demand_projections_*``
    glob, so it never shows up as a snapshot) so a click and the nightly task
    coordinate through one file regardless of which one started the pull.
    """
    return os.path.join(_raw_dir(), ".refresh.lock")


def _clear_lock(lock_path):
    """Remove a refresh lock, ignoring the case where it's already gone."""
    try:
        os.remove(lock_path)
    except OSError:
        pass


def _refresh_state(lock_path, completed_since, label):
    """Shared lock state-machine: (running, started_str) for a background pull.

    Self-healing, so no process has to clean up after itself:
      * ``completed_since(lock_mtime)`` says whether the pull's output has
        landed since the lock appeared — if so, clear the lock and report idle.
      * If the lock is older than REFRESH_STALE_SECONDS with no output, the
        pull crashed/was killed — clear the lock so the button re-enables.

    What "output has landed" means differs per pull (the demand snapshot is one
    atomic workbook; a warehouse snapshot is a five-file set), which is exactly
    the ``completed_since`` seam.
    """
    if not os.path.exists(lock_path):
        return False, None
    lock_mtime = os.path.getmtime(lock_path)

    if completed_since(lock_mtime):
        _clear_lock(lock_path)
        return False, None

    if time.time() - lock_mtime > REFRESH_STALE_SECONDS:
        logger.warning("%s refresh lock is stale (>%ds); clearing it.",
                       label, REFRESH_STALE_SECONDS)
        _clear_lock(lock_path)
        return False, None

    try:
        with open(lock_path, encoding="utf-8") as f:
            started = f.read().strip()
    except OSError:
        started = ""
    return True, started


def refresh_in_progress():
    """(running, started_str): is a background DW pull active, and when it began.

    Completion = any demand snapshot written since the lock appeared (the
    extract writes one atomic workbook, so the first newer file IS the result).
    """
    def _completed(lock_mtime):
        files = discover_raw_files()
        return bool(files) and max(
            os.path.getmtime(p) for _, p in files
        ) >= lock_mtime

    return _refresh_state(_refresh_lock_path(), _completed, "DW")


def start_refresh(incremental: bool = True):
    """Launch extract_demand_details.py in the background. Returns (ok, message).

    Reuses THIS interpreter (``sys.executable``) so the pull runs in the same
    venv the dashboard was started with, and inherits the environment (the SQL_*
    connection vars). ``DEMAND_RAW_DIR`` is pinned to the exact folder the
    dashboard reads so the child writes where we look, regardless of CWD. The
    child's output is appended to logs/<date>/logs_refresh.txt for diagnosis.

    ``incremental`` (the default) pulls only the last few weeks of actuals plus
    all forward projections and merges them into the newest snapshot — minutes
    instead of the ~20-minute full pull. The nightly scheduled task still runs
    the full pull as the self-healing baseline.
    """
    running, started = refresh_in_progress()
    if running:
        return False, f"A refresh is already running (started {started})."

    raw_dir = _raw_dir()
    os.makedirs(raw_dir, exist_ok=True)
    mode = "incremental" if incremental else "full"
    return _launch_refresh(
        _refresh_lock_path(),
        EXTRACT_SCRIPT,
        ["--incremental"] if incremental else [],
        {"DEMAND_RAW_DIR": raw_dir},
        f"DW refresh ({mode})",
    )


def _launch_refresh(lock_path, script, extra_args, env_overrides, header):
    """Write ``lock_path``, then launch ``script`` detached. Returns (ok, msg).

    The lock is written BEFORE launching so a double-click can't spawn two
    pulls. The child reuses THIS interpreter (``sys.executable``) so it runs in
    the same venv the dashboard was started with and inherits the environment
    (the SQL_* connection vars) plus ``env_overrides`` (the raw-dir pin, so the
    child writes exactly where the dashboard looks, regardless of CWD). Output
    is appended to logs/<date>/logs_refresh.txt for diagnosis.
    """
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(now)

    try:
        env = {**os.environ, **env_overrides}
        # Detach on Windows so the pull outlives this Streamlit run/rerun and
        # isn't tied to the parent's console.
        creationflags = 0
        if os.name == "nt":
            creationflags = (
                subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            )
        logf = open(_refresh_log_path(), "a", encoding="utf-8")
        try:
            logf.write(f"\n===== {header} started {now} =====\n")
            logf.flush()
            subprocess.Popen(
                [sys.executable, script] + extra_args,
                cwd=HERE,
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
                close_fds=True,
            )
        finally:
            # The child holds its own duplicated handle; drop ours so the parent
            # doesn't leak a file handle per click.
            logf.close()
    except Exception as exc:
        _clear_lock(lock_path)
        logger.exception("Failed to launch %s", header)
        return False, f"Could not start refresh: {exc}"

    logger.info("%s launched (%s)", header, now)
    return True, now


# --------------------------------------------------------------------------- #
# Manual warehouse-projections refresh                                        #
# --------------------------------------------------------------------------- #
# Same lock-file coordination as the demand refresh above, with one twist: a
# warehouse snapshot is FIVE region files written back-to-back, not one atomic
# workbook. Each file is atomic, but the set is not — so completion means "the
# newest dated group holds every region, all newer than the lock", not "any
# newer file exists" (which would clear the lock after the first region lands
# and briefly serve a partial snapshot).
WAREHOUSE_EXTRACT_SCRIPT = os.path.join(HERE, "extract_warehouse_projections.py")


def _wh_refresh_lock_path():
    """Lock for an in-flight warehouse pull, in the warehouse snapshot folder
    (not matched by the ``*.xlsx`` discovery glob)."""
    return os.path.join(data_io._warehouse_dir(), ".refresh.lock")


def _wh_snapshot_complete_since(lock_mtime):
    """True once a full 5-region snapshot newer than the lock exists."""
    groups = data_io.discover_warehouse_files()
    if not groups:
        return False
    newest = [
        p for p in next(iter(groups.values())) if data_io._warehouse_region(p)
    ]
    regions = {data_io._warehouse_region(p) for p in newest}
    if not set(data_io.REGION_PREFIXES) <= regions:
        return False
    return all(os.path.getmtime(p) >= lock_mtime for p in newest)


def warehouse_refresh_in_progress():
    """(running, started_str): is a background warehouse pull active."""
    return _refresh_state(
        _wh_refresh_lock_path(), _wh_snapshot_complete_since, "Warehouse"
    )


def start_warehouse_refresh():
    """Launch extract_warehouse_projections.py in the background; (ok, msg)."""
    running, started = warehouse_refresh_in_progress()
    if running:
        return False, f"A warehouse refresh is already running (started {started})."

    wh_dir = data_io._warehouse_dir()
    os.makedirs(wh_dir, exist_ok=True)
    return _launch_refresh(
        _wh_refresh_lock_path(),
        WAREHOUSE_EXTRACT_SCRIPT,
        [],
        {"WAREHOUSE_RAW_DIR": wh_dir},
        "Warehouse refresh",
    )


# --------------------------------------------------------------------------- #
# Precompute every view's agent summary (agent.batch)                          #
# --------------------------------------------------------------------------- #
# The "Run Agent Summary" button above runs the agent for ONE view live. This
# section runs it for EVERY view (the same work as `python -m agent.batch`),
# which backtests all models across ~60 views and can take up to an hour. It
# reuses the demand-refresh pattern — a detached background process plus a lock
# file — so the page stays usable while it runs; the batch writes each
# outputs/agent_summary_<view>.json exactly as the nightly job does. We also keep
# the Popen handle in session_state so the current session detects completion
# promptly (the lock's stale timeout is only the cross-restart fallback).
BATCH_STALE_SECONDS = 90 * 60  # generous: a full LLM batch can approach an hour.


def _batch_lock_path():
    """Lock file marking an in-flight all-views batch, kept under outputs/."""
    return os.path.join(REPO_ROOT, "outputs", ".agent_batch.lock")


def _batch_log_path():
    """Today's batch log: ``logs/<date>/logs_agent_batch.txt`` (computed per call
    so a long-running dashboard files each run under the day it ran)."""
    return dated_log_path("logs_agent_batch.txt")


def _batch_result_line():
    """The last 'Done: N ok, M failed …' line from the batch log, or None.

    Lets the completion toast report the outcome without the batch signalling
    back into this process. Best-effort: any read error just yields None.
    """
    try:
        with open(_batch_log_path(), encoding="utf-8") as f:
            done = [ln.strip() for ln in f if ln.strip().startswith("Done:")]
        return done[-1] if done else None
    except OSError:
        return None


def batch_in_progress():
    """(running, started_str): is the all-views batch active, and when it began.

    Self-healing like _refresh_state, but completion is detected two ways:
      * the Popen handle we kept this session has exited (prompt, same-session), or
      * the lock is older than BATCH_STALE_SECONDS (cross-restart / crash fallback).
    """
    lock_path = _batch_lock_path()
    if not os.path.exists(lock_path):
        return False, None
    lock_mtime = os.path.getmtime(lock_path)

    proc = st.session_state.get("agent_batch_proc")
    if proc is not None and proc.poll() is not None:
        _clear_lock(lock_path)
        return False, None

    if time.time() - lock_mtime > BATCH_STALE_SECONDS:
        logger.warning("Agent-batch lock is stale (>%ds); clearing it.",
                       BATCH_STALE_SECONDS)
        _clear_lock(lock_path)
        return False, None

    try:
        with open(lock_path, encoding="utf-8") as f:
            started = f.read().strip()
    except OSError:
        started = ""
    return True, started


def start_agent_batch(provider):
    """Launch `python -m agent.batch` in the background. Returns (ok, message).

    Reuses THIS interpreter/venv and inherits the environment; ``provider`` pins
    the reasoning LLM (LLM_PROVIDER) for the run. The lock is written BEFORE
    launching so a double-click can't spawn two batches. Output is appended to
    logs/<date>/logs_agent_batch.txt; the Popen handle is stored in session_state
    so this session detects completion without waiting for the stale timeout.
    """
    running, started = batch_in_progress()
    if running:
        return False, f"An agent batch is already running (started {started})."

    lock_path = _batch_lock_path()
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(now)

    try:
        env = {**os.environ, "LLM_PROVIDER": provider}
        creationflags = 0
        if os.name == "nt":
            creationflags = (
                subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            )
        logf = open(_batch_log_path(), "a", encoding="utf-8")
        try:
            logf.write(f"\n===== Agent batch (all views) started {now} =====\n")
            logf.flush()
            # `-m agent.batch` resolves because cwd=HERE is src/ (agent is a
            # package under src/). Provider is also passed as a flag so a stale
            # env can't override it.
            proc = subprocess.Popen(
                [sys.executable, "-m", "agent.batch", "--provider", provider],
                cwd=HERE,
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
                close_fds=True,
            )
        finally:
            logf.close()
    except Exception as exc:
        _clear_lock(lock_path)
        logger.exception("Failed to launch agent batch")
        return False, f"Could not start agent batch: {exc}"

    st.session_state["agent_batch_proc"] = proc
    st.session_state["agent_batch_started"] = now
    logger.info("Agent batch launched (%s, provider=%s)", now, provider)
    return True, now


# --------------------------------------------------------------------------- #
# Plytix-based SKU exclusions live in agent/data_io.py (streamlit-free) so the #
# dashboard and the agent's ingest node drop the EXACT same rows before        #
# forecasting: a SKU is never projected or flagged when it is discontinued/    #
# inactive, or in a region it is not "Active in" (see data_io.apply_exclusions #
# and its use in main()). The aliases keep the dashboard's call sites          #
# unchanged; the cached readers stay here because @st.cache_data is            #
# Streamlit-only.                                                              #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Reading Plytix export…")
def read_plytix_from_path(path, _mtime):
    """Read the raw Plytix export from disk (for the 'Active in' check)."""
    return data_io.read_plytix(path)


@st.cache_data(show_spinner="Reading Plytix export…")
def read_plytix_from_bytes(_data, name):
    """Read the raw Plytix export from uploaded bytes (for the 'Active in' check)."""
    return data_io.read_plytix(BytesIO(_data))


@st.cache_data(show_spinner="Fetching Plytix feed…")
def fetch_plytix_from_url(url, _nonce):
    """Fetch the raw Plytix export from the channel feed URL (CSV).

    ``_nonce`` busts the cache when the user clicks "Refresh from Plytix" — a URL
    has no mtime to key on. Returns the raw Plytix frame; list prices are derived
    from it cheaply via ``data_io.prices_from_plytix``."""
    return data_io.read_plytix(url)


# Filter logic + constants live in agent/data_io.py (single source of truth);
# these aliases keep the dashboard's existing call sites unchanged.
WAREHOUSE_REGIONS = data_io.WAREHOUSE_REGIONS
INACTIVE_COLS = data_io.INACTIVE_COLS
DISCONTINUED_COLS = data_io.DISCONTINUED_COLS
MISSING_COLS = data_io.MISSING_COLS
_this_week_start = data_io._this_week_start
_active_in_list = data_io._active_in_list
_region_code = data_io._region_code
compute_active_products = data_io.compute_active_products
compute_inactive_projections = data_io.compute_inactive_projections
compute_discontinued_products = data_io.compute_discontinued_products
compute_discontinued_projections = data_io.compute_discontinued_projections
compute_missing_projections = data_io.compute_missing_projections


# Cleaning lives in agent/data_io.py (shared with the agent's ingest node);
# the alias keeps the dashboard's internal call sites unchanged.
_clean = data_io._clean


@st.cache_data(show_spinner="Loading raw data…")
def load_raw_from_path(path, _mtime, model_path):
    """Read + clean a raw file from disk. ``_mtime`` busts the cache on change.

    ``model_path`` keys the cache on the selected model, since each pipeline
    owns its own cleaning rules.
    """
    P = load_pipeline(model_path)
    raw = data_io.read_raw_frame(path)  # Parquet sidecar when present, else xlsx
    return _clean(raw, P)


@st.cache_data(show_spinner="Loading raw data…")
def load_raw_from_bytes(_data, name, model_path):
    """Read + clean an uploaded raw file (cached on its bytes + model)."""
    P = load_pipeline(model_path)
    raw = pd.read_excel(BytesIO(_data), header=2)
    return _clean(raw, P)


@st.cache_data(show_spinner="Cleaning warehouse projections…")
def load_warehouse_from_paths(paths, _mtimes):
    """Clean + combine warehouse exports from disk into one long frame.

    ``_mtimes`` (a tuple aligned with ``paths``) busts the cache when any file
    changes. Used by the 'missing future projections' table only.
    """
    return data_io.combine_warehouse_projections([(p, p) for p in paths])


@st.cache_data(show_spinner="Cleaning warehouse projections…")
def load_warehouse_from_uploads(items):
    """Clean + combine uploaded warehouse exports (cached on their bytes).

    ``items`` is a tuple of (name, bytes) pairs, one per uploaded file.
    """
    return data_io.combine_warehouse_projections(
        [(BytesIO(data), name) for name, data in items]
    )


@st.cache_data(show_spinner="Loading list prices…")
def load_prices_from_path(path, _mtime, model_path):
    """Load a SKU -> List Price (USD) Series from disk. ``_mtime`` busts cache."""
    P = load_pipeline(model_path)
    if not hasattr(P, "load_list_prices"):
        return None
    return P.load_list_prices(path)


@st.cache_data(show_spinner="Loading list prices…")
def load_prices_from_bytes(_data, name, model_path):
    """Load list prices from an uploaded workbook (cached on its bytes).

    Writes to a temp file so the pipeline's own reader/cleaner is reused
    (keeping a single source of truth for how prices are parsed).
    """
    P = load_pipeline(model_path)
    if not hasattr(P, "load_list_prices"):
        return None

    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tf:
        tf.write(_data)
        tmp = tf.name
    try:
        return P.load_list_prices(tmp)
    finally:
        os.remove(tmp)


def list_views(df):
    """Group views organised by region, plus the combined ALL CUSTOMERS view."""
    P = load_pipeline(pipeline_path())
    groups = sorted(df["Customer Grouping"].dropna().unique().tolist())
    by_region = {}
    for g in groups:
        by_region.setdefault(P.region_for_group(g), []).append(g)
    return by_region


@st.cache_data(show_spinner="Building forecast…")
def compute_view(df, view, today_ts, model_path, prices=None, alpha=None,
                 beta=None, phi=None, min_weeks=None):
    """Recompute summary + weekly + per-week aggregate for the selected view.

    Returns (summary_df, weekly_df, agg_frame) where agg_frame is the SKU-week
    POS/Orders/Projection table (used to draw historical actuals and the original
    projection). For ALL CUSTOMERS the breakdown is included so the summary
    carries 'Top Volume Customer Groups'. When ``prices`` (a SKU -> price Series)
    is supplied and the pipeline supports it, the summary also carries
    'List Price (USD)' and 'Revenue Risk (avg/wk)'. ``alpha`` / ``beta`` / ``phi``,
    when given, override the pipeline's smoothing constants for this call, and
    ``min_weeks`` overrides MIN_WEEKS_FOR_TREND (all are part of the cache key, so
    moving a slider recomputes the forecast). ``model_path`` selects the
    pipeline and keys the cache, so toggling the model recomputes too.
    """
    P = load_pipeline(model_path)
    kwargs = {}
    if prices is not None and _supports_prices(P):
        kwargs["list_prices"] = prices
    if None not in (alpha, beta, phi) and _supports_smoothing(P):
        kwargs.update(alpha=alpha, beta=beta, phi=phi)
    if min_weeks is not None and _supports_min_weeks(P):
        kwargs["min_weeks_for_trend"] = min_weeks
    if view == ALL_CUSTOMERS_VIEW:
        combined_label = getattr(
            P, "ALL_CUSTOMERS_LABEL", getattr(P, "ALL_SKUS_LABEL", ALL_CUSTOMERS_VIEW)
        )
        agg = P.aggregate_to_sku_week(df)
        summary, weekly = P.fit_regression(
            agg, today_ts, grouping_label=combined_label,
            breakdown_df=df, **kwargs,
        )
    elif (region_all := region_from_view(view)) is not None:
        # Per-region rollup: every customer group in the region, combined.
        # breakdown_df mirrors the ALL CUSTOMERS branch so the summary carries
        # 'Top Volume Customer Groups' (here: the region's groups).
        sub = _region_frame(df, P, region_all)
        agg = P.aggregate_to_sku_week(sub)
        summary, weekly = P.fit_regression(
            agg, today_ts, grouping_label=view, breakdown_df=sub, **kwargs
        )
    else:
        sub = df[df["Customer Grouping"] == view]
        agg = P.aggregate_to_sku_week(sub)
        summary, weekly = P.fit_regression(
            agg, today_ts, grouping_label=view, **kwargs
        )
    return summary, weekly, agg


def view_to_excel(summary_df, weekly_df):
    """Build an in-memory .xlsx (same two-sheet layout as the pipeline output)."""
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        summary_df.to_excel(w, sheet_name="summary", index=False)
        weekly_df.to_excel(w, sheet_name="weekly_forecast", index=False)
    buf.seek(0)
    return buf.getvalue()


def summary_to_excel(summary_df, sheet_name="summary"):
    """Build an in-memory single-sheet .xlsx of a summary table.

    Used for the by-SKU-and-customer table, which mirrors the pipeline's
    ALL_CUSTOMERS_demand_projections file (a single concatenated summary sheet).
    """
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        summary_df.to_excel(w, sheet_name=sheet_name, index=False)
    buf.seek(0)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def run_autofit(df, view, today_ts, model_path, min_weeks=None):
    """Grid-search the best alpha/beta/phi for the selected view (cached).

    Builds the same SKU-week aggregate ``compute_view`` fits on, then delegates
    to the pipeline's ``autofit_smoothing`` backtest. Cached on
    (data, view, snapshot, model, min_weeks) so clicking Autofit twice — or
    returning to a view already fitted this session — is instant.
    """
    P = load_pipeline(model_path)
    if not _supports_autofit(P):
        return None
    if view == ALL_CUSTOMERS_VIEW:
        agg = P.aggregate_to_sku_week(df)
    elif (region_all := region_from_view(view)) is not None:
        agg = P.aggregate_to_sku_week(_region_frame(df, P, region_all))
    else:
        agg = P.aggregate_to_sku_week(df[df["Customer Grouping"] == view])
    kwargs = {}
    if min_weeks is not None and "min_weeks_for_trend" in inspect.signature(
        P.autofit_smoothing
    ).parameters:
        kwargs["min_weeks_for_trend"] = min_weeks
    return P.autofit_smoothing(agg, today_ts, **kwargs)


@st.cache_data(show_spinner=False)
def _forecast_one_group(df_group, today_ts, model_path, group_label,
                        prices=None, alpha=None, beta=None, phi=None,
                        min_weeks=None):
    """Forecast a single customer group's SKUs. Cached; calls NO Streamlit
    element, so it is safe to replay on a cache hit. ``group_label`` is a
    normal (hashable) argument so distinct groups get distinct cache entries.
    """
    P = load_pipeline(model_path)
    kwargs = {}
    if prices is not None and _supports_prices(P):
        kwargs["list_prices"] = prices
    if None not in (alpha, beta, phi) and _supports_smoothing(P):
        kwargs.update(alpha=alpha, beta=beta, phi=phi)
    if min_weeks is not None and _supports_min_weeks(P):
        kwargs["min_weeks_for_trend"] = min_weeks
    agg = P.aggregate_to_sku_week(df_group)
    summary, _ = P.fit_regression(
        agg, today_ts, grouping_label=group_label, **kwargs
    )
    return summary


def compute_by_customer(df, today_ts, model_path, prices=None, alpha=None,
                        beta=None, phi=None, min_weeks=None, progress_cb=None):
    """Per-(SKU, Customer Grouping) summary — the rows behind ALL_CUSTOMERS.

    The pipeline's ``ALL_CUSTOMERS_demand_projections`` file is just a
    concatenation of every per-customer-group summary sheet. This reproduces it
    live: for each Customer Grouping we run the identical per-group forecast via
    the cached ``_forecast_one_group`` helper, then stack the summaries.
    Recomputing rather than reading the saved workbook keeps this table on the
    same snapshot / prices / smoothing as the rest of the page.

    This orchestrator is intentionally NOT cached: it may call ``progress_cb``
    (which drives a progress bar), and Streamlit element calls are not allowed
    inside a cached function. Each group's forecast is cached instead, so the
    expensive work is still memoised. On plain reruns this function isn't called
    at all — the result is held in session_state (see main()).

    Returns a DataFrame in the pipeline's SUMMARY_COLUMNS order, or None if no
    group had anything to forecast.
    """
    frames = []
    groups = sorted(df["Customer Grouping"].dropna().unique().tolist())
    n_groups = len(groups)
    for i, group in enumerate(groups):
        sub = df[df["Customer Grouping"] == group]
        summary = _forecast_one_group(
            sub, today_ts, model_path, group,
            prices, alpha, beta, phi, min_weeks,
        )
        if summary is not None and not summary.empty:
            frames.append(summary)
        if progress_cb is not None:
            progress_cb(i + 1, n_groups, group)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _agent_summaries_mtime():
    """Newest mtime among outputs/agent_summary_*.json, or 0.0 if none exist.

    Folded into the combined view's cache signature so the table rebuilds
    automatically as soon as a batch (the "Agent Summary (all views)" button, the
    nightly job, or `agent.batch`) writes fresh summaries — no manual reload."""
    paths = glob.glob(os.path.join(REPO_ROOT, "outputs", "agent_summary_*.json"))
    return max((os.path.getmtime(p) for p in paths), default=0.0)


def _best_model_for_group(group):
    """(label, model_path) for a group's backtest-winning model, or None.

    Reads the group's published agent summary (agent_summary_<group>.json) and
    maps its ``best_model`` label to a MODEL_OPTIONS file path. Returns None when
    the summary is missing, has no best model, or names a label this deployment
    doesn't offer — the caller treats all three as "no summary yet".
    """
    payload = _load_agent_summary(group)
    if not payload:
        return None
    label = payload.get("best_model")
    path = MODEL_OPTIONS.get(label)
    if not label or path is None:
        return None
    return label, path


def compute_by_customer_best(df, today_ts, prices=None, min_weeks=None,
                             progress_cb=None):
    """Per-(SKU, Customer Grouping) summary using each group's BEST model.

    Like ``compute_by_customer``, but instead of one model for every group it
    forecasts each group with the model that won that group's backtest (from
    ``agent_summary_<group>.json``) and stamps a ``MODEL_USED_COL`` column. To
    match what the single-group view shows, groups whose best model supports
    autofit are tuned per group via ``run_autofit`` before forecasting.

    A group is only included if it has a resolvable best model. Groups with no
    published summary, or whose summary has no backtest winner (``best_model`` is
    null — history too short to score any model), are left OUT of the table and
    returned separately so the caller can list them.

    Returns ``(table, excluded)`` where ``table`` is a DataFrame (SUMMARY_COLUMNS
    + MODEL_USED_COL) or None when no group resolved / produced rows, and
    ``excluded`` is the sorted list of group names with no best model.
    """
    groups = sorted(df["Customer Grouping"].dropna().unique().tolist())

    # First pass: split into groups with a resolvable best model vs. those without
    # (no summary file, or a summary whose best_model is null).
    resolved = {}
    excluded = []
    for group in groups:
        best = _best_model_for_group(group)
        if best is None:
            excluded.append(group)
        else:
            resolved[group] = best
    if not resolved:
        return None, excluded

    # Second pass: forecast each resolved group with its own model (autofit when
    # supported).
    frames = []
    n = len(resolved)
    for i, (group, (label, path)) in enumerate(resolved.items()):
        sub = df[df["Customer Grouping"] == group]
        alpha = beta = phi = None
        P = load_pipeline(path)
        if _supports_autofit(P):
            fitted = run_autofit(df, group, today_ts, path, min_weeks)
            if fitted:
                alpha, beta, phi = fitted.get("alpha"), fitted.get("beta"), fitted.get("phi")
        summary = _forecast_one_group(
            sub, today_ts, path, group, prices, alpha, beta, phi, min_weeks,
        )
        if summary is not None and not summary.empty:
            summary = summary.copy()
            summary[MODEL_USED_COL] = label
            frames.append(summary)
        if progress_cb is not None:
            progress_cb(i + 1, n, group)

    if not frames:
        return None, excluded
    combined = pd.concat(frames, ignore_index=True)
    # Surface the model used right after the customer group for readability.
    if "Customer Grouping" in combined.columns:
        cols = [c for c in combined.columns if c != MODEL_USED_COL]
        pos = cols.index("Customer Grouping") + 1
        cols.insert(pos, MODEL_USED_COL)
        combined = combined[cols]
    return combined, excluded


def _render_best_model_combined(df, today_ts, today_str, prices, n_excluded_rows):
    """Render the BEST_MODEL_COMBINED_VIEW: per-group best-model table.

    Builds (and session-caches) the mixed table via ``compute_by_customer_best``,
    renders the winners table + a model-usage line + a download, and lists any
    groups that had no best model (no summary, or too little history to backtest)
    in a dropdown. Called from main() in place of the single-model page body. The
    page title is already rendered by main() before this branch, so we start at the
    section subheader to avoid showing it twice.
    """
    st.subheader("Combined — best model per customer group")
    st.caption(
        "Each customer group is forecast with its own backtest-winning model "
        "(from the latest agent summaries) and stitched into one table. The "
        "sidebar model choice does not apply to this view."
    )

    # Cache on a structural signature so search-box reruns don't rebuild it. The
    # agent-summaries mtime is part of the signature so the table rebuilds as soon
    # as a batch writes fresh summaries (e.g. right after "Agent Summary (all
    # views)" finishes) — without it a stale "run the batch first" result would
    # linger in this session until an unrelated structural change.
    price_marker = None if prices is None else int(len(prices))
    sig = (BEST_MODEL_COMBINED_VIEW, today_str, price_marker, n_excluded_rows,
           _agent_summaries_mtime())
    if st.session_state.get("bestmix_structural") != sig:
        prog = st.progress(0.0, text="Preparing…")
        try:
            def _bump(done, total, group):
                prog.progress(
                    min(0.05 + 0.93 * done / max(total, 1), 0.98),
                    text=f"Forecasting each group with its best model… "
                         f"({done}/{total})",
                )
            result = compute_by_customer_best(
                df, today_ts, prices, min_weeks=None, progress_cb=_bump,
            )
            prog.progress(1.0, text="Done")
        finally:
            prog.empty()
        st.session_state["bestmix_result"] = result
        st.session_state["bestmix_structural"] = sig
    else:
        result = st.session_state.get("bestmix_result")

    combined, excluded = result if result is not None else (None, [])

    def _render_excluded(title):
        """Dropdown listing groups left out (bullet-pointed, one per line)."""
        if not excluded:
            return
        with st.expander(f"{title} ({len(excluded)})"):
            st.caption(
                "These groups had no published summary, or too little history "
                "for any model to be backtested, so no best model could be "
                "chosen — they're left out of the table."
            )
            st.markdown("\n".join(f"- {g}" for g in excluded))

    # No group had a resolvable best model → prompt to run the batch.
    if combined is None or getattr(combined, "empty", True):
        st.warning(
            "No customer group has a backtest-winning model yet. Click **Agent "
            "Summary (all views)** in the sidebar (or run `python -m "
            "agent.batch`), then reopen this view."
        )
        _render_excluded("Groups without a best model")
        return

    # Model-usage summary: how many groups each model won.
    counts = (
        combined.drop_duplicates("Customer Grouping")[MODEL_USED_COL].value_counts()
    )
    parts = ", ".join(f"{m} ×{c}" for m, c in counts.items())
    st.caption(f"{int(counts.sum())} groups — {parts}")

    # Keep each SKU's rows together; largest revenue risk first when present.
    if RISK_COL in combined.columns and combined[RISK_COL].notna().any():
        table = (
            combined.assign(_abs=combined[RISK_COL].abs())
            .sort_values(["SKU", "_abs"], ascending=[True, False], na_position="last")
            .drop(columns="_abs").reset_index(drop=True)
        )
        st.caption("Each SKU broken out by customer group; within a SKU, "
                   "largest revenue risk first (by magnitude).")
    else:
        table = combined.sort_values(["SKU", "Customer Grouping"]).reset_index(drop=True)
        st.caption("Each SKU broken out by customer group.")

    st.dataframe(
        style_summary(search_filter(table, key="search_best_mix")),
        width="stretch", hide_index=True,
    )
    st.download_button(
        "⬇️ Download the combined best-model table",
        data=summary_to_excel(table),
        file_name=f"Combined_best_model_demand_projections_{today_str}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_best_mix",
    )

    _render_excluded("Groups excluded — no backtest-winning model")


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


# --------------------------------------------------------------------------- #
# Charts                                                                      #
# --------------------------------------------------------------------------- #
def _base_layout(fig, title, forecast_start, y_title="Units (POS / Orders)"):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        margin=dict(l=10, r=10, t=80, b=10),
        height=420,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=0.98, x=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(gridcolor=C_GRID, title=None)
    fig.update_yaxes(gridcolor=C_GRID, rangemode="tozero", title=y_title)
    if forecast_start is not None:
        fig.add_vline(
            x=forecast_start, line_width=1, line_dash="dot",
            line_color="rgba(100,116,139,0.7)",
        )
        fig.add_annotation(
            x=forecast_start, yref="paper", y=0.93, yanchor="bottom",
            text="forecast →", showarrow=False, font=dict(size=11, color="#64748b"),
            xshift=4,
        )
    return fig


def _clip_to_range(df, date_range):
    """Clip a trace frame to a chart date-range window on WeekDate (Y auto-fits).

    date_range is None (no clipping — current behavior) or a (start, end) pair of
    Timestamps. Empty frames pass through untouched.
    """
    if date_range is None or df.empty:
        return df
    s, e = date_range
    return df[(df["WeekDate"] >= s) & (df["WeekDate"] <= e)]


def chart_range_control(agg, weekly, lcw, key):
    """Compact date-range picker rendered right above a chart.

    Returns a (view_start, view_end) pair of Timestamps used to clip that chart's
    traces so its Y-axis auto-fits the visible window. Each chart gets its own
    control (unique `key`) and thus its own independent range.

    Presets trim history only — the forecast horizon always stays visible.
    "Custom…" reveals a calendar / typeable range picker.
    """
    RANGE_PRESETS = {
        "1 Month":  pd.DateOffset(months=1),
        "3 Months": pd.DateOffset(months=3),
        "6 Months": pd.DateOffset(months=6),
        "9 Months": pd.DateOffset(months=9),
        "1 Year":   pd.DateOffset(years=1),
        "2 Years":  pd.DateOffset(years=2),
        "3 Years":  pd.DateOffset(years=3),
        "All":      None,
        "Custom…":  "custom",
    }
    data_min = pd.to_datetime(agg["WeekDate"]).min()
    horizon_end = pd.to_datetime(weekly["WeekDate"]).max()

    preset = st.selectbox(
        "Date range", list(RANGE_PRESETS),
        index=list(RANGE_PRESETS).index("6 Months"),
        key=f"{key}_preset",
        help="How much history to show. The forecast always stays visible.",
    )
    if preset == "Custom…":
        default_start = max(data_min, horizon_end - pd.DateOffset(months=6))
        picked = st.date_input(
            "Custom range",
            value=(default_start.date(), horizon_end.date()),
            min_value=data_min.date(), max_value=horizon_end.date(),
            key=f"{key}_custom",
            help="Click the calendar or type dates. Pick a start and an end.",
        )
        # date_input returns a single date mid-selection; apply once both ends chosen.
        if isinstance(picked, (tuple, list)) and len(picked) == 2:
            return pd.Timestamp(picked[0]), pd.Timestamp(picked[1])
        return data_min, horizon_end
    if preset == "All":
        return data_min, horizon_end
    # Preset controls history start; forecast ALWAYS stays visible.
    return max(data_min, lcw - RANGE_PRESETS[preset]), horizon_end


def aggregate_chart(agg, summary, weekly, anchors, view, date_range=None):
    """Total actual demand (historical window) flowing into total forecast (15 wks).

    Historical demand uses each SKU's forecast source (POS or Orders) so the
    actual total is comparable to the forecast total. When date_range is given,
    the plotted traces are clipped to that window so the Y-axis rescales to fit.
    """
    lb, lcw, ffw = anchors

    hist = historical_window(agg, summary, anchors)
    hist_tot = hist.groupby("WeekDate")["demand"].sum(min_count=1).reset_index()

    fc = weekly.copy()
    fc["WeekDate"] = pd.to_datetime(fc["WeekDate"])
    fc_tot = fc.groupby("WeekDate")["projected_pos"].sum().reset_index()

    # Original projection: plot straight from the spreadsheet's Projection column
    # across the SAME span shown for actuals + forecast (history start through the
    # forecast horizon), so the grey line runs the full width of the chart rather
    # than only over the 15 forecast weeks. Weeks with no Projection are dropped
    # (the line simply connects the weeks that have a value); no recomputation.
    horizon_end = pd.to_datetime(weekly["WeekDate"]).max()
    sys_proj = agg[
        (agg["WeekDate"] >= lb) & (agg["WeekDate"] <= horizon_end)
    ].dropna(subset=["Projection"])
    sys_tot = sys_proj.groupby("WeekDate")["Projection"].sum().reset_index()

    # Clip every plotted trace to the chosen chart window so the Y-axis auto-fits
    # the visible weeks (does not affect the summary/forecast math).
    hist_tot = _clip_to_range(hist_tot, date_range)
    fc_tot = _clip_to_range(fc_tot, date_range)
    sys_tot = _clip_to_range(sys_tot, date_range)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_tot["WeekDate"], y=hist_tot["demand"], name="Actual demand",
        mode="lines+markers", line=dict(color=C_ACTUAL, width=3),
        marker=dict(size=6),
    ))
    if not hist_tot.empty and not fc_tot.empty:
        fig.add_trace(go.Scatter(
            x=[hist_tot["WeekDate"].iloc[-1], fc_tot["WeekDate"].iloc[0]],
            y=[hist_tot["demand"].iloc[-1], fc_tot["projected_pos"].iloc[0]],
            mode="lines", showlegend=False,
            line=dict(color=C_UPDATED, width=2, dash="dot"), hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=fc_tot["WeekDate"], y=fc_tot["projected_pos"], name="Updated forecast",
        mode="lines+markers", line=dict(color=C_UPDATED, width=3, dash="dash"),
        marker=dict(size=6),
    ))
    if not sys_tot.empty:
        fig.add_trace(go.Scatter(
            x=sys_tot["WeekDate"], y=sys_tot["Projection"], name="Original projection",
            mode="lines+markers", line=dict(color=C_ORIGINAL, width=2, dash="dot"),
            marker=dict(size=5),
        ))
    return _base_layout(fig, f"Total weekly demand — {view}", ffw)


def sku_chart(sku, desc, source, agg, weekly, anchors, date_range=None):
    """Per-SKU: actuals (historical window, from its source) + updated forecast + original proj.

    When date_range is given, the plotted traces are clipped to that window so the
    Y-axis rescales to fit the visible weeks.
    """
    lb, lcw, ffw = anchors
    col = "Orders" if source == "Orders" else "POS"

    a = agg[agg["SKU"].astype(str) == str(sku)].sort_values("WeekDate")
    hist = a[(a["WeekDate"] >= lb) & (a["WeekDate"] <= lcw)].dropna(subset=[col])
    # Original projection: straight from the spreadsheet's Projection column,
    # across the SAME span shown for actuals + forecast (history start through the
    # forecast horizon), so the grey line runs the full width of the chart. Weeks
    # with no Projection are dropped; no recomputation.
    horizon_end = pd.to_datetime(weekly["WeekDate"]).max()
    sys_proj = a[
        (a["WeekDate"] >= lb) & (a["WeekDate"] <= horizon_end)
    ].dropna(subset=["Projection"])

    fc = weekly[weekly["SKU"].astype(str) == str(sku)].copy()
    fc["WeekDate"] = pd.to_datetime(fc["WeekDate"])

    # Clip every plotted trace to the chosen chart window so the Y-axis auto-fits.
    hist = _clip_to_range(hist, date_range)
    fc = _clip_to_range(fc, date_range)
    sys_proj = _clip_to_range(sys_proj, date_range)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["WeekDate"], y=hist[col], name=f"Actual {source}",
        mode="lines+markers", line=dict(color=C_ACTUAL, width=3),
        marker=dict(size=7),
    ))
    if not hist.empty and not fc.empty:
        fig.add_trace(go.Scatter(
            x=[hist["WeekDate"].iloc[-1], fc["WeekDate"].iloc[0]],
            y=[hist[col].iloc[-1], fc["projected_pos"].iloc[0]],
            mode="lines", showlegend=False,
            line=dict(color=C_UPDATED, width=2, dash="dot"), hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=fc["WeekDate"], y=fc["projected_pos"],
        name=f"Updated forecast (from {source})",
        mode="lines+markers", line=dict(color=C_UPDATED, width=3, dash="dash"),
        marker=dict(size=7),
    ))
    if not sys_proj.empty:
        fig.add_trace(go.Scatter(
            x=sys_proj["WeekDate"], y=sys_proj["Projection"],
            name="Original projection", mode="lines+markers",
            line=dict(color=C_ORIGINAL, width=2, dash="dot"), marker=dict(size=5),
        ))
    title = f"{sku} — {desc}" if isinstance(desc, str) else str(sku)
    return _base_layout(fig, title, ffw, y_title=f"Units ({source})")


# --------------------------------------------------------------------------- #
# Summary table styling                                                       #
# --------------------------------------------------------------------------- #
def style_summary(summary_df):
    """Format numbers and colour the up/down columns (up green / down red)."""
    df = summary_df.copy()
    int_cols = [c for c in [
        "Weeks with data", "Initial Projection Average",
        "Updated Projection Average", "Projection Difference",
    ] if c in df.columns]
    fmt = {c: "{:,.0f}" for c in int_cols}
    avg_col = resolve_avg_col(df)
    if avg_col in df.columns:
        fmt[avg_col] = "{:,.1f}"
    if PRICE_COL in df.columns:
        fmt[PRICE_COL] = lambda v: fmt_dollar(v, decimals=2)
    if RISK_COL in df.columns:
        fmt[RISK_COL] = lambda v: fmt_dollar(v, decimals=0)

    def colour_diff(v):
        if pd.isna(v):
            return ""
        if v > 0:
            return "color:#15803d;font-weight:600"
        if v < 0:
            return "color:#b91c1c;font-weight:600"
        return "color:#64748b"

    sty = df.style.format(fmt, na_rep="—")
    # Colour both the unit difference and the dollar revenue risk by direction.
    diff_cols = [c for c in ["Projection Difference", RISK_COL] if c in df.columns]
    if diff_cols:
        sty = sty.map(colour_diff, subset=diff_cols)
    return sty


def search_filter(df, key, columns=None, placeholder="e.g. a SKU, region, or customer"):
    """Render a search box above a table and return ``df`` filtered to matches.

    Matches the typed query as a case-insensitive substring against every
    column (or just ``columns`` if given). An empty query returns ``df``
    unchanged. Each table needs a unique ``key``. Downloads should stay on the
    unfiltered frame; this only narrows what's shown on screen.
    """
    query = st.text_input("🔍 Search", key=key, placeholder=placeholder).strip()
    if not query:
        return df
    cols = [c for c in (columns or list(df.columns)) if c in df.columns]
    mask = pd.Series(False, index=df.index)
    for c in cols:
        mask |= df[c].astype(str).str.contains(
            query, case=False, na=False, regex=False
        )
    out = df[mask]
    st.caption(f"{len(out):,} of {len(df):,} rows match “{query}”.")
    return out


# --------------------------------------------------------------------------- #
# App                                                                         #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Agent integration (Phase 5)                                                 #
# --------------------------------------------------------------------------- #
# The LangGraph agent runs out-of-band and writes its result to
# outputs/agent_summary_{view}.json (see agent/nodes/publish.py). The dashboard
# only reads that JSON back — it never threads LangGraph's execution model into
# Streamlit's rerun-on-every-interaction model. The agent is invoked strictly
# on an explicit button click (it calls an LLM and backtests every model), and
# the last result is shown from the cached JSON on subsequent reruns.

# Friendly sidebar label -> the LLM_PROVIDER value agent/llm.py resolves at call
# time. "anthropic" = Claude API (ANTHROPIC_API_KEY); "local" = the
# OpenAI-compatible server in LOCAL_LLM_* (see agent/config.py and .env.example).
LLM_PROVIDERS = {
    "Anthropic (Claude API)": "anthropic",
    "Local LLM": "local",
}


def _agent_summary_path(view):
    """Path publish.py writes for a given view (same view->filename mangling)."""
    safe_view = view.replace(" ", "_").replace("/", "-")
    return os.path.join(REPO_ROOT, "outputs", f"agent_summary_{safe_view}.json")


def _load_agent_summary(view):
    """Last agent run for this view, or None if it hasn't run / is unreadable."""
    path = _agent_summary_path(view)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _agent_scores(payload):
    """(scores_dict, is_mase) for an agent summary payload.

    Prefers the current ``mase_by_model`` key; falls back to the legacy
    ``mae_by_model`` written before the MASE migration so a stale JSON still
    renders (with the old MAE wording) until the nightly batch regenerates it.
    """
    if payload.get("mase_by_model") is not None:
        return payload["mase_by_model"], True
    return payload.get("mae_by_model") or {}, False


def _model_fit_callout(payload):
    """(kind, text) for the expected-vs-actual model-fit callout, or None.

    ``kind`` is "info" when the LLM's expected best model differs from the
    selected (MASE-winning) model — worth a prominent callout — and "caption"
    for the quieter agree/mismatch-less cases. Returns None when there's nothing
    to show (older JSONs written before these fields existed). Pure so the render
    branch is unit-testable without a Streamlit context.
    """
    expected = payload.get("expected_best_model")
    best = payload.get("best_model")
    note = payload.get("model_fit_note")
    if expected and best and expected != best:
        return "info", note or f"Expected best fit: {expected} — {best} won on backtest MASE."
    if expected and note:
        return "caption", f"Expected best fit: {expected} (matches the selected model). {note}"
    if note:
        return "caption", note
    return None


@st.dialog("Run agent summary for every view?")
def _confirm_run_all_dialog(provider):
    """Confirmation modal for the all-views batch (it can take up to an hour)."""
    st.write(
        "This backtests all models and writes an LLM narrative for **every** "
        "view — the combined view, each regional rollup, and every customer "
        "group (~60 views)."
    )
    st.warning(
        "It runs in the background and can take **up to 1 hour**. You can keep "
        "using the dashboard while it runs; each view's summary updates as it "
        "finishes. Results land in outputs/agent_summary_<view>.json."
    )
    left, right = st.columns(2)
    if left.button("Cancel", key="confirm_batch_cancel", width="stretch"):
        st.rerun()
    if right.button("Run all views", key="confirm_batch_go",
                    type="primary", width="stretch"):
        ok, msg = start_agent_batch(provider)
        if ok:
            st.session_state["_batch_toast"] = f"Agent batch started ({msg})."
        else:
            st.session_state["_batch_toast"] = f"⚠️ {msg}"
        st.rerun()


# Progress markers for the agent run. Keys are LangGraph node names (see
# agent/graph.py); each maps to (fraction_complete, user-facing label). graph
# .stream() yields one update per node as it finishes, so we bump the bar to the
# node's fraction when its update arrives. evaluate_models is the long pole (4
# models x 6 walk-forward re-fits), hence the big jump to 0.75.
_AGENT_NODE_PROGRESS = {
    "ingest": (0.15, "Loading & cleaning data…"),
    "run_all_models": (0.40, "Fitting the forecast models…"),
    "evaluate_models": (0.75, "Backtesting models (walk-forward) to compare accuracy…"),
    "select_best_model": (0.80, "Selecting the best model…"),
    "flag_anomalies": (0.88, "Flagging anomalies…"),
    "summarize": (0.95, "Writing the summary…"),
    "flag_low_confidence": (0.95, "Writing the low-confidence note…"),
    "publish": (1.0, "Publishing results…"),
}


def _run_agent_job(view, today_ts, shared):
    """Run the agent pipeline on a background thread, streaming progress.

    Runs OFF the main Streamlit script thread so the UI stays interactive. It
    must NOT touch any ``st.*`` API — it only mutates the plain ``shared`` dict
    (created on the main thread, polled by the progress fragment). LLM_PROVIDER
    is read from the env at call time by agent/llm.py, so the caller sets it
    before starting this thread.
    """
    # Per-model progress from inside the fit/backtest node loops. The nodes call
    # this via RunnableConfig (see agent/state.report_progress); it maps each
    # phase to its slice of the bar so the user sees e.g. "Fitting XGBoost (3/4)".
    def _cb(phase, model, done, total):
        total = max(int(total), 1)
        if phase == "fit":  # fit loop occupies 0.15 -> 0.40 of the bar
            shared["progress"] = 0.15 + 0.25 * (done / total)
            shared["step"] = f"Fitting {model} ({done}/{total})"
        elif phase == "backtest":  # backtest loop occupies 0.40 -> 0.75
            shared["progress"] = 0.40 + 0.35 * (done / total)
            shared["step"] = f"Backtesting {model} ({done}/{total})"

    try:
        # Import here, not at module top: keeps langgraph off the hot import
        # path for every rerun and matches the "only touched on click" rule.
        from agent.graph import build_graph

        graph = build_graph()
        # stream_mode="updates" (the default) yields {node_name: state_delta}
        # after each node finishes; accumulate the deltas so best_model/errors
        # are available when the run ends. The progress_cb (passed via config)
        # supplies finer per-model updates from inside the fit/backtest nodes.
        config = {"configurable": {"progress_cb": _cb}}
        for update in graph.stream({"view": view, "today_ts": today_ts}, config=config):
            for node_name, delta in update.items():
                frac, label = _AGENT_NODE_PROGRESS.get(
                    node_name, (shared.get("progress", 0.0), "Working…")
                )
                shared["progress"] = frac
                shared["step"] = label
                if isinstance(delta, dict):
                    shared["result"].update(delta)
        shared["status"] = "done"
    except Exception as exc:  # surface any failure to the UI instead of a dead spinner
        shared["error"] = f"{type(exc).__name__}: {exc}"
        shared["status"] = "error"


@st.fragment(run_every=0.5)
def _agent_progress_fragment():
    """Poll the background agent job and render its progress bar.

    Only THIS fragment reruns on the 0.5s timer — the rest of the page stays
    interactive while the pipeline runs on its background thread. When the job
    finishes, trigger one full app rerun so main() can finalize (toast, switch
    the model toggle, render the summary).
    """
    job = st.session_state.get("agent_job") or {}
    started = job.get("started_at")
    elapsed_txt = ""
    if started:
        secs = int(time.time() - started)
        elapsed_txt = f"  ·  {secs // 60}:{secs % 60:02d} elapsed"
    st.progress(
        min(float(job.get("progress", 0.0)), 1.0),
        text=f"Running agent — {job.get('step', 'Working…')}{elapsed_txt}",
    )
    if started:
        st.caption(f"Started at {time.strftime('%H:%M:%S', time.localtime(started))}")
    if job.get("status") in ("done", "error"):
        st.rerun(scope="app")


def _render_agent_summary(view):
    """Render the cached agent summary for `view` in the main body, if any."""
    payload = _load_agent_summary(view)
    if payload is None:
        return
    with st.expander("Agent summary", expanded=True):
        gen = payload.get("generated_at")
        if gen:
            st.caption(f"Generated {gen}  ·  view: {payload.get('view', view)}")

        if payload.get("errors"):
            st.error("\n".join(payload["errors"]))

        scores, is_mase = _agent_scores(payload)

        best = payload.get("best_model")
        if best:
            score = scores.get(best)
            label = f"Best model: {best}"
            if score is not None:
                label += (
                    f"  (backtest MASE {score:.2f})" if is_mase
                    else f"  (backtest MAE {score:.1f})"
                )
            if payload.get("confidence_flag"):
                st.warning(label + "  —  ⚠️ low confidence")
            else:
                st.success(label)

        # Expected vs. actual best model: the LLM's a-priori pick from the view's
        # demand pattern, reconciled against the MASE winner. Guarded so older
        # summary JSONs (written before these fields existed) render as before.
        callout = _model_fit_callout(payload)
        if callout is not None:
            kind, text = callout
            (st.info if kind == "info" else st.caption)(text)

        # All models' backtest scores side by side, so the user can see how
        # close the call was. Scores come straight from publish.py; a model
        # whose backtest failed has None -> shown as "n/a", sorted last.
        if scores:
            col = "Backtest MASE (vs 8-wk avg)" if is_mase else "Backtest MAE"
            rows = [
                {
                    "Model": name,
                    col: (
                        "n/a" if score is None
                        else round(float(score), 2 if is_mase else 1)
                    ),
                    "Best": "✓" if name == best else "",
                    "_sort": (float("inf") if score is None else float(score)),
                }
                for name, score in scores.items()
            ]
            score_df = (
                pd.DataFrame(rows)
                .sort_values("_sort")
                .drop(columns="_sort")
                .reset_index(drop=True)
            )
            st.markdown("**Model comparison:**")
            st.dataframe(score_df, hide_index=True)
            if is_mase:
                st.caption(
                    "Backtest MASE from walk-forward (one-step-ahead) validation: "
                    "model error ÷ a plain 8-week moving average's error on the "
                    "same weeks. < 1 beats the 8-week average; lower = better; "
                    "winner chosen by lowest MASE."
                )
            else:
                st.caption(
                    "Backtest MAE from walk-forward (one-step-ahead) validation — "
                    "lower = closer fit; winner chosen by lowest MAE."
                )

        if payload.get("narrative"):
            st.write(payload["narrative"])

        anomalies = payload.get("anomalies") or []
        if anomalies:
            st.markdown("**Flagged anomalies:**")
            for a in anomalies:
                # publish stores bullets as-is; add a marker only if missing.
                st.markdown(a if a.lstrip().startswith(("-", "*", "•")) else f"- {a}")

        # Active SKUs the winning model leaves out because their demand predates
        # its history window (e.g. the 8-week moving average). Surfaced so a SKU
        # that an all-history model (Holt/XGBoost) would forecast isn't silently
        # dropped without explanation. Empty for an all-history winner.
        excluded = payload.get("window_excluded_skus") or []
        if excluded:
            best_lbl = payload.get("best_model") or "this model"
            # Rendered as a native HTML <details> dropdown (collapsed by
            # default) so this list — often 15+ SKUs — doesn't dominate the
            # summary. A Streamlit st.expander can't be used here: it's illegal
            # to nest one inside the "Agent summary" expander this runs in.
            items = "".join(
                "<li>{}{}</li>".format(
                    html.escape(str(row.get("SKU", ""))),
                    " — " + html.escape(str(row.get("Description", "")))
                    if row.get("Description")
                    else "",
                )
                for row in excluded
            )
            st.markdown(
                "<details style='margin:0.25rem 0 0.5rem;'>"
                "<summary style='cursor:pointer;font-weight:600;'>"
                f"Active SKUs outside {html.escape(best_lbl)}'s history window "
                f"({len(excluded)})</summary>"
                "<div style='opacity:0.75;font-size:0.9em;margin:0.35rem 0;'>"
                "These have demand history but none inside the model's window, "
                "so they carry no projection here. Switch to an all-history "
                "model (Holt or XGBoost) to forecast them.</div>"
                f"<ul style='margin:0;padding-left:1.2rem;'>{items}</ul>"
                "</details>",
                unsafe_allow_html=True,
            )


def main():
    st.set_page_config(
        page_title="Demand Projections", page_icon="📦", layout="wide"
    )

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

        st.radio(
            "Forecasting model", list(MODEL_OPTIONS.keys()),
            key="model_choice", on_change=_on_model_change,
            help="Switching recomputes everything with the selected pipeline.",
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
                f"⏳ Refreshing data… started {started or wh_started}. "
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
                help="Off: always load the newest snapshot, Plytix feed, and "
                     "warehouse files. On: pick specific files or upload your "
                     "own in the boxes that appear below.",
            )
        with col_btn:
            do_refresh = False
            if running or wh_running:
                if st.button("Check for new data", key="check_refresh"):
                    st.rerun()
            else:
                do_refresh = st.button(
                    "🔄 Refresh data",
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
                    locs = ", ".join(sorted(warehouse_df["Location"].unique()))
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
            [ALL_CUSTOMERS_VIEW, BEST_MODEL_COMBINED_VIEW, "By region"],
            index=0,
            help="“Combined (best model per group)” forecasts each customer "
                 "group with its own backtest-winning model (from the agent "
                 "summaries) and stitches them into one table.",
        )
        if scope == ALL_CUSTOMERS_VIEW:
            view = ALL_CUSTOMERS_VIEW
            region = None
        elif scope == BEST_MODEL_COMBINED_VIEW:
            view = BEST_MODEL_COMBINED_VIEW
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

    # Now that the view is known, fill the header caption — except for the
    # combined best-model view, which uses a different model per group (so the
    # selected-model blurb would mislead) and renders its own caption instead.
    if view != BEST_MODEL_COMBINED_VIEW:
        _header_caption_slot.caption(header_caption)

    # ----- Agent summary (LangGraph pipeline) ------------------------------
    # Button-triggered only: invoking the graph backtests all three models AND
    # calls an LLM, which is far too slow/expensive to run on every rerun. The
    # provider selector switches the reasoning nodes between the Claude API and
    # a local OpenAI-compatible server; agent/llm.py re-reads LLM_PROVIDER from
    # the env at call time, so setting it here just before invoke() is enough.
    with st.sidebar:
        st.header("Agent")
        provider_label = st.radio(
            "Reasoning LLM",
            list(LLM_PROVIDERS.keys()),
            key="agent_llm_provider",
            help="Which LLM writes the narrative + anomaly flags. Anthropic "
                 "calls the Claude API (needs ANTHROPIC_API_KEY in .env); Local "
                 "uses the OpenAI-compatible server in LOCAL_LLM_* (.env).",
        )
        # Anthropic needs a key; without one, block the run and steer the user
        # to Local rather than silently degrading to it behind the scenes.
        anthropic_no_key = LLM_PROVIDERS[provider_label] == "anthropic" and not (
            os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        )
        if anthropic_no_key:
            st.caption("⚠️ No ANTHROPIC_API_KEY found — select **Local LLM** to run the agent.")
        # The combined best-model view isn't a single agent view — it reads the
        # per-group summaries. Steer the user to the all-views batch instead.
        single_view_agent = view != BEST_MODEL_COMBINED_VIEW
        if not single_view_agent:
            st.caption("This view combines every group's best model — use "
                       "**Agent Summary (all views)** below to (re)generate them.")
        run_agent = st.button(
            "Run Agent Summary",
            key="run_agent_summary",
            disabled=anthropic_no_key or not single_view_agent,
            help="Backtests all models for this view, picks the best, and writes "
                 "an LLM narrative + flagged anomalies. Slow/expensive — runs "
                 "only when you click, never on a normal rerun.",
        )

        # All-views batch (same work as `python -m agent.batch`). Detached
        # background process; while it runs the button becomes a status check.
        batch_running, batch_started = batch_in_progress()
        if batch_running:
            st.info(f"⏳ Running agent summary for all views… started "
                    f"{batch_started or '?'}. Runs in the background — "
                    "see logs/<date>/logs_agent_batch.txt.")
            if st.button("Check progress", key="check_agent_batch"):
                st.rerun()
        else:
            # A just-finished batch (this session): surface its outcome once.
            proc = st.session_state.get("agent_batch_proc")
            if proc is not None and proc.poll() is not None:
                line = _batch_result_line() or "Agent batch finished."
                st.session_state["_batch_toast"] = line
                st.session_state.pop("agent_batch_proc", None)
            run_all = st.button(
                "Agent Summary (all views)",
                key="run_agent_all",
                disabled=anthropic_no_key,
                help="Backtests all models for EVERY view and writes each "
                     "agent_summary_<view>.json. Runs ~60 views — can take up "
                     "to 1 hour. Asks for confirmation first.",
            )
            if run_all:
                _confirm_run_all_dialog(LLM_PROVIDERS[provider_label])

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
                st.error(job.get("error") or "\n".join(result.get("errors", [])) or "Agent run failed.")
                job["status"] = "shown"  # consume so the error isn't re-raised on later reruns
            else:
                best = result.get("best_model") or (_load_agent_summary(view) or {}).get("best_model")
                started = job.get("started_at")
                dur = f" in {int(time.time() - started)}s" if started else ""
                st.toast(f"Agent finished{dur}: {best or 'n/a'}")
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
        _render_best_model_combined(df, today_ts, today_str, prices, n_excluded_rows)
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
    # Avg. weekly demand = the mean of the TOTAL weekly demand actually plotted
    # on the chart's "Actual demand" line (POS/Orders summed across SKUs per
    # week, then averaged over the weeks in the window). Do NOT sum the per-SKU
    # "N Week POS/Orders Average" column here: that per-SKU average divides each
    # SKU by its own weeks-with-data, so summing it counts a SKU that sold in
    # only a few weeks as if it sold every week and overstates the total.
    avg_col = resolve_avg_col(summary)
    hist_demand = historical_window(agg, summary, (lb, lcw, ffw))
    weekly_totals = hist_demand.groupby("WeekDate")["demand"].sum(min_count=1)
    total_avg = float(weekly_totals.mean()) if not weekly_totals.empty else 0.0
    total_updated = summary["Updated Projection Average"].sum()
    total_initial = summary["Initial Projection Average"].sum()
    diff = total_updated - total_initial
    # Total Projection Value = Σ (list price × updated weekly-avg forecast) over
    # priced SKUs. Unpriced SKUs map to NaN and are skipped, so this covers the
    # same population as Revenue Risk. Per-week basis (Updated Projection Average
    # is already a weekly mean).
    has_price = PRICE_COL in summary.columns and summary[PRICE_COL].notna().any()
    proj_value = (
        (summary[PRICE_COL] * summary["Updated Projection Average"]).sum()
        if has_price else None
    )
    n_orders = int((summary.get("Data Source") == "Orders").sum()) \
        if "Data Source" in summary.columns else 0

    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    k1.metric(
        "SKUs Forecasted", f"{len(summary):,}",
        help=f"{n_orders} forecast from Orders (no POS)" if n_orders else None,
    )
    k2.metric(
        "Historical Demand (avg/wk)", f"{total_avg:,.0f}",
        help=f"Mean of total weekly actual demand (POS/Orders) over the "
             f"{avg_window_phrase(avg_col).lower()} window — the average of the "
             f"chart's actual-demand line.",
    )
    k3.metric(
        "Initial Forecast (avg/wk)", f"{total_initial:,.0f}",
        help="Mean of the existing system projection over the forecast horizon "
             "(the 15 future weeks) — the average of the chart's original-"
             "projection line over the forecast window.",
    )
    k4.metric(
        "Updated Forecast (avg/wk)", f"{total_updated:,.0f}",
        help="Mean of this model's updated forecast over the 15 future weeks — "
             "the average of the chart's updated-forecast line.",
    )
    k5.metric(
        "Projection Difference (avg/wk)", f"{diff:+,.0f}",
        delta=f"{(diff / total_initial * 100):+.1f}%" if total_initial else None,
    )
    has_risk = RISK_COL in summary.columns and summary[RISK_COL].notna().any()
    if has_risk:
        net_risk = summary[RISK_COL].sum()
        k6.metric(
            "Revenue Risk (avg/wk)", fmt_dollar(net_risk, signed=True),
            help="Σ (projection difference × list price) over priced SKUs. "
                 "Negative = forecast fell below the original projection.",
        )
    else:
        k6.metric(
            "Revenue Risk (avg/wk)", "—",
            help="Load a list_prices_*.xlsx (sidebar) to enable revenue risk.",
        )
    if proj_value is not None:
        k7.metric(
            "Projected Revenue (avg/wk)", fmt_dollar(proj_value),
            help="Σ (list price × updated weekly-avg forecast) over priced SKUs "
                 "— the gross value at list price of the forecasted weekly demand.",
        )
    else:
        k7.metric(
            "Projected Revenue (avg/wk)", "—",
            help="Load a list_prices_*.xlsx (sidebar) to enable projection value.",
        )
    if n_orders:
        st.caption(
            f"⚑ {n_orders} of {len(summary)} SKUs had no POS in the window and "
            "were forecast from Orders (see the Data Source column)."
        )
    if PRICE_COL in summary.columns:
        n_noprice = int(summary[PRICE_COL].isna().sum())
        if n_noprice:
            st.caption(
                f"💲 {n_noprice} of {len(summary)} SKUs have no list price; "
                "their revenue risk is left blank."
            )

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
    st.dataframe(
        style_summary(search_filter(summary_table, key="search_by_sku")),
        width="stretch", hide_index=True,
    )
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
            st.dataframe(
                style_summary(
                    search_filter(by_cust_table, key="search_by_customer")
                ),
                width="stretch", hide_index=True,
            )
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

    # ----- Discontinued/inactive products with projections ------------------
    render_discontinued_section(
        view, region, disc_check_ran, discontinued_df, today_str,
    )


def render_inactive_section(view, region, check_ran, inactive_df,
                            excluded_counts_by_key, n_excluded_rows, today_str):
    """Table of active products projected in regions they are not 'Active in'.

    The rows behind these SKU × customer × region combos were dropped from every
    summary table above (the SKU isn't 'Active in' that region). Surface them so
    the exclusion is visible and auditable.
    """
    HEADER = "### SKUs with forecasts in locations they are not active in"
    st.markdown(HEADER)
    if not check_ran:
        st.info(
            "Upload a Plytix export with an 'Active in' column (sidebar) to run "
            "the active-in check."
        )
        return

    # In a "By customer group" view, mirror the summary table above: only
    # show rows whose region matches the selected region (e.g. a US view
    # shouldn't list "JP (NETDEPOT)" rows). ALL CUSTOMERS shows every region.
    region_scoped = view != ALL_CUSTOMERS_VIEW and region is not None
    table_df = inactive_df
    if region_scoped:
        table_df = inactive_df[inactive_df["Region"] == region]

    # Always show only non-zero future projections: rows whose Last_WeekDate is
    # this week and onward (Sunday-anchored via _this_week_start, matching the
    # "future avg/wk" projection column) with a non-zero projection.
    week_start = _this_week_start().date()
    fdf = table_df.copy()
    fdf["First_WeekDate"] = pd.to_datetime(fdf["First_WeekDate"]).dt.date
    fdf["Last_WeekDate"] = pd.to_datetime(fdf["Last_WeekDate"]).dt.date
    fdf["Original_Projection"] = pd.to_numeric(
        fdf["Original_Projection"], errors="coerce"
    ).round(0)
    fdf = fdf[
        (fdf["Last_WeekDate"] >= week_start)
        & (fdf["Original_Projection"].notna())
        & (fdf["Original_Projection"] != 0)
    ]

    if fdf.empty:
        if region_scoped:
            st.success(
                f"None found for {region} — every active product here is "
                "only forecast in regions it is active in."
            )
        else:
            st.success(
                "None found — every active product is only forecast in "
                "regions it is active in."
            )
        return

    n_skus = fdf["SKU"].nunique()
    scope_note = f" for {region}" if region_scoped else ""
    st.caption(
        f"Excluded from the forecast above{scope_note}: "
        f"{n_skus:,} distinct SKUs. Each is an active product being "
        "forecast in a region (US/CA/EU/JP/AU) that is not in its Plytix "
        "'Active in' list."
    )

    show = fdf[[
        'SKU', 'Region', 'Active in', 'Customer Grouping',
        'First_WeekDate', 'Last_WeekDate', 'Original_Projection',
    ]].rename(columns={"Original_Projection": "Original Projection (future avg/wk)"})
    st.dataframe(
        search_filter(show, key="search_inactive"),
        width="stretch", hide_index=True,
    )
    st.download_button(
        "⬇️ Download the excluded (inactive-region) projections table",
        data=summary_to_excel(show, sheet_name="inactive_projections"),
        file_name=f"inactive_projections_{today_str}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_inactive_projections",
    )


def render_missing_section(view, region, warehouse_df, check_ran, missing_df,
                           today_str, cust_source=None, P=None):
    """Table of active products MISSING future projections in active regions.

    Ported from active_missing_projections.py. The inverse of the inactive
    section above: these are active SKUs that ARE 'Active in' a region but have
    no projection for one or more of the coming 15 weeks there. Sourced from the
    warehouse projection grid (sidebar), the only place a blank/missing week is
    visible. In a "By customer group" view, only rows whose region matches the
    selected region are shown; ALL CUSTOMERS shows every region.
    """
    HEADER = "### SKUs missing forecasts in locations they are active in"
    st.markdown(HEADER)
    if warehouse_df is None or warehouse_df.empty:
        st.info(
            "Upload the warehouse projection files (AU/CA/EU/JP/US) in the "
            "sidebar to run the missing-projections check."
        )
        return
    if not check_ran:
        st.info(
            "Upload a Plytix export with an 'Active in' column (sidebar) to run "
            "the missing-projections check."
        )
        return

    # Each CUSTNMBR folds to its forecast customer group (e.g. AMAZON-DS ->
    # AMAZON-DC); used both to scope a by-customer view and to look up the source.
    grouping = getattr(P, "COMBINED_GROUPING", {}) if P is not None else {}
    row_group = missing_df["CUSTNMBR"].map(lambda c: grouping.get(c, c))

    # A by-customer-group view shows only that group's rows (not every customer
    # in the region); a per-region "All Customers" rollup shows every group in
    # its region; ALL CUSTOMERS shows everything.
    group_scoped = view != ALL_CUSTOMERS_VIEW
    region_all = region_from_view(view)
    table_df = missing_df
    if region_all is not None and P is not None:
        table_df = missing_df[
            row_group.map(lambda g: str(P.region_for_group(g))) == region_all
        ]
    elif group_scoped:
        table_df = missing_df[row_group == view]

    if table_df.empty:
        if group_scoped:
            st.success(
                f"None found for {view} — every active product here has "
                "future projections in the regions it is active in."
            )
        else:
            st.success(
                "None found — every active product has future projections in "
                "the regions it is active in."
            )
        return

    n_skus = table_df["SKU"].nunique()
    scope_note = f" for {view}" if group_scoped else ""
    st.caption(
        f"Flagged{scope_note}: {n_skus:,} distinct SKUs. Each is an active "
        "product (Plytix) with no projection for one or more of the coming 15 "
        "weeks in a region (US/CA/EU/JP/AU) it IS 'Active in'."
    )

    show = table_df[[
        'SKU', 'Location', 'Active in', 'CUSTNMBR',
        'First_WeekDate', 'Last_WeekDate',
    ]].rename(columns={
        "First_WeekDate": "First Missing Week",
        "Last_WeekDate": "Last Missing Week",
    })
    # Data source (POS/Orders) from the summary table, keyed by (customer, SKU).
    src_lookup = cust_source or {}
    show.insert(
        show.columns.get_loc("CUSTNMBR") + 1,
        "Data Source",
        [
            src_lookup.get((grouping.get(c, c), str(s).rstrip("*")))
            for s, c in zip(show["SKU"], show["CUSTNMBR"])
        ],
    )
    st.dataframe(
        search_filter(show, key="search_missing"),
        width="stretch", hide_index=True,
    )
    st.download_button(
        "⬇️ Download the missing-projections table",
        data=summary_to_excel(show, sheet_name="missing_projections"),
        file_name=f"missing_projections_{today_str}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_missing_projections",
    )


def render_discontinued_section(view, region, disc_check_ran, discontinued_df,
                                today_str):
    """Table of Discontinued/Inactive products that still carry projections.

    Ported from discontinued_with_projections.ipynb. In a "By customer group"
    view, only rows whose region matches the selected region are shown (e.g. an
    EU view won't list AAFES, a US customer); ALL CUSTOMERS shows every region.
    """
    HEADER = "### Inactive/discontinued SKUs with forecasts"
    st.markdown(HEADER)
    if not disc_check_ran:
        st.info(
            "Upload a Plytix export with a 'SKU Status' column (sidebar) to run "
            "the discontinued-product check."
        )
        return

    region_scoped = view != ALL_CUSTOMERS_VIEW and region is not None
    table_df = discontinued_df
    if region_scoped:
        table_df = discontinued_df[discontinued_df["Region"] == region]

    # Apply the non-zero future-projection filter BEFORE the empty check, so a
    # scope whose rows all zero out still shows the "None found" message rather
    # than an empty table (mirrors render_inactive_section above).
    disc = table_df.copy()
    disc["First_WeekDate"] = pd.to_datetime(disc["First_WeekDate"]).dt.date
    disc["Last_WeekDate"] = pd.to_datetime(disc["Last_WeekDate"]).dt.date
    disc["Original_Projection"] = pd.to_numeric(
        disc["Original_Projection"], errors="coerce"
    ).round(0)
    disc = disc[
        disc["Original_Projection"].notna() &
        (disc["Original_Projection"] != 0)
    ]

    if disc.empty:
        if region_scoped:
            st.success(
                f"None found for {region} — no discontinued or inactive "
                "products carry future projections here."
            )
        else:
            st.success(
                "None found — no discontinued or inactive products carry "
                "future projections."
            )
        return

    n_skus = disc["SKU"].nunique()
    scope_note = f" for {region}" if region_scoped else ""
    st.caption(
        f"Flagged{scope_note}: {n_skus:,} distinct SKUs marked Discontinued or "
        "Inactive in Plytix that still carry future projections (future weeks "
        "only)."
    )

    disc = disc[[
        'SKU', 'SKU Status', 'Region', 'Customer Grouping',
        'First_WeekDate', 'Last_WeekDate', 'Original_Projection',
    ]].rename(columns={"Original_Projection": "Original Projection (future avg/wk)"})

    st.dataframe(
        search_filter(disc, key="search_discontinued"),
        width="stretch", hide_index=True,
    )
    st.download_button(
        "⬇️ Download the discontinued/inactive projections table",
        data=summary_to_excel(disc, sheet_name="discontinued_projections"),
        file_name=f"discontinued_with_projections_{today_str}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_discontinued_projections",
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
