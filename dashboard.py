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
import glob
import json
import inspect
import logging
import tempfile
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

# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
# Developer-facing log. Written next to this file as ``logs.txt`` so issues can
# be inspected after the fact (on Streamlit Cloud, also visible via Manage app
# → logs). Configured once per process; Streamlit reruns import the module only
# once, so the handler isn't attached repeatedly.
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs.txt")

logger = logging.getLogger("demand_dashboard")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    try:
        _fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
        _fh.setFormatter(_fmt)
        logger.addHandler(_fh)
    except OSError:
        # Read-only filesystem (some hosts): fall back to the console only so
        # the app still runs; logs then live in the platform's own log stream.
        pass
    _sh = logging.StreamHandler()
    _sh.setFormatter(_fmt)
    logger.addHandler(_sh)
    logger.propagate = False

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))

# The forecasting models on offer. Each entry maps the label shown in the
# sidebar toggle to the pipeline file implementing it. When a DEMAND_PIPELINE
# env var is set, that file is offered as an extra option and is the default.
MODEL_OPTIONS = {
    "Simple Regression": os.path.join(HERE, "models/regression.py"),
    "Holt's Exponential Smoothing": os.path.join(HERE, "models/exponential_smoothing.py"),
    "XGBoost": os.path.join(HERE, "models/xgboost.py"),
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
            "models/exponential_smoothing.py, models/xgboost.py or "
            "models/regression.py next to dashboard.py "
            "(or set the DEMAND_PIPELINE env var)."
        )
    return MODEL_OPTIONS[choice]

ALL_CUSTOMERS_VIEW = "ALL CUSTOMERS (combined)"

# The warehouse regions we check "Active in" against. A SKU should only be
# projected in a region it is "Active in" (per the Plytix export); a projection
# in any other region is flagged and excluded from the forecast (see the
# inactive-projections logic below, ported from inactive_projections.ipynb).
WAREHOUSE_REGIONS = ["AU", "CA", "EU", "JP", "US"]

# Column names produced by the pipeline's fit_regression when list prices are
# supplied (see DISPLAY_NAMES in the pipeline). Kept here so the dashboard can
# format / sort on them.
PRICE_COL = "List Price (USD)"
RISK_COL = "Revenue Risk (USD)"

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
# "Active in" check (ported from inactive_projections.ipynb): active products  #
# should only carry projections in regions they are Active in. Projections in   #
# any other region are flagged, dropped from the forecast, and surfaced in     #
# their own table. The check reads the demand file directly — the same data    #
# the dashboard forecasts.                                                     #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Reading Plytix export…")
def read_plytix_from_path(path, _mtime):
    """Read the raw Plytix export from disk (for the 'Active in' check)."""
    return pd.read_excel(path)


@st.cache_data(show_spinner="Reading Plytix export…")
def read_plytix_from_bytes(_data, name):
    """Read the raw Plytix export from uploaded bytes (for the 'Active in' check)."""
    return pd.read_excel(BytesIO(_data))


INACTIVE_COLS = [
    "SKU", "Location", "Region", "Active in", "Customer Grouping",
    "CUSTNMBR", "First_WeekDate", "Last_WeekDate", "Original_Projection", "Source",
]


def _this_week_start():
    """Sunday-anchored start of the current week as a Timestamp.

    WeekDates fall on Sunday-anchored 7-day boundaries, so "this week" is the
    most recent Sunday on or before today (e.g. today 7/7 -> 7/5). Shared by the
    excluded-table future-projection column and its "future only" toggle so both
    use the exact same boundary.
    """
    today = pd.Timestamp.today().normalize()
    return today - pd.Timedelta(days=(today.weekday() + 1) % 7)


def _active_in_list(sku_active_in, sku):
    """The list of regions a SKU is 'Active in' (e.g. ['US', 'CA', 'EU'])."""
    return [x.strip() for x in str(sku_active_in.get(sku, "")).split(",")]


def compute_active_products(plytix_df):
    """From the Plytix export, the set of active-product SKUs and a
    SKU -> 'Active in' string lookup.

    "Active product" mirrors inactive_projections.ipynb: SKU Status == Active,
    SKU Type == Product, and SKUs starting LS/AS excluded. Trailing '*' markers
    are stripped so SKUs line up with the demand file.

    Returns (active_sku_set, sku_active_in) or (None, None) if the Plytix export
    lacks the columns the check needs (an older list-price file).
    """
    required = {"SKU", "SKU Status", "SKU Type", "Active in"}
    if plytix_df is None or not required.issubset(plytix_df.columns):
        return None, None
    p = plytix_df.copy()
    p["SKU"] = p["SKU"].astype(str).str.rstrip("*")
    act = p[(p["SKU Status"] == "Active") & (p["SKU Type"] == "Product")]
    act = act[~act["SKU"].str.startswith(("LS", "AS"))]
    active_sku_set = set(act["SKU"])
    # Full (un-exploded) Active-in string per SKU, e.g. "US,CA,UK,SG,EU,AU".
    sku_active_in = dict(zip(p["SKU"], p["Active in"].astype(str)))
    return active_sku_set, sku_active_in


def _region_code(P, grouping):
    """Two-letter region code for a customer grouping (US/CA/EU/JP/AU), or None.

    The pipeline's region_for_group returns labels like "JP (NETDEPOT)" or
    "US (LBC+NJ)"; the leading two letters are the region code we match against
    Plytix 'Active in'. Anything else (e.g. "Other") returns None.
    """
    try:
        label = P.region_for_group(grouping)
    except Exception:
        return None
    code = str(label)[:2].upper()
    return code if code in WAREHOUSE_REGIONS else None


def compute_inactive_projections(df, active_sku_set, sku_active_in, P,
                                 anchors=None):
    """Active products showing up in a region they are not 'Active in'.

    This is the fix for cases like ST1082 (active in US/CA/UK/SG/EU/AU but not
    JP), which still appeared in the JP (NETDEPOT) summary: the dashboard builds
    a forward forecast for any SKU with demand history in a region. We look at
    the *demand file itself* — the same data the dashboard forecasts — map each
    customer to its region via the pipeline's region_for_group, and flag any
    active product whose region is not in its Plytix 'Active in' list.

    Returns a table (columns = INACTIVE_COLS) of the flagged
    SKU x customer x region combinations, empty if none or inputs are missing.
    """
    if not active_sku_set or not sku_active_in:
        return pd.DataFrame(columns=INACTIVE_COLS)

    frames = []

    # ----- Primary: the demand file, by customer-group region ---------------
    if df is not None and not df.empty:
        m = df.copy()
        m["SKU"] = m["SKU"].astype(str).str.rstrip("*")
        m = m[m["SKU"].isin(active_sku_set)]
        if not m.empty:
            m["Region"] = m["Customer Grouping"].map(
                lambda g: P.region_for_group(g)
            )
            m["Location"] = m["Customer Grouping"].map(
                lambda g: _region_code(P, g)
            )
            m = m[m["Location"].notna()]
            keep = [
                loc not in _active_in_list(sku_active_in, sku)
                for sku, loc in zip(m["SKU"], m["Location"])
            ]
            m = m[keep]
            m["WeekDate"] = pd.to_datetime(m["WeekDate"])
            # Only flag pairs the dashboard would actually forecast — i.e. that
            # carry a POS/Orders demand signal in the historical window (that is
            # exactly what puts a SKU in a region's summary). Without anchors we
            # fall back to any presence.
            if anchors is not None and not m.empty:
                lb, lcw, _ = anchors
                sig = (
                    (m["WeekDate"] >= lb) & (m["WeekDate"] <= lcw)
                    & (m["POS"].notna() | m.get("Orders", pd.Series(index=m.index)).notna())
                )
                live = m.loc[sig, ["SKU", "CUSTNMBR", "Location"]].drop_duplicates()
                m = m.merge(live, on=["SKU", "CUSTNMBR", "Location"], how="inner")
            if not m.empty:
                # Original projection over future weeks (this week onward) —
                # averaged per week — the projected weekly volume being excluded
                # going forward. Uses the same week boundary as the excluded
                # table's "future only" toggle.
                m["_future_proj"] = pd.to_numeric(
                    m["Projection"], errors="coerce"
                ).where(m["WeekDate"] >= _this_week_start())
                g = m.groupby(
                    ["SKU", "Location", "Region", "Customer Grouping", "CUSTNMBR"],
                    as_index=False,
                ).agg(
                    First_WeekDate=("WeekDate", "min"),
                    Last_WeekDate=("WeekDate", "max"),
                    Original_Projection=("_future_proj", "mean"),
                )
                g["Active in"] = g["SKU"].map(lambda s: sku_active_in.get(s))
                g["Source"] = "Demand file"
                frames.append(g)

    if not frames:
        return pd.DataFrame(columns=INACTIVE_COLS)

    out = pd.concat(frames, ignore_index=True)[INACTIVE_COLS]
    out = out.drop_duplicates(subset=["SKU", "Location", "CUSTNMBR"], keep="first")
    return out.sort_values(["SKU", "Location", "CUSTNMBR"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Discontinued/inactive-product check (ported from                             #
# discontinued_with_projections.ipynb): a SKU marked Discontinued or Inactive  #
# in Plytix should not carry forward-looking projections. We look at the       #
# demand file, keep the SKUs whose Plytix status is Discontinued/Inactive,     #
# and surface any that still have future projection weeks.                     #
# --------------------------------------------------------------------------- #
DISCONTINUED_COLS = [
    "SKU", "SKU Status", "Region", "Customer Grouping", "CUSTNMBR",
    "First_WeekDate", "Last_WeekDate", "Original_Projection",
]


def compute_discontinued_products(plytix_df):
    """SKU -> 'SKU Status' lookup for Discontinued/Inactive products.

    Mirrors discontinued_with_projections.ipynb: keep rows whose SKU Status is
    'Discontinued' or 'Inactive'. Trailing '*' markers are stripped so SKUs line
    up with the demand file. Returns None if the Plytix export lacks the columns
    the check needs (an older list-price file).
    """
    required = {"SKU", "SKU Status"}
    if plytix_df is None or not required.issubset(plytix_df.columns):
        return None
    p = plytix_df.copy()
    p["SKU"] = p["SKU"].astype(str).str.rstrip("*")
    disc = p[p["SKU Status"].isin(["Discontinued", "Inactive"])]
    return dict(zip(disc["SKU"], disc["SKU Status"]))


def compute_discontinued_projections(df, disc_status, P):
    """Discontinued/inactive products that still carry future projections.

    Ported from discontinued_with_projections.ipynb: intersect the demand file
    with the discontinued/inactive SKU set, keep only future projection weeks
    (WeekDate after today), and aggregate to one row per SKU x customer with the
    first/last projected week. A Region column (via the pipeline's
    region_for_group) is added so a by-customer-group view can be scoped to its
    own region.

    Returns a table (columns = DISCONTINUED_COLS), empty if none or inputs are
    missing.
    """
    if not disc_status or df is None or df.empty:
        return pd.DataFrame(columns=DISCONTINUED_COLS)

    m = df.copy()
    m["SKU"] = m["SKU"].astype(str).str.rstrip("*")
    m = m[m["SKU"].isin(disc_status)]
    if m.empty:
        return pd.DataFrame(columns=DISCONTINUED_COLS)

    # Future projections only. Like the active-in table, "future" starts at the
    # beginning of the current week (Sunday-anchored via _this_week_start), so
    # the in-progress week is included — e.g. 7/5 counts while the 7/7 week is
    # not yet over.
    m["WeekDate"] = pd.to_datetime(m["WeekDate"])
    week_start = _this_week_start()
    m = m[m["WeekDate"] >= week_start]
    if m.empty:
        return pd.DataFrame(columns=DISCONTINUED_COLS)

    m["Region"] = m["Customer Grouping"].map(lambda g: P.region_for_group(g))
    m["_future_proj"] = pd.to_numeric(m["Projection"], errors="coerce")
    g = m.groupby(
        ["SKU", "Region", "Customer Grouping", "CUSTNMBR"], as_index=False,
    ).agg(
        First_WeekDate=("WeekDate", "min"),
        Last_WeekDate=("WeekDate", "max"),
        Original_Projection=("_future_proj", "mean"),
    )
    g["SKU Status"] = g["SKU"].map(lambda s: disc_status.get(s))
    out = g[DISCONTINUED_COLS]
    return out.sort_values(["SKU", "CUSTNMBR"]).reset_index(drop=True)


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
    raw = pd.read_excel(path, header=2)
    return _clean(raw, P)


@st.cache_data(show_spinner="Loading raw data…")
def load_raw_from_bytes(_data, name, model_path):
    """Read + clean an uploaded raw file (cached on its bytes + model)."""
    P = load_pipeline(model_path)
    raw = pd.read_excel(BytesIO(_data), header=2)
    return _clean(raw, P)


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
    'List Price (USD)' and 'Revenue Risk (USD)'. ``alpha`` / ``beta`` / ``phi``,
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


def aggregate_chart(agg, summary, weekly, anchors, view):
    """Total actual demand (historical window) flowing into total forecast (15 wks).

    Historical demand uses each SKU's forecast source (POS or Orders) so the
    actual total is comparable to the forecast total.
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


def sku_chart(sku, desc, source, agg, weekly, anchors):
    """Per-SKU: actuals (historical window, from its source) + updated forecast + original proj."""
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
        fmt[PRICE_COL] = "${:,.2f}"
    if RISK_COL in df.columns:
        fmt[RISK_COL] = "${:,.0f}"

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
    return os.path.join(HERE, "outputs", f"agent_summary_{safe_view}.json")


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

        best = payload.get("best_model")
        if best:
            mae = (payload.get("mae_by_model") or {}).get(best)
            label = f"Best model: {best}"
            if mae is not None:
                label += f"  (backtest MAE {mae:.1f})"
            if payload.get("confidence_flag"):
                st.warning(label + "  —  ⚠️ low confidence")
            else:
                st.success(label)

        if payload.get("narrative"):
            st.write(payload["narrative"])

        anomalies = payload.get("anomalies") or []
        if anomalies:
            st.markdown("**Flagged anomalies:**")
            for a in anomalies:
                # publish stores bullets as-is; add a marker only if missing.
                st.markdown(a if a.lstrip().startswith(("-", "*", "•")) else f"- {a}")


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
                "models/exponential_smoothing.py, models/xgboost.py or "
                "models/regression.py next to dashboard.py "
                "(or set DEMAND_PIPELINE)."
            )
            st.stop()

        def _on_model_change():
            # Bump the parameter nonce so the sliders are rebuilt as fresh
            # widgets keyed to the new nonce, re-reading the newly selected
            # pipeline's value= defaults. (A structural change also recomputes
            # automatically via the compute gate.)
            st.session_state["param_nonce"] = (
                st.session_state.get("param_nonce", 0) + 1
            )
            # Autofitted parameters belong to the previous model; drop them so
            # the new pipeline starts from its own file defaults.
            st.session_state.pop("autofit_params", None)

        st.radio(
            "Forecasting model", list(MODEL_OPTIONS.keys()),
            key="model_choice", on_change=_on_model_change,
            help="Switching recomputes everything with the selected pipeline.",
        )

    P = load_pipeline(pipeline_path())
    st.title("📦 Demand Projection Dashboard")
    # Header caption: the pipeline can supply its own (DASHBOARD_CAPTION, e.g.
    # the XGBoost pipeline); otherwise fall back to the smoothing-aware blurbs.
    caption = getattr(P, "DASHBOARD_CAPTION", None)
    if caption:
        st.caption(caption)
    elif _supports_smoothing(P):
        st.caption(
            "15-week Holt damped-trend forecast from the historical demand "
            "window (POS where available, else Orders). Tune the smoothing "
            "(α/β/φ) in the sidebar — changes recompute live."
        )
    else:
        tw = getattr(P, "TREND_WEIGHT", None)
        st.caption(
            "15-week forecasts from the historical demand window "
            "(POS where available, else Orders"
            + (f"; trend weight = {tw})." if tw is not None else ").")
        )

    # ----- Data source -----------------------------------------------------
    with st.sidebar:
        st.header("Data source")
        files = discover_raw_files()
        df = None
        today_str = None

        if files:
            labels = {f"{d}  ({os.path.basename(p)})": (d, p) for d, p in files}
            choice = st.selectbox("Snapshot (raw file)", list(labels.keys()))
            today_str, path = labels[choice]
            df = load_raw_from_path(path, os.path.getmtime(path), pipeline_path())
        else:
            st.info("Upload the Demand Planning Details and Plytix files below.")

        with st.expander("Upload the Demand Planning Details Projections from PowerBI", expanded=not files):
            up = st.file_uploader("all_demand_projections_*.xlsx", type=["xlsx"])
            if up is not None:
                data = up.getvalue()
                df = load_raw_from_bytes(data, up.name, pipeline_path())
                today_str = _date_from_name(up.name)

        # ----- List prices (drive revenue risk) ---------------------------
        # The Plytix export doubles as the source of each SKU's list price AND
        # its 'Active in' regions (used by the active-in check below), so we read
        # both from whichever Plytix file is in play — uploaded or on disk.
        st.header("Revenue risk")
        prices = None
        plytix_df = None
        price_file = discover_price_file()
        with st.expander("Upload Plytix file with list prices", expanded=not files):
            up_price = st.file_uploader(
                "list_prices_*.xlsx", type=["xlsx"], key="price_upload",
                help="SKU + List Price USD, plus SKU Status / SKU Type / "
                    "'Active in'. Drives revenue risk (projection difference × "
                    "list price) and the active-in check.",
            )
            if up_price is not None:
                prices = load_prices_from_bytes(
                    up_price.getvalue(), up_price.name, pipeline_path()
                )
                plytix_df = read_plytix_from_bytes(up_price.getvalue(), up_price.name)
                if prices is not None:
                    st.success(f"{len(prices):,} list prices (uploaded)")
            elif price_file is not None:
                prices = load_prices_from_path(
                    price_file, os.path.getmtime(price_file), pipeline_path()
                )
                plytix_df = read_plytix_from_path(
                    price_file, os.path.getmtime(price_file)
                )
                if prices is not None:
                    st.success(
                        f"{len(prices):,} list prices "
                        f"({os.path.basename(price_file)})"
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

    # ----- Active-in check: flag + drop out-of-region projections ----------
    # An active product should only be forecast in a region it is 'Active in'
    # (per the Plytix export). We look at the demand file itself — mapping each
    # customer to its region via the pipeline's region_for_group — and flag any
    # active product whose region is not in its 'Active in' list (e.g. ST1082,
    # active in US/CA/UK/SG/EU/AU, appearing under JP (NETDEPOT)). Those SKU ×
    # customer rows are then dropped from the data BEFORE forecasting and
    # surfaced in their own table.
    active_sku_set, sku_active_in = compute_active_products(plytix_df)
    check_ran = active_sku_set is not None
    inactive_df = compute_inactive_projections(
        df, active_sku_set, sku_active_in, P,
        anchors=(lb, lcw, ffw),
    )

    # ----- Discontinued/inactive-product check -----------------------------
    # Flag SKUs marked Discontinued/Inactive in Plytix that still carry future
    # projections (ported from discontinued_with_projections.ipynb). These SKUs
    # are then dropped from the data BEFORE forecasting — a discontinued product
    # should not appear in the summary, projections or revenue figures — and
    # surfaced in their own table below.
    disc_status = compute_discontinued_products(plytix_df)
    disc_check_ran = disc_status is not None
    discontinued_df = compute_discontinued_projections(df, disc_status, P)
    n_excluded_rows = 0
    excluded_counts_by_key = pd.Series(dtype="int64")
    if not inactive_df.empty:
        exclude_keys = {
            f"{str(s)}||{str(c)}"
            for s, c in zip(inactive_df["SKU"], inactive_df["CUSTNMBR"])
        }
        key = df["SKU"].astype(str).str.rstrip("*") + "||" + df["CUSTNMBR"].astype(str)
        drop_mask = key.isin(exclude_keys)
        n_excluded_rows = int(drop_mask.sum())
        # Per SKU||CUSTNMBR demand-row counts, so the excluded table can report
        # accurate row totals when scoped to a single region's view below.
        excluded_counts_by_key = key[drop_mask].value_counts()
        if n_excluded_rows:
            df = df[~drop_mask].reset_index(drop=True)
            logger.info(
                "Active-in check: dropped %d raw rows across %d SKU×customer×"
                "region combos not in the SKU's 'Active in' list.",
                n_excluded_rows, len(inactive_df),
            )

    # Drop discontinued/inactive SKUs entirely so they are excluded from every
    # summary, projection and revenue figure below. The status is SKU-level, so
    # the whole SKU is removed (not just the flagged customer combos). Done after
    # discontinued_df is computed above so the table still lists them.
    if disc_status:
        df_sku = df["SKU"].astype(str).str.rstrip("*")
        disc_mask = df_sku.isin(disc_status)
        n_disc_rows = int(disc_mask.sum())
        if n_disc_rows:
            n_disc_skus = df_sku[disc_mask].nunique()
            df = df[~disc_mask].reset_index(drop=True)
            logger.info(
                "Discontinued check: dropped %d raw rows across %d "
                "discontinued/inactive SKUs.",
                n_disc_rows, n_disc_skus,
            )

    # ----- View selector ---------------------------------------------------
    with st.sidebar:
        st.header("View")
        by_region = list_views(df)
        scope = st.radio(
            "Scope", [ALL_CUSTOMERS_VIEW, "By customer group"], index=0
        )
        if scope == ALL_CUSTOMERS_VIEW:
            view = ALL_CUSTOMERS_VIEW
            region = None
        else:
            # key=str: a custom pipeline's region_for_group may return non-string
            # labels; sorting by their string form keeps the selectbox from
            # crashing on mixed types (see logs.txt, 2026-07-06).
            region = st.selectbox("Region", sorted(by_region.keys(), key=str))
            view = st.selectbox("Customer group", by_region[region])

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
        run_agent = st.button(
            "Run Agent Summary",
            key="run_agent_summary",
            help="Backtests all models for this view, picks the best, and writes "
                 "an LLM narrative + flagged anomalies. Slow/expensive — runs "
                 "only when you click, never on a normal rerun.",
        )

    if run_agent:
        os.environ["LLM_PROVIDER"] = LLM_PROVIDERS[provider_label]
        with st.spinner(f"Running agentic pipeline ({provider_label})…"):
            # Import here, not at module top: keeps langgraph off the hot import
            # path for every rerun and matches the "only touched on click" rule.
            from agent.graph import build_graph

            graph = build_graph()
            final_state = graph.invoke({"view": view, "today_ts": today_ts})
        if final_state.get("errors"):
            st.error("\n".join(final_state["errors"]))
        else:
            st.toast(f"Agent finished: {final_state.get('best_model', 'n/a')}")

    # Show the last cached run for this view (from the JSON publish wrote),
    # whether it was produced just now or on an earlier click.
    _render_agent_summary(view)

    # ----- Model parameters (Holt damped-trend smoothing) ------------------
    min_weeks = None
    with st.sidebar:
        st.header("Model parameters")
        smoothing_ok = _supports_smoothing(P)
        min_weeks_ok = _supports_min_weeks(P)

        if smoothing_ok or min_weeks_ok:
            # The pipeline's own constants are the "file defaults" the reset
            # button snaps back to.
            a0 = float(getattr(P, "ALPHA", 0.5))
            b0 = float(getattr(P, "BETA", 0.3))
            p0 = float(getattr(P, "PHI", 0.85))
            mw0 = int(getattr(P, "MIN_WEEKS_FOR_TREND", 4))

            # If Autofit has run (automatically on first sight, or via the
            # button), its winning parameters become the slider defaults (the
            # nonce was bumped when they were stored, so the sliders rebuild and
            # re-read value=). Results are keyed to the model/view/snapshot they
            # were fitted on; anything else falls back to the file defaults. The
            # sliders stay fully adjustable afterwards.
            autofit = st.session_state.get("autofit_params")
            autofit_active = bool(
                autofit
                and autofit.get("model") == pipeline_path()
                and autofit.get("view") == view
                and autofit.get("today") == today_str
            )

            # ----- Auto-run Autofit on first sight --------------------------
            # So that selecting a smoothing model (or a new view / snapshot)
            # opens the sliders already on the backtested α/β/φ rather than the
            # file defaults. It runs once per (model, view, snapshot): the
            # "autofit_tried" marker below records that we've attempted it, so a
            # failed backtest isn't retried on every rerun and a good fit isn't
            # re-run on every slider move. The user can re-fit any time with the
            # Autofit button, or fine-tune the sliders by hand.
            autofit_key = (pipeline_path(), view, today_str)
            autofit_tried = st.session_state.get("autofit_tried") == autofit_key
            if (
                smoothing_ok
                and _supports_autofit(P)
                and not autofit_active
                and not autofit_tried
            ):
                st.session_state["autofit_tried"] = autofit_key
                with st.spinner("Autofitting α/β/φ for this view…"):
                    best = run_autofit(df, view, today_ts, pipeline_path(), mw0)
                if best is not None:
                    logger.info(
                        "Auto-Autofit [%s]: alpha=%.2f beta=%.2f phi=%.2f "
                        "(MAE %.2f vs %.2f with file defaults)",
                        view, best["alpha"], best["beta"], best["phi"],
                        best["mae"], best["baseline_mae"],
                    )
                    st.session_state["autofit_params"] = {
                        **best, "model": pipeline_path(),
                        "view": view, "today": today_str,
                    }
                    # Rebuild the sliders on the fitted values and recompute the
                    # forecast with them — exactly like the manual button does.
                    st.session_state["param_nonce"] = (
                        st.session_state.get("param_nonce", 0) + 1
                    )
                    st.session_state["_do_recompute"] = True
                    st.rerun()

            if smoothing_ok and autofit_active:
                a0, b0, p0 = autofit["alpha"], autofit["beta"], autofit["phi"]

            # Slider defaults come from the pipeline constants, passed via each
            # slider's value= argument. To reset reliably across Streamlit
            # versions we use a "nonce": the slider keys embed an integer that
            # we bump on reset (or model switch), which makes Streamlit build
            # brand-new widgets that re-read value=. This is more robust than
            # deleting a fixed key, which didn't reliably restore value=.
            st.session_state.setdefault("param_nonce", 0)
            nonce = st.session_state["param_nonce"]

            if smoothing_ok:
                st.caption(
                    "Lower α/β lean on more history and less on recent weeks; "
                    "lower φ flattens the projection. Moving a slider recomputes."
                )
                alpha = st.slider(
                    "α — level smoothing", min_value=0.01, max_value=0.99,
                    value=a0, step=0.01, key=f"sl_alpha_{nonce}",
                    help="Higher tracks recent weeks faster; lower ≈ a longer moving "
                         "average (effective window ≈ 2/α − 1 weeks).",
                )
                beta = st.slider(
                    "β — trend smoothing", min_value=0.0, max_value=0.99,
                    value=b0, step=0.01, key=f"sl_beta_{nonce}",
                    help="Higher re-estimates the slope from recent weeks; 0 freezes it.",
                )
                phi = st.slider(
                    "φ — trend damping", min_value=0.0, max_value=1.0,
                    value=p0, step=0.05, key=f"sl_phi_{nonce}",
                    help="Lower flattens the forecast toward the current level; "
                         "1 = plain (undamped) Holt.",
                )
            else:
                alpha = beta = phi = None

            if min_weeks_ok:
                min_weeks = st.slider(
                    "min weeks for trend", min_value=2, max_value=12,
                    value=mw0, step=1, key=f"sl_min_weeks_{nonce}",
                    help="SKUs with fewer completed weeks than this are forecast "
                         "flat at their mean instead of extrapolating a trend — "
                         "prevents runaway projections from 1–2 weeks of data. "
                         "2 disables the guard.",
                )

            # ----- Autofit: backtest a grid of α/β/φ and keep the winner ----
            if smoothing_ok and _supports_autofit(P):
                if st.button(
                    "✨ Autofit α/β/φ",
                    help="Grid-searches the smoothing parameters by backtesting: "
                         "the last few completed weeks of each SKU's history are "
                         "hidden, forecast with each parameter combination, and "
                         "compared to what actually happened (repeated from "
                         "several rolling origins). The combination with the "
                         "lowest total forecast error wins and is applied to the "
                         "sliders. You can still fine-tune the sliders afterwards.",
                ):
                    with st.spinner("Backtesting the parameter grid…"):
                        best = run_autofit(
                            df, view, today_ts, pipeline_path(), min_weeks
                        )
                    if best is None:
                        st.warning(
                            "Not enough completed weeks of history in this view "
                            "to backtest — autofit needs SKUs with at least "
                            f"~{getattr(P, 'AUTOFIT_MIN_TRAIN_WEEKS', 8) + 1} "
                            "completed weeks."
                        )
                    else:
                        logger.info(
                            "Autofit [%s]: alpha=%.2f beta=%.2f phi=%.2f "
                            "(MAE %.2f vs %.2f with file defaults; "
                            "%d SKUs, %d holdout points)",
                            view, best["alpha"], best["beta"], best["phi"],
                            best["mae"], best["baseline_mae"],
                            best["n_series"], best["n_points"],
                        )
                        st.session_state["autofit_params"] = {
                            **best, "model": pipeline_path(),
                            "view": view, "today": today_str,
                        }
                        # Mark this (model, view, snapshot) as fitted so the
                        # auto-run-on-first-sight logic doesn't re-fire over it.
                        st.session_state["autofit_tried"] = (
                            pipeline_path(), view, today_str
                        )
                        # Rebuild the sliders with the fitted values as their
                        # defaults and recompute the forecast with them.
                        st.session_state["param_nonce"] = nonce + 1
                        st.session_state["_do_recompute"] = True
                        st.rerun()

                if autofit_active:
                    improve = autofit["baseline_mae"] - autofit["mae"]
                    pct = (
                        f" ({improve / autofit['baseline_mae'] * 100:.0f}% better)"
                        if autofit["baseline_mae"] > 0 and improve > 0 else ""
                    )
                    st.success(
                        f"Autofit applied: α={autofit['alpha']:g}, "
                        f"β={autofit['beta']:g}, φ={autofit['phi']:g} — "
                        f"backtest error {autofit['mae']:.1f} u/wk vs "
                        f"{autofit['baseline_mae']:.1f} with file defaults"
                        f"{pct}. Scored on {autofit['n_series']} SKUs, "
                        f"{autofit['folds']} rolling folds of "
                        f"{autofit['holdout_weeks']} held-out weeks."
                    )

        else:
            alpha = beta = phi = None
            st.info(
                "This pipeline's fit_regression doesn't accept α/β/φ or min-weeks "
                "overrides, so its own module-level constants are used."
            )

        # Recompute is a manual action: moving a slider no longer rebuilds the
        # whole forecast on every tick. The button sets a flag (via callback,
        # so it's honoured on the very next run) that the compute gate reads.
        def _request_recompute():
            st.session_state["_do_recompute"] = True

        st.button(
            "🔄 Recompute forecast", type="primary", on_click=_request_recompute,
            help="Apply the current parameters. Structural changes (view, "
                 "model, snapshot, data) recompute automatically.",
        )

    # ----- Compute (manual, with a progress bar) ---------------------------
    # The forecast is cached in session_state and only (re)built when:
    #   * there is no result yet (first load), or
    #   * a structural input changed (view / model / snapshot / data / prices), or
    #   * the user pressed "Recompute".
    # Slider changes alone leave the last result on screen and surface a
    # "parameters changed — recompute to apply" notice.
    price_marker = None if prices is None else int(len(prices))
    structural_sig = (
        view, pipeline_path(), today_str, price_marker, n_excluded_rows
    )
    param_sig = (alpha, beta, phi, min_weeks)

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
            if view == ALL_CUSTOMERS_VIEW and summary is not None and not summary.empty:
                def _bump(done, total, group):
                    frac = 0.4 + 0.55 * (done / max(total, 1))
                    prog.progress(
                        min(frac, 0.98),
                        text=f"Per-customer forecast… ({done}/{total})",
                    )
                by_cust = compute_by_customer(
                    df, today_ts, pipeline_path(),
                    prices, alpha, beta, phi, min_weeks, progress_cb=_bump,
                )
            prog.progress(1.0, text="Done")
        finally:
            prog.empty()

        st.session_state["fc_result"] = (summary, weekly, agg, by_cust)
        st.session_state["fc_structural"] = structural_sig
        st.session_state["fc_params"] = param_sig
    else:
        summary, weekly, agg, by_cust = stored

    if summary is None or summary.empty:
        st.error(
            f"No POS or Orders in the 8-week window for **{view}** — "
            "nothing to forecast."
        )
        st.stop()

    # If the sliders were moved since the on-screen result was built, the view
    # is stale until the user recomputes. Flag it rather than silently updating.
    if st.session_state.get("fc_params") != param_sig:
        st.info(
            "Parameters changed since this forecast was built — click "
            "**🔄 Recompute forecast** in the sidebar to apply them.",
            icon="⚠️",
        )

    # ----- Header / windows -------------------------------------------------
    st.subheader(view)
    model_bits = []
    if alpha is not None:
        model_bits.append(f"α = {alpha:.2f}, β = {beta:.2f}, φ = {phi:.2f}")
    if min_weeks is not None:
        model_bits.append(f"min weeks for trend = {min_weeks}")
    if model_bits:
        st.caption("Model in use — " + "; ".join(model_bits))
    w1, w2 = st.columns(2)
    # The fixed-window regression pipeline always fits exactly the last 8
    # completed weeks, so we can state that. The other pipelines expose a
    # LOOKBACK_WEEKS mechanism (all-history by default), so the window isn't a
    # fixed 8 weeks — leave the count off rather than mislabel it.
    hist_span = f"**Historical window** &nbsp; {lb.date()} → {lcw.date()}"
    if not hasattr(P, "LOOKBACK_WEEKS"):
        hist_span += " <span style='color:#64748b'>(8 completed weeks)</span>"
    w1.markdown(hist_span, unsafe_allow_html=True)
    fc_weeks = pd.to_datetime(weekly["WeekDate"])
    w2.markdown(
        f"**Forecast window** &nbsp; {ffw.date()} → "
        f"{fc_weeks.max().date()} "
        f"<span style='color:#64748b'>({fc_weeks.nunique()} weeks)</span>",
        unsafe_allow_html=True,
    )

    # ----- KPIs -------------------------------------------------------------
    avg_col = resolve_avg_col(summary)
    total_avg = summary[avg_col].sum()
    total_updated = summary["Updated Projection Average"].sum()
    total_initial = summary["Initial Projection Average"].sum()
    diff = total_updated - total_initial
    n_orders = int((summary.get("Data Source") == "Orders").sum()) \
        if "Data Source" in summary.columns else 0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric(
        "SKUs Forecasted", f"{len(summary):,}",
        help=f"{n_orders} forecast from Orders (no POS)" if n_orders else None,
    )
    k2.metric("Avg. Weekly Demand", f"{total_avg:,.0f}")
    k3.metric("Initial Projection (avg/wk)", f"{total_initial:,.0f}")
    k4.metric("Updated Projection (avg/wk)", f"{total_updated:,.0f}")
    k5.metric(
        "Projection Difference (avg/wk)", f"{diff:+,.0f}",
        delta=f"{(diff / total_initial * 100):+.1f}%" if total_initial else None,
    )
    has_risk = RISK_COL in summary.columns and summary[RISK_COL].notna().any()
    if has_risk:
        net_risk = summary[RISK_COL].sum()
        k6.metric(
            "Revenue Risk (net)", f"${net_risk:+,.0f}",
            help="Σ (projection difference × list price) over priced SKUs. "
                 "Negative = forecast fell below the original projection.",
        )
    else:
        k6.metric(
            "Revenue Risk (net)", "—",
            help="Load a list_prices_*.xlsx (sidebar) to enable revenue risk.",
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
    st.plotly_chart(
        aggregate_chart(agg, summary, weekly, (lb, lcw, ffw), view),
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
        st.plotly_chart(
            sku_chart(sku, desc, source, agg, weekly, (lb, lcw, ffw)),
            width="stretch",
        )
    with cR:
        st.metric("Data source", source)
        avg_col = resolve_avg_col(summary)
        st.metric(f"{avg_window_phrase(avg_col)} {source} avg", f"{row[avg_col]:,.1f}")
        st.metric("Updated proj.", f"{row['Updated Projection Average']:,.0f}")
        sysv = row.get("Initial Projection Average")
        st.metric("Original proj.", "—" if pd.isna(sysv) else f"{sysv:,.0f}")
        st.metric(
            "Difference",
            f"{row['Projection Difference']:+,.0f}"
            if pd.notna(row["Projection Difference"]) else "—",
        )
        if RISK_COL in summary.columns:
            pv = row.get(PRICE_COL)
            rv = row.get(RISK_COL)
            st.metric("List price", "—" if pd.isna(pv) else f"${pv:,.2f}")
            st.metric(
                "Revenue risk",
                "—" if pd.isna(rv) else f"${rv:+,.0f}",
                help="Projection difference × list price.",
            )
        if "Top Volume Customer Groups" in summary.columns:
            st.markdown("**Top volume groups**")
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
        style_summary(summary_table), width="stretch", hide_index=True
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
    if view == ALL_CUSTOMERS_VIEW:
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
                style_summary(by_cust_table), width="stretch", hide_index=True
            )
            st.download_button(
                "⬇️ Download the summary table by SKU and Customer",
                data=summary_to_excel(by_cust_table),
                file_name=f"ALL_CUSTOMERS_demand_projections_{today_str}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_by_customer",
            )

    # ----- Excluded: active products projected in non-active regions --------
    render_inactive_section(
        view, region, check_ran, inactive_df,
        excluded_counts_by_key, n_excluded_rows, today_str,
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
    HEADER = "### Active products with future projections in locations they are not active in"
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
    st.dataframe(show, width="stretch", hide_index=True)
    st.download_button(
        "⬇️ Download the excluded (inactive-region) projections table",
        data=summary_to_excel(show, sheet_name="inactive_projections"),
        file_name=f"inactive_projections_{today_str}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_inactive_projections",
    )


def render_discontinued_section(view, region, disc_check_ran, discontinued_df,
                                today_str):
    """Table of Discontinued/Inactive products that still carry projections.

    Ported from discontinued_with_projections.ipynb. In a "By customer group"
    view, only rows whose region matches the selected region are shown (e.g. an
    EU view won't list AAFES, a US customer); ALL CUSTOMERS shows every region.
    """
    HEADER = "### Inactive/discontinued products with future projections"
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

    if table_df.empty:
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

    n_skus = table_df["SKU"].nunique()
    scope_note = f" for {region}" if region_scoped else ""
    st.caption(
        f"Flagged{scope_note}: {n_skus:,} distinct SKUs marked Discontinued or "
        "Inactive in Plytix that still carry future projections (future weeks "
        "only)."
    )

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
    disc = disc[[
        'SKU', 'SKU Status', 'Region', 'Customer Grouping',
        'First_WeekDate', 'Last_WeekDate', 'Original_Projection',
    ]].rename(columns={"Original_Projection": "Original Projection (future avg/wk)"})

    st.dataframe(disc, width="stretch", hide_index=True)
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
            st.caption(f"Full traceback is also recorded in {LOG_PATH}.")
        st.stop()


if __name__ == "__main__":
    _run()
