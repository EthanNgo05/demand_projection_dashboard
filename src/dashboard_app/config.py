"""Constants, model catalog, and pure view/format helpers (streamlit-free)."""
import os

import pandas as pd


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
# HERE is the src/ directory (this module lives in src/dashboard_app/, one level
# down), kept anchored there so the model/extract-script paths below resolve
# exactly as they did when this code lived in src/dashboard.py.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Repo root (parent of src/) — where the data folders raw_inputs/, outputs/ and
# logs/ live. HERE is used for sibling CODE (models/, extract script); REPO_ROOT
# for DATA that stays at the repo root.
REPO_ROOT = os.path.dirname(HERE)

# The simplehuman brand logo, bundled as a committed asset (assets/ sits beside
# the code and deploys to Streamlit Cloud). Used as the browser favicon and the
# on-page header mark, replacing the old 📦 emoji.
LOGO_PATH = os.path.join(REPO_ROOT, "assets", "sh_logo.jpg")

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

# Title-cased model names for the UI. The MODEL_OPTIONS keys stay the canonical
# model IDs (the agent's best_model, agent/config.py, agent_summary_*.json), so
# we only prettify what the planner sees — never the stored identifier.
MODEL_DISPLAY = {
    "Holt's (double) exponential smoothing": "Holt's Exponential Smoothing",
    "Holt-Winters (triple) exponential smoothing": "Holt-Winters Exponential Smoothing",
    "TSB (intermittent demand)": "TSB",
}


def model_display(label):
    """The UI label for a model ID (title-cased); unknown labels pass through."""
    return MODEL_DISPLAY.get(label, label) if label is not None else label


ALL_CUSTOMERS_VIEW = "All customers (combined)"

# A combined view that, unlike ALL_CUSTOMERS_VIEW (one model over all SKUs
# summed), forecasts each customer group with ITS OWN backtest-winning model —
# the model published in that group's agent_summary_<group>.json — and stitches
# every group's per-SKU rows into one table with a "Model Used" column. It is the
# "best model per group, combined" table, so it depends on the agent batch having
# run for every group (the "Agent Summary (all views)" button / `agent.batch`).
BEST_MODEL_COMBINED_VIEW = "Combined (best model per group)"
MODEL_USED_COL = "Model Used"

# A scan-everything view that flags SKUs whose recent actual sell-through (POS,
# falling back to Orders) has diverged sharply from the existing SYSTEM
# projection (the plan of record) — not from our model forecast. It is a pure
# actuals-vs-plan comparison, so it needs no forecasting fit and does not depend
# on the agent batch. Like the other scope constants this string doubles as the
# stable view ID; unlike them it is a dashboard-only analysis scope and is never
# returned by list_views/enumerate_views, so the agent never forecasts it.
EXCEPTIONS_VIEW = "Exceptions"

# A pin-board view: users star (SKU, Customer Grouping) combinations on a single
# shared watchlist and jump straight to their projection detail. Like
# EXCEPTIONS_VIEW this string is a dashboard-only top-level scope token — it never
# reaches the compute path and is never returned by list_views/enumerate_views,
# so the agent never forecasts it. Its detail numbers reuse the best-model-per-
# group table (BEST_MODEL_COMBINED_VIEW).
WATCHLIST_VIEW = "Watchlist"

# Dashboard-only top-level selector token for the standard single-model scope. It
# is NOT a forecast `view` ID — it never reaches the compute path and is never
# returned by list_views/enumerate_views, so the agent never forecasts it. Two
# plain selectboxes rendered under the tab strip (Region, then Customer group)
# resolve it to a real view ID: ALL_CUSTOMERS_VIEW, a region_all_view(<region>)
# rollup, or a bare Customer Grouping. `scope` itself stays QUICK_VIEW for the
# whole run, so the model selector and the page body branch on it directly.
QUICK_VIEW = "Quick Projections"

# Friendly labels shown in the view tab strip. The keys are the stable internal
# view IDs (also the agent-summary filenames / agent config), so we rename only
# what the planner sees, never the ID.
SCOPE_LABELS = {
    QUICK_VIEW: "Quick Projections",
    BEST_MODEL_COMBINED_VIEW: "Optimized Projections",
    EXCEPTIONS_VIEW: "Exceptions",
    WATCHLIST_VIEW: "Watchlist",
}

# One-line description shown as a caption under the view tab strip — describes the
# *active* tab only (the tabs replaced the old "About these views" expander that
# listed all four at once). Keyed by the same internal view IDs as SCOPE_LABELS.
SCOPE_CAPTIONS = {
    QUICK_VIEW: (
        "Standard 15-week forecasts using the model selected below — all customer "
        "groups combined, or drill into a single fulfillment region or customer "
        "group with the dropdowns above."
    ),
    BEST_MODEL_COMBINED_VIEW: (
        "Forecasts each customer group with its own most-accurate model, combined "
        "into one table. Requires model analysis to have been run for all groups."
    ),
    EXCEPTIONS_VIEW: (
        "Scans every customer group for SKUs whose recent actual sell-through has "
        "diverged sharply from the existing system projection (the plan of "
        "record). Model-agnostic — no forecast is run."
    ),
    WATCHLIST_VIEW: (
        "Star SKU / customer-group combinations on named lists to pin their "
        "projection detail (best-model numbers + an actuals-vs-plan chart). The "
        "active list drives the ★ marker on every table; lists are shared across "
        "everyone using this dashboard."
    ),
}

# Per-region rollup views: "All Customers - <region label>" combines every
# customer group in one region into a single forecast (e.g.
# "All Customers - AU (ACR)" = Web Sales - AU + Others - AU). The region is
# embedded in the view string so everything keyed on `view` — cache keys,
# session signatures, headers, filename mangling, the agent's summary path —
# works unchanged. Mirrored in agent/config.py (must match exactly).
REGION_ALL_PREFIX = "All Customers - "

# UI-only sentinel for the Quick Projections Region dropdown's "don't filter by
# region" option. NEVER a view ID and never persisted: picking it just means the
# Customer-group dropdown offers ALL_CUSTOMERS_VIEW plus every group, so the
# selected value is always one of the real view IDs.
ALL_REGIONS = "All regions"


def region_all_view(region):
    """The synthetic per-region combined view string for ``region``."""
    return f"{REGION_ALL_PREFIX}{region}"


def region_from_view(view):
    """Region label if ``view`` is a per-region rollup, else None."""
    if isinstance(view, str) and view.startswith(REGION_ALL_PREFIX):
        return view[len(REGION_ALL_PREFIX):]
    return None


def quick_group_label(view):
    """Friendly label for a Quick Projections Customer-group option.

    Derived from the view string itself, never from the separately-selected
    region: the Customer-group selectbox is re-optioned whenever Region changes,
    and Streamlit decides whether a stored value is still offered by comparing its
    FORMATTED form — so a label that closed over the region would make that reset
    behaviour depend on which region happened to be selected.
    """
    if view == ALL_CUSTOMERS_VIEW:
        return "All customers (combined)"
    region = region_from_view(view)
    return f"All customers in {region} (combined)" if region is not None else str(view)


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

# The two descriptive-average column names, and the ONLY spellings the UI shows.
# The model files' AVG_COL_LABEL / DISPLAY_NAMES match these exactly so
# ``compute.attach_descriptive_averages`` replaces the model's column in place
# instead of leaving two differently-named copies of one figure on the same table.
#
# They live HERE rather than in compute.py (which re-exports them for the callers
# that have always imported them from there) because KPI_ORDER below needs them
# and this module must stay streamlit-free — compute.py imports streamlit, and it
# imports this module, so the dependency can only run one way.
ALL_TIME_AVG_COL = "All-Time POS/Orders Average"
EIGHT_WK_AVG_COL = "8-Week POS/Orders Average"
# Percent change from the 8 weeks before last to the last 8 completed weeks — is
# this SKU/customer accelerating or decaying? Derived in
# ``compute._descriptive_averages`` from a prior-8-week average that is
# deliberately NOT published as a column of its own: a third average column beside
# the two above would recreate the same-window/different-number confusion that
# ALL_TIME_AVG_COL's rename existed to remove.
TREND_COL = "Recent Trend"
# SKU-level supply figures (constant across a SKU's customer rows). ONHAND_COL is
# the SKU's current total On Hand; WOS_COL is that divided by the SKU's total
# CURRENT weekly projection across all customers. Same definition everywhere it
# appears — there is deliberately no second "WOS vs the updated forecast".
ONHAND_COL = "On Hand"
WOS_COL = "WOS Impact"


# --------------------------------------------------------------------------- #
# Detail-card KPI tiles: one canonical order, one set of tooltips              #
# --------------------------------------------------------------------------- #
# Every view picks WHICH fields its detail card shows (the ``*_CARD_COLS`` lists in
# dashboard.py / kpis.py / exceptions.py / watchlist_view.py); this list decides
# WHERE each one lands, so a field sits in the same place no matter which view
# opened the card. Before this existed each view ordered its own card and a
# planner had to re-learn every one.
#
# The sequence reads as a story: who is this -> what did it sell -> what is
# planned -> what is it worth -> what is on the shelf.
#
# Names owned by exceptions.py (Status, First Week Spike, Weeks Since Spike,
# Container Impact) appear here as literals rather than imports: exceptions.py
# imports tables.py, which imports this module, so importing back would cycle.
# They mirror the constants at the top of exceptions.py — keep the two in sync.
KPI_ORDER = [
    # --- who ---------------------------------------------------------------
    "Customer Grouping", "Customer", "Region", "Region Code",
    "Data Source", "Active in", MODEL_USED_COL, "Status",
    # --- what it sold ------------------------------------------------------
    "Weeks with data", ALL_TIME_AVG_COL, EIGHT_WK_AVG_COL, TREND_COL,
    "First Week Spike", "Weeks Since Spike",
    # --- what is planned ---------------------------------------------------
    "Current Projection Average", "Updated Projection Average",
    "Projection Difference", "% Deviation",
    # --- what it is worth --------------------------------------------------
    PRICE_COL, RISK_COL, "Projected Revenue",
    # --- what is on the shelf ----------------------------------------------
    ONHAND_COL, WOS_COL, "Container Impact",
]

# Identity fields: short strings, not measurements. Rendered in the same tile as
# everything else but with a smaller, wrapping value, so a long value like
# "Holt-Winters (triple) exponential smoothing" reads as a caption instead of a
# headline. Everything not listed here is treated as a stat (big tabular figures).
KPI_TEXT_FIELDS = {
    "Customer Grouping", "Customer", "Region", "Region Code",
    "Data Source", "Active in", MODEL_USED_COL, "Status", "First Week Spike",
}

# Tile tooltips. Wording is lifted from the page-top KPI row's help text in
# kpis._render_kpis where the same quantity appears there, so the card and the KPI
# row explain a number the same way. Fields with no entry simply get no tooltip.
KPI_HELP = {
    ALL_TIME_AVG_COL: (
        "Observed weekly demand (POS/Orders) averaged over this SKU's whole "
        "history with this customer — total demand ÷ weeks from its first sale "
        "through the last completed week, so silent weeks count as zeros."
    ),
    EIGHT_WK_AVG_COL: (
        "Observed weekly demand over the last 8 completed weeks — the recent "
        "run-rate. Compare against the all-time average to see if demand is "
        "running hot or cold."
    ),
    TREND_COL: (
        "Percent change from the 8 weeks before last to the last 8 completed "
        "weeks. “New” means there were no sales in the earlier window, so there "
        "is no baseline to compare against."
    ),
    "Weeks with data": "Completed weeks with any POS/Orders activity.",
    "Current Projection Average": (
        "Mean of the existing system projection over the forecast horizon (the "
        "15 future weeks)."
    ),
    "Updated Projection Average": (
        "Mean of this model's updated forecast over the same 15 future weeks."
    ),
    "Projection Difference": (
        "Updated forecast − current projection, per week. Negative means the "
        "forecast fell below the projection already in the system."
    ),
    "% Deviation": (
        "Projection difference as a percent of the current projection — the same "
        "gap in relative terms, so a small SKU and a large one are comparable."
    ),
    PRICE_COL: "List price per unit (Plytix).",
    RISK_COL: (
        "Projection difference × list price. Negative = forecast fell below the "
        "original projection. Blank when the SKU has no list price."
    ),
    "Projected Revenue": (
        "List price × this row's updated weekly-average forecast — the gross "
        "value at list price of the demand being forecast."
    ),
    ONHAND_COL: (
        "The SKU's current TOTAL On Hand across all warehouses — a SKU-level "
        "figure, the same on every one of this SKU's customer rows."
    ),
    WOS_COL: (
        "Weeks of Supply = the SKU's total On Hand ÷ its total current weekly "
        "projection summed across ALL customers. SKU-level, so it is the same on "
        "every one of this SKU's customer rows. Blank when On Hand is unavailable."
    ),
    "Container Impact": (
        "The SKU's total cumulative spike units ÷ its Container Load — how many "
        "containers of unplanned demand this represents. SKU-level."
    ),
    "Data Source": (
        "Which signal the forecast used: POS (sell-through) where the SKU has "
        "any, else Orders."
    ),
    MODEL_USED_COL: "The model that won this customer group's 5-model backtest.",
    "Status": "Whether the recent run-rate sits above, below, or on the plan.",
}


def kpi_sort(cols):
    """Order ``cols`` by ``KPI_ORDER``, keeping unknown names at the end.

    Unknown names sort last (stably, in the order given) rather than raising, so a
    view can pass a bespoke column without this module knowing about it. The
    ordering test asserts the shipped ``*_CARD_COLS`` lists contain no unknowns —
    a name that falls off the end is nearly always a typo, not an intent.
    """
    rank = {c: i for i, c in enumerate(KPI_ORDER)}
    return sorted(cols, key=lambda c: rank.get(c, len(KPI_ORDER)))


def fmt_dollar(v, decimals=0, signed=False):
    """Format a dollar amount with the sign OUTSIDE the $ (e.g. -$500, +$500).

    Python's ``{:+,.0f}`` puts the sign after the ``$`` (``$-500``); this keeps
    it in front so negatives read like ``-$500``.
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    sign = "-" if v < 0 else ("+" if signed else "")
    return f"{sign}${abs(v):,.{decimals}f}"


def bounded_put(store, key, value, cap):
    """Insert ``key -> value`` into an insertion-ordered dict, evicting the
    oldest entries once ``cap`` is exceeded. Popping the key first gives
    refreshed keys move-to-end (LRU-ish) semantics.

    Lives here (rather than in dashboard.py, where it started) so every module
    holding a per-session cache in ``st.session_state`` can bound it the same
    way — dashboard.py's forecast/autofit caches and exceptions.py's spike-scan
    cache all share these semantics.
    """
    store.pop(key, None)
    store[key] = value
    while len(store) > cap:
        del store[next(iter(store))]


# Chart palette -- actuals are the anchor (solid), the two projections are
# de-emphasised dashed/dotted lines so the eye reads "history -> forecast".
C_ACTUAL = "#2563eb"   # blue   - historical actual demand (POS or Orders)
C_UPDATED = "#ea580c"  # orange - our recomputed 15-week forecast
C_ORIGINAL = "#9ca3af"   # grey   - the existing projection
C_GRID = "rgba(148,163,184,0.18)"
