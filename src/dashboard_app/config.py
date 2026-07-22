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

# Friendly labels shown in the Scope selector. The keys are the stable internal
# view IDs (also the agent-summary filenames / agent config), so we rename only
# what the planner sees, never the ID.
SCOPE_LABELS = {
    ALL_CUSTOMERS_VIEW: "Executive Overview",
    BEST_MODEL_COMBINED_VIEW: "Optimized Projections",
    "By region": "By Region",
    EXCEPTIONS_VIEW: "Exceptions",
}

# One-line description shown as a caption under the view tab strip — describes the
# *active* tab only (the tabs replaced the old "About these views" expander that
# listed all four at once). Keyed by the same internal view IDs as SCOPE_LABELS.
SCOPE_CAPTIONS = {
    ALL_CUSTOMERS_VIEW: (
        "Forecasts all customer groups as one combined demand series using the "
        "model selected below."
    ),
    "By region": (
        "Forecasts only the selected fulfillment region (or a customer group "
        "within it) using the model selected below."
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
}

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
