"""Constants, model catalog, and pure view/format helpers (streamlit-free)."""
import datetime
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

# A backward-looking analytics view: what sell-through ALREADY did, with no
# forecast anywhere in it. Every other view exists to produce or interrogate the
# 15-week projection; this one answers the descriptive questions that come before
# that ("how much revenue have we booked this year?", "is this quarter up or down
# on last?", "which SKUs actually pay the bills?", "when does our season hit?").
# Like EXCEPTIONS_VIEW and WATCHLIST_VIEW this string is a dashboard-only top-level
# scope token -- it never reaches the compute path and is never returned by
# list_views/enumerate_views, so the agent never forecasts it.
#
# It is the one view that reads ExclusionResult.df_with_discontinued rather than
# .df: a discontinued SKU is correctly never projected, but it still SOLD for the
# years it was active, and dropping that history would understate every prior-year
# total and flatter every year-over-year comparison. See agent/data_io.py.
HISTORICAL_VIEW = "Historical Summary"

# Friendly labels shown in the view tab strip. The keys are the stable internal
# view IDs (also the agent-summary filenames / agent config), so we rename only
# what the planner sees, never the ID.
SCOPE_LABELS = {
    QUICK_VIEW: "Quick Projections",
    BEST_MODEL_COMBINED_VIEW: "Optimized Projections",
    EXCEPTIONS_VIEW: "Exceptions",
    HISTORICAL_VIEW: "Historical Summary",
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
    HISTORICAL_VIEW: (
        "What sell-through has already done — no forecast anywhere in this view. "
        "Revenue and unit KPIs across YTD / rolling windows with year-over-year "
        "comparisons, plus trend, mix, mover and seasonality charts. Unlike the "
        "projection views this one counts discontinued SKUs for the years they "
        "were active, so its totals run higher than theirs."
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

# ``Data Source`` at SKU level, when the SKU's customer groups don't agree.
#
# Which signal a forecast uses is decided PER SERIES: POS where that series has
# any, else Orders (the models' fallback). Customers who report no sell-through
# are therefore forecast from Orders while their neighbours are forecast from POS,
# so a SKU's total genuinely sums two measurement bases — on the live snapshot that
# is 363 of 566 SKUs. The alternative is worse: resolving one source per SKU is
# what made the old combined fit drop every Orders-only customer's demand outright.
# So the total keeps every customer and says so here. Per-customer rows keep their
# own single source; only the SKU-level roll-up can read MIXED_SOURCE.
MIXED_SOURCE = "Mixed (POS + Orders)"


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
# "Key SKU" is a literal for the same reason (it is keyskus.KEY_SKU_COL, and
# keyskus.py imports datasources.py which imports this module).
KPI_ORDER = [
    # --- who ---------------------------------------------------------------
    "Customer Grouping", "Customer", "Region", "Region Code",
    "Data Source", "Active in", MODEL_USED_COL, "Status",
    # Not columns on any frame — the detail card derives these two from the
    # watchlist and the key-SKU snapshot (tables._render_row_detail). They sit
    # with the other identity fields because that is what they are: what this
    # row IS, not what it measured.
    "Watchlist", "Key SKU",
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
    "Watchlist", "Key SKU",
}

# Tile tooltips. Wording is lifted from the page-top KPI row's help text in
# kpis._render_kpis where the same quantity appears there, so the card and the KPI
# row explain a number the same way. Fields with no entry simply get no tooltip.
KPI_HELP = {
    "Watchlist": (
        "Whether this SKU + customer group is on the active watchlist. Starred "
        "rows are also marked with a ★ on every table."
    ),
    "Key SKU": (
        "Whether the planning team flags this SKU as a key item (KeyItem = Yes "
        "in the week-of-supply parameters). Key rows carry a blue “Key” chip on "
        "every table."
    ),
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
        f"any, else Orders. “{MIXED_SOURCE}” on a SKU total means its customer "
        "groups differ — the ones reporting sell-through are forecast from POS "
        "and the rest from their Orders, so the total covers every customer."
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


def fmt_compact(v, money=False, decimals=1):
    """Abbreviate a large number for a KPI tile: 1.2K / $210.0M / $2.1B.

    SCOPE: KPI tiles ONLY. charts.py deliberately sets ``tickformat=",.0f"`` to get
    plain grouped integers "instead of Plotly's default SI abbreviation", and that
    decision stands -- axis ticks and table cells must stay exact, because reading a
    value off an axis is a precision job. A tile is a different job: twelve of them
    sit side by side in quarter-width columns, and "$210,022,936" next to "24%" gives
    the grid no common rhythm and nothing comparable at a glance. The exact figure is
    always one hover (or one click into the breakdown) away, so nothing is lost.

    One decimal place at every magnitude on purpose -- "$210.0M" and "$2.1B" occupy
    the same width, which is the entire point. ``None``/NaN render as an em dash,
    matching ``fmt_dollar``.
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    try:
        val = float(v)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(val):
        return "—"
    prefix = "$" if money else ""
    sign = "-" if val < 0 else ""
    n = abs(val)
    for cutoff, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= cutoff:
            return f"{sign}{prefix}{n / cutoff:,.{decimals}f}{suffix}"
    # Below 1,000 an abbreviation would lose information for no width saving.
    return f"{sign}{prefix}{n:,.0f}"


def _to_datetime(ts):
    """Coerce any timestamp shape this app produces into a ``datetime``, or None.

    Four mechanisms wrote timestamps before this existed -- ``pd.Timestamp``,
    ``datetime``, ``time.time()`` epoch floats, and the ``"%Y-%m-%d %H:%M:%S"``
    strings that ``refresh.py`` persists to its lock files -- so the display
    helpers below need one funnel rather than a format assumption per call site.

    Returns None (never raises) for anything unparseable, so a cosmetic
    timestamp can NEVER take down a render path. The lock-file string in
    particular is read off a network share and may be empty or half-written.
    """
    if ts is None or ts == "":
        return None
    # Reject containers up front. pd.to_datetime([]) returns an empty index
    # rather than raising, and pd.isna() on that yields an ARRAY, whose truth
    # value then raises out of the very guard meant to catch bad input.
    if not pd.api.types.is_scalar(ts):
        return None
    # Epoch seconds (time.time()) -- checked before the pandas path, which would
    # otherwise read a bare float as nanoseconds since 1970.
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        try:
            if pd.isna(ts):
                return None
            return datetime.datetime.fromtimestamp(float(ts))
        except (ValueError, OSError, OverflowError):
            return None
    try:
        dt = pd.to_datetime(ts)
    except (ValueError, TypeError, OverflowError):
        return None
    if dt is None or pd.isna(dt):
        return None
    return dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt


def _time_of_day(dt):
    """``3:15 PM`` from a datetime.

    ``%I`` is zero-padded (``03``), so the leading zero is stripped; midnight and
    noon both render as ``12``, where the strip would otherwise leave an empty
    hour.
    """
    hour = dt.strftime("%I").lstrip("0") or "12"
    return dt.strftime(f"{hour}:%M %p")


def fmt_when(ts):
    """Format a timestamp as ``2026-08-11 3:18 PM``.

    Absolute, never relative. "yesterday at 3:18 PM" made the reader do date
    arithmetic to answer the only question they have ("which snapshot is this?"),
    and it went stale in a tab left open overnight -- a page open past midnight
    still claimed "today". The ISO date matches the ``%Y-%m-%d`` form used for
    snapshot filenames and lock files, so a caption and a filename can be
    compared by eye.

    Seconds are dropped -- they are never read. Returns an em dash, never raises,
    for anything unparseable.
    """
    dt = _to_datetime(ts)
    if dt is None:
        return "—"
    return f"{dt.strftime('%Y-%m-%d')} {_time_of_day(dt)}"


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

# --------------------------------------------------------------------------- #
# Categorical palette (Historical Summary breakdowns)                          #
# --------------------------------------------------------------------------- #
# The three colors above are SEMANTIC -- each means one specific thing, and they
# are not a categorical set. The breakdown charts (region donut, region stacked
# area, SKU-type donut) encode IDENTITY across up to eight categories, which needs
# a palette whose members stay distinguishable from each other, not just from the
# background.
#
# These eight hues were validated with the dataviz skill's checker against BOTH of
# this app's real surfaces -- the light theme's #ffffff and the dark theme's
# #141416 (.streamlit/config.toml) -- and pass every gate in both:
#   lightness band, chroma floor, contrast >= 3:1 vs surface,
#   worst adjacent CVD dE 8.4 (protan; >= 8 target),
#   worst adjacent normal-vision dE 19.3 (>= 15 floor).
# One mode-invariant set rather than a light/dark pair, matching how C_ACTUAL and
# friends already work: charts.py deliberately does not branch on the server-side
# theme (that read can lag the displayed theme), so a trace color has to be legible
# either way. The light-stepped variant of these hues FAILS the dark lightness band,
# which is why these specific steps are the ones written down.
#
# ORDER IS THE CVD-SAFETY MECHANISM, not decoration -- the checks above are on
# ADJACENT pairs, so reordering invalidates them. Re-run the validator before
# changing the sequence. Assign slots in order and never cycle: a ninth category
# folds into "Other" (see historical_metrics.OTHER_LABEL) rather than reusing slot 1.
C_CATEGORICAL = (
    "#3987e5",  # 1 blue
    "#d95926",  # 2 orange
    "#199e70",  # 3 aqua
    "#c98500",  # 4 yellow
    "#d55181",  # 5 magenta
    "#008300",  # 6 green
    "#9085e9",  # 7 violet
    "#e66767",  # 8 red
)

# Translucent neutral: lightens against a dark surface and darkens against a light
# one, so one value separates adjacent fills (stacked areas, pie slices) in both
# themes (the app's trace colours are deliberately theme-invariant -- see charts.py).
C_SEPARATOR = "rgba(128,128,128,0.45)"

# Grey used for a fold-to-tail bucket, so "Other" never impersonates a real
# category by borrowing a categorical slot.
C_OTHER = C_ORIGINAL

# Sequential blue ramp for continuous magnitude (the month x year heatmap).
# One hue, light -> dark, per the sequential rule. Heatmap cells tile the whole
# plot area so the surface never shows through, which is what lets a single ramp
# serve both themes; cell labels pick their own ink from the cell's value (see
# historical_charts._cell_text_color) rather than from the page theme.
C_SEQUENTIAL_BLUE = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
)

# Growth / decline for the year-over-year movers chart. Deliberately the SAME two
# hues the app already uses for money moving up or down (charts._RISK_POS /
# _RISK_NEG, tables.colour_diff), so a planner reads one visual language across
# every view. Red/green alone is not colorblind-safe, so this pairing is never the
# only signal: the movers chart sorts gainers and decliners into separate blocks
# and direct-labels every bar with a signed dollar figure.
C_GROWTH = "#16a34a"   # green - revenue up vs the prior year
C_DECLINE = "#dc2626"  # red   - revenue down vs the prior year


def categorical_color_map(keys):
    """Stable ``{key: hex}`` over ``C_CATEGORICAL``, assigned in slot order.

    Colour follows the ENTITY, never its rank: the mapping is built from the keys
    sorted by name, so filtering a region out of the Historical Summary never
    repaints the regions that remain (assigning by position in a
    largest-first list would). Keys past slot 8 all get the last slot -- callers
    are expected to have folded the tail into a single "Other" bucket before
    calling, so that is a backstop, not the intended path.
    """
    ordered = sorted({str(k) for k in keys})
    return {
        k: C_CATEGORICAL[min(i, len(C_CATEGORICAL) - 1)]
        for i, k in enumerate(ordered)
    }
