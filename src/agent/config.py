"""Agent configuration: model registry, view labels, thresholds, env loading.

Mirrors dashboard.py's MODEL_OPTIONS (dashboard.py:90-101). Once Phase 2 makes
importing dashboard.py side-effect-free, import from there instead so the two
never drift. Until then the three paths are duplicated, with the same
existence filter so a missing optional model degrades identically.

Must never import streamlit (directly or transitively).
"""

import os

from dotenv import load_dotenv

# Load ANTHROPIC_API_KEY etc. from a .env file at the repo root, if present.
# Never hardcode keys here.
load_dotenv()

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_OPTIONS = {
    "8-Week Moving Average": os.path.join(HERE, "models/regression.py"),
    "Holt's (double) exponential smoothing": os.path.join(HERE, "models/exponential_smoothing.py"),
    "Holt-Winters (triple) exponential smoothing": os.path.join(HERE, "models/holt_winters.py"),
    "XGBoost": os.path.join(HERE, "models/xgboost.py"),
}
# dashboard.py also supports an optional DEMAND_PIPELINE env override and
# filters out non-existent paths (dashboard.py:90-101) — mirror the existence
# filter so a missing optional model degrades the same way in both places.
MODEL_OPTIONS = {k: v for k, v in MODEL_OPTIONS.items() if os.path.exists(v)}

# MUST match dashboard.py's ALL_CUSTOMERS_VIEW exactly (dashboard.py:122) —
# every node and parity test keys the combined view off this string, and a
# mismatch silently compares/filters the wrong view.
ALL_CUSTOMERS_VIEW = "All customers (combined)"

# MUST match dashboard.py's REGION_ALL_PREFIX exactly — per-region rollup views
# are "All Customers - <region label>" (e.g. "All Customers - AU (ACR)"):
# every customer group in one region combined into a single forecast view.
REGION_ALL_PREFIX = "All Customers - "


def region_from_view(view):
    """Region label if ``view`` is a per-region rollup, else None.

    Mirrors dashboard.region_from_view — both sides must parse the view
    string identically or the agent filters a different frame than the
    dashboard forecasts (the parity tests would catch the drift).
    """
    if isinstance(view, str) and view.startswith(REGION_ALL_PREFIX):
        return view[len(REGION_ALL_PREFIX):]
    return None


def region_all_view(region):
    """The synthetic per-region combined view string for ``region``.

    Streamlit-free mirror of dashboard.region_all_view, so the headless batch
    runner can enumerate the same per-region rollup views the dashboard offers
    without importing streamlit. Inverse of ``region_from_view``.
    """
    return f"{REGION_ALL_PREFIX}{region}"

# The comparison score is a pooled MASE: the winning model's pooled backtest
# MAE divided by the pooled MAE of a plain 8-week moving average of each SKU's
# actuals over the same points (see evaluate._generic_backtest). Scale-free
# across views, so one threshold fits the huge combined view and the tiny
# groupings alike. 1.0 = "no better than the 8-week average of actuals".
# Interim value pending recalibration -- after the MASE migration lands, run
# `python -m agent.batch --no-llm`, harvest the winning-MASE distribution from
# outputs/agent_summary_*.json, and rewrite this block with the observed
# percentiles (mirroring the old Phase 3 MAE calibration note).
MASE_CONFIDENCE_THRESHOLD = 1.0

# --- Phase 4: LLM provider selection ---------------------------------------
# "anthropic" = Claude API (needs ANTHROPIC_API_KEY in .env)
# "local"     = any OpenAI-compatible server (LiteLLM / LM Studio / vLLM),
#               e.g. the gemma4-31b endpoint on james-workstation.
# agent/llm.py re-reads LLM_PROVIDER from the env at call time, so these are
# import-time defaults, not the last word.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "local").strip().lower()
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
LOCAL_LLM_BASE_URL = os.environ.get(
    "LOCAL_LLM_BASE_URL", "http://james-workstation:4000/v1"
)
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "gemma4-31b")
# Most local servers ignore the key but the OpenAI client requires a non-empty one.
LOCAL_LLM_API_KEY = os.environ.get("LOCAL_LLM_API_KEY", "not-needed")
