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
    "Simple Regression": os.path.join(HERE, "models/regression.py"),
    "Holt's Exponential Smoothing": os.path.join(HERE, "models/exponential_smoothing.py"),
    "XGBoost": os.path.join(HERE, "models/xgboost.py"),
}
# dashboard.py also supports an optional DEMAND_PIPELINE env override and
# filters out non-existent paths (dashboard.py:90-101) — mirror the existence
# filter so a missing optional model degrades the same way in both places.
MODEL_OPTIONS = {k: v for k, v in MODEL_OPTIONS.items() if os.path.exists(v)}

# MUST match dashboard.py's ALL_CUSTOMERS_VIEW exactly (dashboard.py:122) —
# every node and parity test keys the combined view off this string, and a
# mismatch silently compares/filters the wrong view.
ALL_CUSTOMERS_VIEW = "ALL CUSTOMERS (combined)"

# Phase 3 calibration (2026-07-08, all_demand_projections_2026-07-07.xlsx):
# ran evaluate/select across all 48 real views (ALL + 47 groupings). Winning
# backtest MAEs (shared 6-week single-holdout, raw actuals): p50=7.7, p80=39.1,
# p90=70.9, max=1208 (one huge-volume view; MAE is unit-scaled, so the largest
# views dominate the tail). Threshold set at the ~80th percentile per the
# Phase 3 plan -- flags the worst ~20% of views (incl. the combined view when
# it backtests poorly) without drowning Phase 4 in false alarms. 7 tiny views
# produced no backtestable holdout at all; they get flagged via best_model=None
# regardless of this value. Revisit once MAE is normalised per-view (e.g. MASE).
MAE_CONFIDENCE_THRESHOLD = 40

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
