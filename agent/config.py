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

MAE_CONFIDENCE_THRESHOLD = None  # set in Phase 3 once real MAE ranges are known
ANTHROPIC_MODEL = "claude-sonnet-5"
