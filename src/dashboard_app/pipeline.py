"""Load the forecasting pipeline module by path and introspect its signature."""
import os
import inspect
import importlib.util

import streamlit as st

from dashboard_app.config import MODEL_OPTIONS, DEFAULT_MODEL


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
