"""Dynamic import of a forecasting pipeline module by file path.

Mirrors ``dashboard.load_pipeline`` (dashboard.py) minus the Streamlit
``st.cache_resource`` layer — the agent re-imports on demand, which is cheap
at agent scale (a handful of loads per run) and always fresh.

Must never import streamlit (directly or transitively).
"""

import importlib.util
import os


def load_pipeline(path):
    """Import the forecasting pipeline module at ``path`` and return it."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pipeline not found at {path}")
    spec = importlib.util.spec_from_file_location("pipeline_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
