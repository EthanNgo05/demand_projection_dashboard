"""File discovery + cached readers/loaders over agent.data_io (streamlit cache)."""
import os
import tempfile
from io import BytesIO

import pandas as pd
import streamlit as st

from agent import data_io

from dashboard_app.pipeline import load_pipeline, pipeline_path


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


def discover_key_skus_file():
    """Newest key-SKU list file (extract_key_skus.py output), or None."""
    return data_io.discover_key_skus_file()


@st.cache_data(show_spinner=False)
def load_key_skus(path, _mtime):
    """Cached read of the key-SKU list into a frozenset of SKU strings. ``_mtime``
    is part of the cache key so a freshly extracted list invalidates the cache."""
    return data_io.read_key_skus(path)


_date_from_name = data_io._date_from_name


def discover_raw_files():
    """Return [(date_str, path)] newest first, mirroring resolve_input_file()."""
    return data_io.discover_raw_files(load_pipeline(pipeline_path()))


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
MISSING_POS_COLS = data_io.MISSING_POS_COLS
_this_week_start = data_io._this_week_start
_active_in_list = data_io._active_in_list
_region_code = data_io._region_code
compute_active_products = data_io.compute_active_products
compute_inactive_projections = data_io.compute_inactive_projections
compute_discontinued_products = data_io.compute_discontinued_products
compute_discontinued_projections = data_io.compute_discontinued_projections
compute_missing_projections = data_io.compute_missing_projections
compute_missing_pos_orders = data_io.compute_missing_pos_orders


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
