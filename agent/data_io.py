"""Shared, Streamlit-free I/O extracted from dashboard.py (Phase 2).

Single source of truth for raw/price-file discovery and raw-frame cleaning,
used by both ``dashboard.py`` (which keeps its ``@st.cache_data`` wrappers
around thin calls into this module) and the agent's ingest node.

Each function takes the pipeline module ``P`` explicitly instead of relying
on the dashboard's Streamlit-session ``pipeline_path()``/``load_pipeline()``
globals. Passing ``P=None`` falls back to the first configured model, which
is safe for discovery/cleaning because ``RAW_INPUTS_FOLDER`` /
``LIST_PRICE_GLOB`` / ``CUSTOMERS_TO_IGNORE`` / ``COMBINED_GROUPING`` are
identical across the three model files (see README, "The pipeline contract").

Must never import streamlit (directly or transitively).
"""

import glob
import os
import re

import numpy as np
import pandas as pd

from agent.config import MODEL_OPTIONS
from agent.model_loader import load_pipeline

# Repo root (the folder holding dashboard.py), so relative RAW_INPUTS_FOLDER /
# LIST_PRICE_GLOB paths resolve exactly as dashboard.py's HERE does.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_pipeline():
    """Load the first configured model (any model drives discovery/cleaning)."""
    if not MODEL_OPTIONS:
        raise FileNotFoundError(
            "No forecasting pipeline found — expected "
            "models/exponential_smoothing.py, models/xgboost.py or "
            "models/regression.py next to dashboard.py."
        )
    return load_pipeline(next(iter(MODEL_OPTIONS.values())))


def _resolve_pipeline(P):
    return default_pipeline() if P is None else P


def _raw_dir(P=None):
    """Resolve the folder holding the raw + price files.

    Honours DEMAND_RAW_DIR if set; otherwise uses the pipeline's own
    RAW_INPUTS_FOLDER constant (e.g. ``raw_inputs/demand_projections``),
    resolved relative to the repo root when it is a relative path. This means
    moving the raw folder in the pipeline is picked up here automatically.
    """
    P = _resolve_pipeline(P)
    folder = os.environ.get("DEMAND_RAW_DIR")
    if folder is None:
        folder = getattr(P, "RAW_INPUTS_FOLDER", None)
        if folder is None:
            # Older pipeline without the constant: derive it from INPUT_GLOB
            # if present, otherwise use the standard default location.
            input_glob = getattr(P, "INPUT_GLOB", None)
            folder = (
                os.path.dirname(input_glob)
                if input_glob
                else "raw_inputs/demand_projections"
            )
        if not os.path.isabs(folder):
            folder = os.path.join(HERE, folder)
    return folder


def raw_glob(P=None):
    """Build the raw-file glob, tracking the pipeline's RAW_INPUTS_FOLDER."""
    return os.path.join(_raw_dir(P), "all_demand_projections_*.xlsx")


def price_glob(P=None):
    """Build the list-price glob, mirroring the pipeline's LIST_PRICE_GLOB.

    The pipeline's glob (folder included) is used as-is, resolved relative to
    the repo root when it is a relative path — so every caller scans the same
    folder the batch pipeline does, regardless of the working directory.
    """
    P = _resolve_pipeline(P)
    pattern = getattr(
        P, "LIST_PRICE_GLOB",
        os.path.join("raw_inputs/list_prices", "list_prices_*.xlsx"),
    )
    if not os.path.isabs(pattern):
        pattern = os.path.join(HERE, pattern)
    return pattern


def discover_price_file(P=None):
    """Newest list-price file in the raw folder, or None if there isn't one."""
    matches = glob.glob(price_glob(P))
    return max(matches, key=os.path.getmtime) if matches else None


def _date_from_name(name):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(name))
    return m.group(1) if m else None


def discover_raw_files(P=None):
    """Return [(date_str, path)] newest first, mirroring resolve_input_file()."""
    out = []
    for path in glob.glob(raw_glob(P)):
        d = _date_from_name(path)
        if d:
            out.append((d, path))
    return sorted(out, reverse=True)


def _clean(raw_df, P):
    """Apply the exact preprocessing from the pipeline's __main__ block.

    Mirrors the updated pipeline: 'Sum of Quantity' -> Orders, and POS /
    Orders / Projection are all carried through. Falls back gracefully if an
    older file lacks the Orders column (an all-NaN Orders column is added so
    the POS-then-Orders logic still runs without a KeyError).
    """
    rename = {"'Demand'[DisplaySKU]": "SKU", "Custnmbr": "CUSTNMBR"}
    if "Sum of Quantity" in raw_df.columns:
        rename["Sum of Quantity"] = "Orders"
    df = raw_df.rename(columns=rename)

    if "Orders" not in df.columns:
        df["Orders"] = np.nan  # legacy file without an Orders/Sum of Quantity col

    df = df[["SKU", "Description", "CUSTNMBR", "WeekDate", "POS", "Orders", "Projection"]]
    df = df[~df["CUSTNMBR"].isin(P.CUSTOMERS_TO_IGNORE)]
    df["WeekDate"] = pd.to_datetime(df["WeekDate"])
    df["Customer Grouping"] = (
        df["CUSTNMBR"].map(P.COMBINED_GROUPING).fillna(df["CUSTNMBR"])
    )
    return df


def load_raw(path, P=None):
    """Read + clean a raw demand workbook from disk.

    ``header=2`` matches the PowerBI export layout (two banner rows above the
    header) — the same read dashboard.py's ``load_raw_from_path`` does.
    """
    P = _resolve_pipeline(P)
    raw = pd.read_excel(path, header=2)
    return _clean(raw, P)
