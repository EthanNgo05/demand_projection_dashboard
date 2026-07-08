"""Ingest node: discover the newest raw + price files, load and clean them.

Calls through the ``data_io`` module object (not ``from ... import name``) so
tests can monkeypatch ``agent.data_io.discover_raw_files`` / ``discover_price_file``.
"""

import pandas as pd

from agent import data_io
from agent.config import MODEL_OPTIONS
from agent.model_loader import load_pipeline
from agent.state import AgentState


def ingest(state: AgentState) -> dict:
    # Honour a pre-set raw_path (parity tests / reruns pin the input file);
    # otherwise take the newest discovered file, mirroring resolve_input_file().
    raw_path = state.get("raw_path")
    if not raw_path:
        files = data_io.discover_raw_files()
        if not files:
            return {
                "errors": state.get("errors", [])
                + ["No raw demand files found in raw_inputs/demand_projections"]
            }
        _, raw_path = files[0]  # newest first

    # Same for the price file — "price_path" explicitly present (even as None)
    # is respected, so tests can force the no-prices path.
    if "price_path" in state:
        price_path = state["price_path"]
    else:
        price_path = data_io.discover_price_file()

    # Any model file works to drive _clean's schema (CUSTOMERS_TO_IGNORE /
    # COMBINED_GROUPING are identical across the three model modules today).
    P = load_pipeline(next(iter(MODEL_OPTIONS.values())))
    raw = pd.read_excel(raw_path, header=2)  # header=2 matches dashboard load_raw_from_path
    cleaned = data_io._clean(raw, P)

    prices = (
        P.load_list_prices(price_path)
        if price_path and hasattr(P, "load_list_prices")
        else None
    )

    return {
        "raw_path": raw_path,
        "price_path": price_path,
        "cleaned_df": cleaned,
        "prices": prices,
    }
