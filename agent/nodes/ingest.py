"""Ingest node: discover the newest raw + price files, load and clean them.

Calls through the ``data_io`` module object (not ``from ... import name``) so
tests can monkeypatch ``agent.data_io.discover_raw_files`` / ``discover_price_file``.
"""

import pandas as pd

from agent import data_io
from agent.config import MODEL_OPTIONS
from agent.logging_util import logger
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

    # Apply the exact same pre-forecast exclusions the dashboard does, so the
    # agent never forecasts or flags a SKU that is discontinued/inactive or in a
    # region it is not 'Active in'. The Plytix export doubles as the list-price
    # file, so it is read from the same price_path. With no Plytix file the
    # checks degrade to no-ops (bar the trailing-'*' drop), preserving parity.
    plytix_df = data_io.read_plytix(price_path) if price_path else None
    today_ts = state.get("today_ts")
    anchors = P.week_anchors(today_ts) if today_ts is not None else None
    excl = data_io.apply_exclusions(cleaned, plytix_df, P, anchors=anchors)
    cleaned = excl.df
    if excl.n_excluded_rows:
        logger.info(
            "Active-in check: dropped %d raw rows across %d SKU×customer×region "
            "combos not in the SKU's 'Active in' list.",
            excl.n_excluded_rows, len(excl.inactive_df),
        )
    if excl.n_disc_rows:
        logger.info(
            "Discontinued check: dropped %d raw rows across %d "
            "discontinued/inactive SKUs (trailing '*' or Plytix status).",
            excl.n_disc_rows, excl.n_disc_skus,
        )

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
