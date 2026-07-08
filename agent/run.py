"""CLI entry point for the agent pipeline (no Streamlit, no LLM yet).

    python -m agent.run --view "ALL CUSTOMERS (combined)"
    python -m agent.run --view "AMAZON-DC"
"""

import argparse

import pandas as pd

from agent.config import ALL_CUSTOMERS_VIEW
from agent.graph import build_graph


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the demand-projection agent pipeline.")
    ap.add_argument("--view", default=ALL_CUSTOMERS_VIEW,
                    help="Customer Grouping to forecast, or the combined view (default).")
    args = ap.parse_args(argv)

    graph = build_graph()
    final_state = graph.invoke(
        {"view": args.view, "today_ts": pd.Timestamp.today().normalize()}
    )
    for label, r in final_state.get("results", {}).items():
        print(label, "->", len(r["summary_df"]), "rows")
    if final_state.get("errors"):
        print("ERRORS:", final_state["errors"])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
