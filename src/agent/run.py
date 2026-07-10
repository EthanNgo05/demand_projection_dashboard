"""CLI entry point for the agent pipeline.

    python -m agent.run --view "ALL CUSTOMERS (combined)"
    python -m agent.run --view "AMAZON-DC" --provider local

--provider switches the Phase 4 reasoning nodes between the Claude API
("anthropic", needs ANTHROPIC_API_KEY) and a local OpenAI-compatible server
("local", see LOCAL_LLM_* in .env.example). Defaults to LLM_PROVIDER from .env.
"""

import argparse
import os

import pandas as pd

from agent.config import ALL_CUSTOMERS_VIEW
from agent.graph import build_graph


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the demand-projection agent pipeline.")
    ap.add_argument("--view", default=ALL_CUSTOMERS_VIEW,
                    help="Customer Grouping to forecast, or the combined view (default).")
    ap.add_argument("--provider", choices=["anthropic", "local"], default=None,
                    help="LLM provider for the reasoning nodes (overrides LLM_PROVIDER).")
    args = ap.parse_args(argv)

    if args.provider:
        # agent/llm.py resolves the provider from the env at call time.
        os.environ["LLM_PROVIDER"] = args.provider

    graph = build_graph()
    final_state = graph.invoke(
        {"view": args.view, "today_ts": pd.Timestamp.today().normalize()}
    )
    for label, r in final_state.get("results", {}).items():
        mae = r.get("mae")
        print(
            label, "->", len(r["summary_df"]), "rows,",
            f"backtest MAE {mae:.2f}" if mae is not None else "backtest MAE n/a",
        )
    best = final_state.get("best_model")
    if best is not None:
        print(
            f"Selected: {best}"
            + (" [LOW CONFIDENCE]" if final_state.get("confidence_flag") else "")
        )
    if final_state.get("anomalies"):
        print("\nAnomalies:")
        for line in final_state["anomalies"]:
            print(" ", line)
    excluded = final_state.get("window_excluded_skus")
    if excluded:
        print(
            f"\nActive SKUs outside {best}'s history window "
            f"({len(excluded)}) — an all-history model would forecast these:"
        )
        for row in excluded:
            desc = row.get("Description", "")
            print(f"  {row.get('SKU', '')}" + (f" — {desc}" if desc else ""))
    if final_state.get("narrative"):
        print("\nNarrative:\n" + final_state["narrative"])
    if final_state.get("errors"):
        print("\nERRORS:", final_state["errors"])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
