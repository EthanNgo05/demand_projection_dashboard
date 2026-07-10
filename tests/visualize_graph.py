"""Debug & visualization tools for the agent graph.

Run: python debug_graph.py
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.graph import build_graph
from agent.config import ALL_CUSTOMERS_VIEW

if __name__ == "__main__":
    graph = build_graph()

    print(graph.get_graph().draw_ascii())

    state = {
        "view": ALL_CUSTOMERS_VIEW,
        "today_ts": pd.Timestamp.today(),
    }

    for step in graph.stream(state, stream_mode="updates"):
        print("=" * 80)
        print(step)