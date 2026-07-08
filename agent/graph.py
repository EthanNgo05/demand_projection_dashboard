"""Builds and compiles the agent StateGraph.

Phase 2: linear ingest -> run_all_models. Phase 3 adds evaluate/select and
conditional routing; Phases 4-5 add the LLM and publish nodes.
"""

from langgraph.graph import END, StateGraph

from agent.nodes.forecast import run_all_models
from agent.nodes.ingest import ingest
from agent.state import AgentState


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("ingest", ingest)
    g.add_node("run_all_models", run_all_models)
    g.set_entry_point("ingest")
    g.add_edge("ingest", "run_all_models")
    g.add_edge("run_all_models", END)
    return g.compile()
