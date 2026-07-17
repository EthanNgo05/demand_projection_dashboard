"""Builds and compiles the agent StateGraph.

ingest -> run_all_models -> evaluate_models -> select_best_model, then the
conditional edge routes to either flag_low_confidence or the confident path
flag_anomalies -> summarize. Phase 5 adds a terminal `publish` node that both
paths fold into: it persists the run to outputs/ + logs.txt for the dashboard.
The LLM nodes are provider-agnostic (Claude API or a local OpenAI-compatible
server) via agent/llm.py.
"""

from langgraph.graph import END, StateGraph

from agent.nodes.evaluate import evaluate_models
from agent.nodes.forecast import run_all_models
from agent.nodes.ingest import ingest
from agent.nodes.publish import publish
from agent.nodes.reasoning import flag_anomalies, flag_low_confidence, summarize
from agent.nodes.select import route_after_select, select_best_model
from agent.state import AgentState


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("ingest", ingest)
    g.add_node("run_all_models", run_all_models)
    g.add_node("evaluate_models", evaluate_models)
    g.add_node("select_best_model", select_best_model)
    g.add_node("flag_anomalies", flag_anomalies)
    g.add_node("summarize", summarize)
    g.add_node("flag_low_confidence", flag_low_confidence)
    g.add_node("publish", publish)
    g.set_entry_point("ingest")
    g.add_edge("ingest", "run_all_models")
    g.add_edge("run_all_models", "evaluate_models")
    g.add_edge("evaluate_models", "select_best_model")
    # The confident path flags anomalies first, then folds them into the summary.
    g.add_conditional_edges(
        "select_best_model",
        route_after_select,
        {"flag_low_confidence": "flag_low_confidence", "summarize": "flag_anomalies"},
    )
    g.add_edge("flag_anomalies", "summarize")
    # Both terminal paths publish before ending, so every run writes its output.
    g.add_edge("summarize", "publish")
    g.add_edge("flag_low_confidence", "publish")
    g.add_edge("publish", END)
    return g.compile()
