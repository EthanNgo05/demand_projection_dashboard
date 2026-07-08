"""Phase 4 placeholder nodes: pass-throughs so Phase 3's conditional routing
compiles and both branches are reachable/testable today.

Phase 4 replaces these with the real LLM nodes (anomaly flagging + narrative
summary). Keep them side-effect-free.
"""

from agent.logging_util import logger
from agent.state import AgentState


def flag_low_confidence(state: AgentState) -> dict:
    """Placeholder for Phase 4's LLM low-confidence investigation node."""
    logger.info(
        "flag_low_confidence [%s]: best=%s — Phase 4 LLM node not built yet (pass-through)",
        state.get("view", "?"),
        state.get("best_model"),
    )
    return {}


def summarize(state: AgentState) -> dict:
    """Placeholder for Phase 4's LLM narrative summary node."""
    return {}
