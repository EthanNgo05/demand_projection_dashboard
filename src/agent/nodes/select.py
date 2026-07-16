"""Select node: pick the lowest-MASE model and set the confidence flag.

Pure state-in / state-out logic (unit-testable with hand-built dicts) plus a
logs.txt entry so the agent's decision sits in the same audit trail as the
manual Autofit runs already there.
"""

from agent import config
from agent.logging_util import logger
from agent.state import AgentState


def select_best_model(state: AgentState) -> dict:
    scored = {
        k: v["mase"] for k, v in state["results"].items() if v.get("mase") is not None
    }
    if not scored:
        logger.warning(
            "Model eval [%s]: no model produced a backtest MASE — flagging low confidence",
            state.get("view", "?"),
        )
        return {"best_model": None, "confidence_flag": True}

    best = min(scored, key=scored.get)
    threshold = state.get("mase_confidence_threshold")
    if threshold is None:
        threshold = config.MASE_CONFIDENCE_THRESHOLD
    flag = threshold is not None and scored[best] > threshold

    table = " | ".join(f"{k}: MASE {v:.2f}" for k, v in sorted(scored.items()))
    logger.info(
        "Model eval [%s]: %s -> selected %s (threshold %s, %s)",
        state.get("view", "?"),
        table,
        best,
        threshold,
        "LOW CONFIDENCE" if flag else "confidence ok",
    )
    return {"best_model": best, "confidence_flag": flag}


def route_after_select(state: AgentState) -> str:
    return "flag_low_confidence" if state["confidence_flag"] else "summarize"
