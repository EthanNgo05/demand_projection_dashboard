"""Phase 5 publish node: persist the agent's output for the dashboard.

The terminal node of the graph. It writes a per-view JSON summary to
``outputs/agent_summary_{view}.json`` and appends one ``AGENT`` line to
``logs.txt`` (append, never overwrite — matching the existing Autofit lines).
The dashboard reads the JSON back to show the last run without re-invoking the
(slow, LLM-backed) graph on every Streamlit rerun.

Path note: this file lives at ``agent/nodes/publish.py``, so the repo root is
THREE levels up. ``outputs/`` and ``logs.txt`` both live at the repo root
alongside dashboard.py — not under ``agent/``. Tests monkeypatch OUTPUT_DIR,
and the log path is derived from it at call time so the two stay consistent.
"""

import json
import os
from datetime import datetime

from agent.state import AgentState

# agent/nodes/publish.py -> agent/nodes -> agent -> <repo root>
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs")


def publish(state: AgentState) -> dict:
    view = state["view"]
    best = state.get("best_model")
    results = state.get("results", {})
    payload = {
        "view": view,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "best_model": best,
        "mae_by_model": {k: v.get("mae") for k, v in results.items()},
        "narrative": state.get("narrative"),
        "anomalies": state.get("anomalies", []),
        # Persist these too — the dashboard (and anyone reading the JSON later)
        # needs to know whether the run was low-confidence or partially failed.
        "confidence_flag": state.get("confidence_flag", False),
        "errors": state.get("errors", []),
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_view = view.replace(" ", "_").replace("/", "-")
    path = os.path.join(OUTPUT_DIR, f"agent_summary_{safe_view}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    # logs.txt sits one level up from outputs/ (i.e. the repo root). Deriving it
    # from OUTPUT_DIR keeps the monkeypatched test layout consistent.
    log_path = os.path.join(os.path.dirname(OUTPUT_DIR), "logs.txt")
    with open(log_path, "a") as f:
        f.write(
            f"{datetime.now():%Y-%m-%d %H:%M:%S}  AGENT     [{view}] "
            f"best={best} mae={payload['mae_by_model'].get(best)} -> {path}\n"
        )

    return {}
