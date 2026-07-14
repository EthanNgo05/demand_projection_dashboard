"""Phase 5 publish node: persist the agent's output for the dashboard.

The terminal node of the graph. It writes a per-view JSON summary to
``outputs/agent_summary_{view}.json`` and appends one ``AGENT`` line to the
daily app log, ``logs/<date>/app.log`` (append, never overwrite — matching the
existing Autofit lines). The dashboard reads the JSON back to show the last run
without re-invoking the (slow, LLM-backed) graph on every Streamlit rerun.

Path note: this file lives at ``src/agent/nodes/publish.py``, so the repo root
is FOUR levels up. ``outputs/`` lives at the repo root (the folder holding
raw_inputs/ + logs/) — not under ``src/`` or ``agent/``. Tests monkeypatch
OUTPUT_DIR for the JSON and ``log_config.LOG_ROOT`` for the audit line.
"""

import json
import os
from datetime import datetime

from agent.config import MODEL_OPTIONS
from agent.data_io import view_frame
from agent.model_loader import load_pipeline
from agent.state import AgentState
from log_config import dated_log_path

# src/agent/nodes/publish.py -> agent/nodes -> agent -> src -> <repo root>
REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
)
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs")


def _window_excluded_skus(state: AgentState) -> list[dict]:
    """Active SKUs the winning model omits because their demand predates its
    history window.

    A SKU is forecast only if it has POS/Orders inside the model's window. The
    8-week moving average uses the last 8 completed weeks, so a SKU whose only
    sales are older than that gets no projection — while an all-history model
    (Holt/XGBoost) still forecasts it. This returns exactly that gap: SKUs with
    demand history in the view but none inside the WINNER's window, using the
    winner's own ``week_anchors``. An all-history winner therefore returns [].

    The frame is the post-exclusion ``cleaned_df`` (discontinued/inactive and
    out-of-region SKUs were already dropped at ingest), so every SKU here is an
    active, in-region product the model is silently leaving out.
    """
    best = state.get("best_model")
    df = state.get("cleaned_df")
    today_ts = state.get("today_ts")
    view = state.get("view")
    if best is None or best not in MODEL_OPTIONS or df is None or today_ts is None:
        return []
    try:
        P = load_pipeline(MODEL_OPTIONS[best])
        lookback_start, last_complete_week, _ = P.week_anchors(today_ts)
    except Exception:  # noqa: BLE001 — a note must never break the terminal node
        return []

    sub = view_frame(df, view, P)
    demand = sub[sub["POS"].notna() | sub["Orders"].notna()]
    if demand.empty:
        return []
    in_window = demand[
        (demand["WeekDate"] >= lookback_start)
        & (demand["WeekDate"] <= last_complete_week)
    ]
    excluded = set(demand["SKU"].astype(str)) - set(in_window["SKU"].astype(str))
    if not excluded:
        return []

    desc_map: dict[str, str] = {}
    if "Description" in demand.columns:
        for sku, d in zip(demand["SKU"].astype(str), demand["Description"]):
            if sku not in desc_map and d == d and d is not None:  # d==d skips NaN
                desc_map[sku] = str(d)
    return [{"SKU": s, "Description": desc_map.get(s, "")} for s in sorted(excluded)]


def publish(state: AgentState) -> dict:
    view = state["view"]
    best = state.get("best_model")
    results = state.get("results", {})
    window_excluded = _window_excluded_skus(state)
    payload = {
        "view": view,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "best_model": best,
        "mae_by_model": {k: v.get("mae") for k, v in results.items()},
        "narrative": state.get("narrative"),
        "anomalies": state.get("anomalies", []),
        # SKUs the winning model omits for having no demand inside its window
        # (see _window_excluded_skus). Empty for an all-history winner.
        "window_excluded_skus": window_excluded,
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

    # The agent's audit line joins the same daily log the dashboard writes:
    # logs/<date>/app.log at the repo root (tests redirect via log_config.LOG_ROOT).
    log_path = dated_log_path("app.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now():%Y-%m-%d %H:%M:%S}  AGENT     [{view}] "
            f"best={best} mae={payload['mae_by_model'].get(best)} -> {path}\n"
        )

    return {"window_excluded_skus": window_excluded}
