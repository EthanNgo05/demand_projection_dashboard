"""Phase 5: Streamlit integration via the headless AppTest harness.

Two things must hold:
  1. Clicking "Run Agent Summary" actually invokes the graph and the result is
     rendered in the UI.
  2. A plain rerun (changing any other widget) does NOT invoke the LLM — the
     agent is button-triggered only. This is the load-bearing test: it enforces
     the "never on rerun" rule rather than trusting it by inspection.

The graph is faked so no real LLM/backtest runs. The dashboard imports
build_graph lazily inside the button handler (`from agent.graph import
build_graph`), so patching `agent.graph.build_graph` is what the click resolves.

These run the full dashboard against the real raw_inputs workbook, so the
initial forecast compute can take a few seconds — hence the generous timeout.
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = os.path.join(REPO_ROOT, "dashboard.py")
HAS_RAW = bool(
    __import__("glob").glob(
        os.path.join(REPO_ROOT, "raw_inputs", "demand_projections", "*.xlsx")
    )
)
needs_data = pytest.mark.skipif(
    not HAS_RAW, reason="no raw_inputs workbook to drive the full dashboard"
)


class _FakeGraph:
    """Stands in for the compiled LangGraph graph. Its invoke() writes the same
    summary JSON publish would (so the dashboard's cached-render path shows it)
    and records that it was called."""

    def __init__(self, recorder):
        self._recorder = recorder

    def invoke(self, state):
        self._recorder["called_with"] = state
        import dashboard  # the module under test

        view = state["view"]
        path = dashboard._agent_summary_path(view)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "view": view,
                    "generated_at": "2026-07-08T12:00:00",
                    "best_model": "XGBoost",
                    "mae_by_model": {"XGBoost": 22.1},
                    "narrative": "Demand is flat.",
                    "anomalies": ["- SKU-1 spiked"],
                    "confidence_flag": False,
                    "errors": [],
                },
                f,
            )
        return {"view": view, "best_model": "XGBoost", "narrative": "Demand is flat.",
                "anomalies": ["- SKU-1 spiked"], "errors": []}


@needs_data
def test_run_agent_button_triggers_graph(monkeypatch):
    recorder = {}
    monkeypatch.setattr(
        "agent.graph.build_graph", lambda: _FakeGraph(recorder)
    )

    # Clean any stale summary for the default (ALL CUSTOMERS) view first.
    import dashboard

    default_path = dashboard._agent_summary_path(dashboard.ALL_CUSTOMERS_VIEW)
    if os.path.exists(default_path):
        os.remove(default_path)

    try:
        at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
        assert not at.exception
        at.button(key="run_agent_summary").click().run()

        assert recorder.get("called_with"), "graph.invoke was never called"
        assert recorder["called_with"]["view"] == dashboard.ALL_CUSTOMERS_VIEW
        # The rendered summary reports the best model from the written JSON.
        assert "XGBoost" in " ".join(m.value for m in at.success)
    finally:
        if os.path.exists(default_path):
            os.remove(default_path)


@needs_data
def test_provider_selector_change_does_not_call_llm(monkeypatch):
    """Moving the reasoning-LLM selector (a plain rerun) must not fire the LLM.
    The agent only runs on the button click, so any non-button interaction is a
    valid probe; the provider radio is the on-point one."""
    called = {"llm": False}

    def _boom(*a, **k):
        called["llm"] = True
        raise AssertionError("LLM must not be invoked on a plain rerun")

    monkeypatch.setattr("agent.llm.get_llm", _boom)

    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not at.exception
    at.radio(key="agent_llm_provider").set_value("Local LLM").run()
    assert called["llm"] is False
    assert not at.exception
