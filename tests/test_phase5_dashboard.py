"""Phase 5: Streamlit integration via the headless AppTest harness.

Two things must hold:
  1. Clicking "Run Agent Summary" runs the graph (on a background thread so the
     UI doesn't freeze) and the result is rendered in the UI.
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

DASHBOARD = os.path.join(REPO_ROOT, "src", "dashboard.py")
HAS_RAW = bool(
    __import__("glob").glob(
        os.path.join(REPO_ROOT, "raw_inputs", "demand_projections", "*.xlsx")
    )
)
needs_data = pytest.mark.skipif(
    not HAS_RAW, reason="no raw_inputs workbook to drive the full dashboard"
)


class _FakeGraph:
    """Stands in for the compiled LangGraph graph. Its stream() writes the same
    summary JSON publish would (so the dashboard's cached-render path shows it),
    records that it was called, and yields one {node: delta} update per node the
    way LangGraph's stream(stream_mode="updates") does, so the dashboard's
    progress bar advances."""

    def __init__(self, recorder):
        self._recorder = recorder

    def stream(self, state, config=None):
        self._recorder["called_with"] = state
        self._recorder["config"] = config
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
                    "mase_by_model": {"XGBoost": 0.85},
                    "narrative": "Demand is flat.",
                    "anomalies": ["- SKU-1 spiked"],
                    "confidence_flag": False,
                    "errors": [],
                },
                f,
            )
        yield {"ingest": {}}
        yield {"run_all_models": {}}
        yield {"evaluate_models": {}}
        yield {"select_best_model": {"best_model": "XGBoost", "confidence_flag": False}}
        yield {"flag_anomalies": {"anomalies": ["- SKU-1 spiked"]}}
        yield {"summarize": {"narrative": "Demand is flat."}}
        yield {"publish": {"window_excluded_skus": []}}


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
        # 300s: on a cold cache the real workbook's load + exclusion checks
        # alone take ~60s and the first ALL CUSTOMERS compute (incl. the
        # per-customer breakdown) another ~60s+ (outgrew the old 60s budget,
        # 2026-07-14). The timeout is a cap, not a wait — a passing run
        # returns as soon as the script finishes.
        at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
        assert not at.exception
        # The click starts the pipeline on a background thread (non-blocking)
        # and reruns; wait for that thread before asserting on the summary.
        at.button(key="run_agent_summary").click().run()
        assert "agent_job_thread" in at.session_state, (
            "background agent thread was not started"
        )
        thread = at.session_state["agent_job_thread"]
        thread.join(timeout=30)
        assert not thread.is_alive(), "agent thread did not finish in time"
        # Re-run so the finished job is finalized and its summary rendered.
        at.run()
        assert not at.exception

        assert recorder.get("called_with"), "graph.stream was never called"
        assert recorder["called_with"]["view"] == dashboard.ALL_CUSTOMERS_VIEW
        # The rendered summary reports the best model from the written JSON.
        assert "XGBoost" in " ".join(m.value for m in at.success)
    finally:
        if os.path.exists(default_path):
            os.remove(default_path)


@needs_data
def test_smoothing_params_survive_model_round_trip():
    """Selecting Holt autofits tuned α/β/φ; switching model away and back must
    re-establish them, not silently fall back to the file defaults.

    Regression test for the autofit_tried / autofit_params desync: switching
    model dropped ``autofit_params`` but left the ``autofit_tried`` marker set,
    so returning to the smoothing model saw "already tried", skipped the
    backtest, and computed the forecast with file-default α/β/φ instead of the
    tuned ones — changing the displayed forecast for an unchanged view/snapshot.
    """
    import dashboard

    HOLT = "Holt's (double) exponential smoothing"
    OTHER = "8-Week Moving Average"
    assert HOLT in dashboard.MODEL_OPTIONS and OTHER in dashboard.MODEL_OPTIONS

    at = AppTest.from_file(DASHBOARD, default_timeout=120).run()
    assert not at.exception

    # Select Holt -> autofit runs and stores tuned params for this view.
    # The model widget is now a top-of-page dropdown (selectbox), not a radio.
    at.selectbox(key="model_choice").set_value(HOLT).run()
    assert not at.exception
    assert "autofit_params" in at.session_state, (
        "autofit did not run on first Holt selection"
    )
    assert at.session_state["autofit_params"]["model"] == dashboard.MODEL_OPTIONS[HOLT]

    # Leave to another model, then come back to Holt.
    at.selectbox(key="model_choice").set_value(OTHER).run()
    assert not at.exception
    at.selectbox(key="model_choice").set_value(HOLT).run()
    assert not at.exception

    # Returning to Holt must re-establish autofit params for this view; otherwise
    # the forecast is silently computed with file defaults -> a different number.
    assert "autofit_params" in at.session_state, (
        "autofit params lost after a model round-trip: the forecast falls back "
        "to file-default alpha/beta/phi and changes for an unchanged view"
    )
    assert at.session_state["autofit_params"]["model"] == dashboard.MODEL_OPTIONS[HOLT]


@needs_data
def test_exceptions_view_renders():
    """Selecting the Exceptions scope renders its own (model-agnostic) view
    without error — the routing branch and render_exceptions wiring both work,
    including the All Exceptions / Key SKUs tabs."""
    import dashboard

    at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
    assert not at.exception
    # "scope" is now a top-of-page segmented control, which AppTest has no direct
    # accessor for; drive it through session state (the key is unchanged).
    at.session_state["scope"] = dashboard.EXCEPTIONS_VIEW
    at.run()
    assert not at.exception
    # It draws its own subheader and both tabs.
    assert any("Exceptions" == s.value for s in at.subheader)
    tab_labels = {t.label for t in at.tabs}
    assert {"All Exceptions", "Key SKUs"} <= tab_labels
    # The severity-threshold inputs live in the All Exceptions tab.
    assert {ni.label for ni in at.number_input} >= {"Min % deviation", "Min revenue risk / wk"}


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

    at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
    assert not at.exception
    at.radio(key="agent_llm_provider").set_value("Local LLM").run()
    assert called["llm"] is False
    assert not at.exception


def test_agent_scores_prefers_mase_and_falls_back():
    """_agent_scores reads the current mase_by_model key, falls back to the
    legacy mae_by_model for stale pre-MASE JSONs, and degrades to empty."""
    import dashboard

    assert dashboard._agent_scores({"mase_by_model": {"A": 0.9}}) == ({"A": 0.9}, True)
    assert dashboard._agent_scores({"mae_by_model": {"A": 22.1}}) == ({"A": 22.1}, False)
    # Both present -> mase wins (a regenerated file never carries both, but
    # prefer-current is the documented contract).
    assert dashboard._agent_scores(
        {"mase_by_model": {"A": 0.9}, "mae_by_model": {"A": 22.1}}
    ) == ({"A": 0.9}, True)
    assert dashboard._agent_scores({}) == ({}, False)
