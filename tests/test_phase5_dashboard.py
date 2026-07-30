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

    ``autofit_params`` is now a per-(model, view, snapshot) map — its keys are
    ``(pipeline_path, view, today_str)`` tuples — so "has Holt params" means some
    key carries the Holt pipeline path.
    """
    import dashboard

    HOLT = "Holt's (double) exponential smoothing"
    OTHER = "8-Week Moving Average"
    assert HOLT in dashboard.MODEL_OPTIONS and OTHER in dashboard.MODEL_OPTIONS

    def _has_holt_params(state):
        params = state["autofit_params"] if "autofit_params" in state else {}
        return any(key[0] == dashboard.MODEL_OPTIONS[HOLT] for key in params)

    at = AppTest.from_file(DASHBOARD, default_timeout=120).run()
    assert not at.exception

    # Select Holt -> autofit runs and stores tuned params for this view.
    # The model widget is now a top-of-page dropdown (selectbox), not a radio.
    at.selectbox(key="model_choice").set_value(HOLT).run()
    assert not at.exception
    assert _has_holt_params(at.session_state), (
        "autofit did not run on first Holt selection"
    )

    # Leave to another model, then come back to Holt.
    at.selectbox(key="model_choice").set_value(OTHER).run()
    assert not at.exception
    at.selectbox(key="model_choice").set_value(HOLT).run()
    assert not at.exception

    # Returning to Holt must re-establish autofit params for this view; otherwise
    # the forecast is silently computed with file defaults -> a different number.
    assert _has_holt_params(at.session_state), (
        "autofit params lost after a model round-trip: the forecast falls back "
        "to file-default alpha/beta/phi and changes for an unchanged view"
    )


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


def _headings(at):
    """The view body's "### ..." section headings, as one searchable string."""
    return " ".join(m.value for m in at.markdown if m.value.startswith("###"))


@needs_data
def test_quick_default_view_is_all_customers_in_the_new_order():
    """The default Quick Projections landing view, and its section order.

    Region defaults to the ALL_REGIONS sentinel and Customer group to the raw
    ALL_CUSTOMERS_VIEW string (raw, not the prettified label — several callers,
    including the agent-run button, read the selected value as a view ID). The
    body then reads KPIs -> total demand -> Customer detail -> the by-SKU-and-
    customer table, with the view-level per-SKU table in a collapsed expander.
    """
    import dashboard

    at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
    assert not at.exception
    assert at.session_state["quick_region"] == dashboard.ALL_REGIONS
    assert at.selectbox(key="quick_group").value == dashboard.ALL_CUSTOMERS_VIEW

    headings = _headings(at)
    assert "### Customer detail" in headings
    assert "### Summary table by SKU and customer" in headings
    # Customer detail comes before the table.
    assert headings.index("### Customer detail") < \
        headings.index("### Summary table by SKU and customer")
    # The old dropdown-driven SKU-detail section is gone (it lives in the table's
    # row detail cards now).
    assert "### SKU detail" not in headings
    assert any("Summary table by SKU (view total)" in (e.label or "")
               for e in at.expander)


@needs_data
def test_quick_region_change_resets_the_customer_group():
    """Switching Region must re-option the Customer-group selectbox cleanly.

    Both are keyed, which keeps the options list out of the widget's element
    identity — so the stored group survives the switch and Streamlit is the one
    that has to notice it is no longer offered and fall back to the first option
    (that region's rollup), writing the corrected value back in the same run. If a
    Streamlit upgrade changes that behaviour, this fails loudly rather than the
    page throwing "value not in options" for a planner.

    Only the two selectboxes matter here — they render into region_slot before the
    body runs, so a region whose rollup has no recent demand (and therefore stops
    the body with a "nothing to forecast" error) is still a valid probe.
    """
    import dashboard

    at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
    assert not at.exception
    regions = [r for r in at.selectbox(key="quick_region").options
               if r != dashboard.ALL_REGIONS]
    # A region with at least one individual group, so there is something to hold
    # across the switch.
    for region in regions:
        at.selectbox(key="quick_region").set_value(region).run()
        assert not at.exception
        # .options are FORMATTED labels; .value is the raw view string.
        assert at.selectbox(key="quick_group").options[0] == \
            dashboard.quick_group_label(dashboard.region_all_view(region))
        if len(at.selectbox(key="quick_group").options) > 1:
            break
    else:
        pytest.skip("no region has an individual customer group")

    # Hold a specific group (index >= 1, where the label IS the raw group name).
    at.selectbox(key="quick_group").select_index(1).run()
    assert not at.exception
    held = at.selectbox(key="quick_group").value
    assert dashboard.region_from_view(held) is None, "expected a bare group"

    # Switch to a DIFFERENT region: the held group is no longer offered, so
    # Streamlit must fall back to the new region's rollup.
    other = next(r for r in regions if r != region)
    at.selectbox(key="quick_region").set_value(other).run()
    assert not at.exception
    assert at.selectbox(key="quick_group").value == dashboard.region_all_view(other)


@needs_data
def test_quick_single_group_renders_the_by_customer_table():
    """A single customer group still gets the main by-SKU-and-customer table.

    It used to be gated to the combined / region-rollup views, so this branch had
    no coverage at all. It now runs through single_group_frames (the fast path that
    reuses compute_view's frames). Customer detail and the view-total expander are
    deliberately absent: with one group both would just restate the totals above.

    Groups are probed rather than assumed — a group with no demand in the model's
    window legitimately stops the body with a "nothing to forecast" error, and
    which groups those are depends on the snapshot.
    """
    import dashboard

    at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
    assert not at.exception
    for region in [r for r in at.selectbox(key="quick_region").options
                   if r != dashboard.ALL_REGIONS]:
        at.selectbox(key="quick_region").set_value(region).run()
        assert not at.exception
        for i in range(1, len(at.selectbox(key="quick_group").options)):
            at.selectbox(key="quick_group").select_index(i).run()
            assert not at.exception
            if not at.error:
                break
        else:
            continue
        break
    else:
        pytest.skip("no individual customer group has demand in this snapshot")

    headings = _headings(at)
    assert "### Summary table by SKU and customer" in headings
    assert "### Customer detail" not in headings
    assert not any("Summary table by SKU (view total)" in (e.label or "")
                   for e in at.expander)


def test_historical_window_label_covers_every_models_avg_column():
    """The window prefix on the Historical Demand metrics, per model.

    The figure's window follows the selected model, so the label has to be driven
    off that model's own average-column name. Read the labels out of the real model
    files rather than hardcoding them, so a model whose LOOKBACK_WEEKS changes (or a
    new model file) shows up here instead of silently rendering a bare label.
    """
    from dashboard_app.config import MODEL_OPTIONS
    from dashboard_app.summaries import historical_window_label
    from agent.model_loader import load_pipeline

    # regression reports the space-spelled column and defines no AVG_COL_LABEL.
    assert historical_window_label("8 Week POS/Orders Average") == "8-Week"
    # The hyphenated central columns (compute.py's constants).
    assert historical_window_label("8-Week POS/Orders Average") == "8-Week"
    assert historical_window_label("All-History POS/Orders Average") == "All-Time"

    seen = set()
    for label, path in MODEL_OPTIONS.items():
        P = load_pipeline(path)
        avg_col = getattr(P, "AVG_COL_LABEL", "8 Week POS/Orders Average")
        window = historical_window_label(avg_col)
        assert window in {"8-Week", "All-Time"} or window.endswith("-Week"), (
            f"{label}: {avg_col!r} produced an unusable window label {window!r}"
        )
        # Never leak the raw column name or the internal "All-History" wording
        # into a metric label.
        assert "POS/Orders" not in window
        assert window != "All-History"
        seen.add(window)
    assert seen >= {"8-Week", "All-Time"}, (
        f"expected both windows across the model catalog, saw {seen}"
    )


@needs_data
def test_quick_kpi_row_names_the_historical_demand_window():
    """The KPI row must say WHICH window its Historical Demand covers.

    With the default 8-Week Moving Average model this reads "8-Week Historical
    Demand"; the other four models make it "All-Time". Without the prefix the two
    are indistinguishable on screen even though they can differ substantially.
    """
    from dashboard_app.summaries import historical_window_label
    from dashboard_app.pipeline import load_pipeline, pipeline_path

    at = AppTest.from_file(DASHBOARD, default_timeout=300).run()
    assert not at.exception

    P = load_pipeline(pipeline_path())
    expected = historical_window_label(
        getattr(P, "AVG_COL_LABEL", "8 Week POS/Orders Average")
    )
    labels = [m.label for m in at.metric]
    assert f"{expected} Historical Demand (avg/wk)" in labels, (
        f"expected a {expected}-labelled historical-demand metric, got {labels}"
    )
    # And no un-windowed one left behind.
    assert "Historical Demand (avg/wk)" not in labels


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
