"""Phase 3: selection logic + conditional routing (pure functions + compiled graph)."""

from agent.nodes.select import route_after_select, select_best_model


def test_select_best_model_picks_lowest_mase():
    state = {"results": {"A": {"mase": 1.30}, "B": {"mase": 0.85}, "C": {"mase": 2.10}}}
    out = select_best_model(state)
    assert out["best_model"] == "B"


def test_confidence_flag_boundary():
    state = {"results": {"A": {"mase": 1.0}}, "mase_confidence_threshold": 1.0}
    assert select_best_model(state)["confidence_flag"] is False  # exactly at threshold = ok
    state["results"]["A"]["mase"] = 1.01
    assert select_best_model(state)["confidence_flag"] is True


def test_route_after_select_branches_correctly():
    assert route_after_select({"confidence_flag": True}) == "flag_low_confidence"
    assert route_after_select({"confidence_flag": False}) == "summarize"


def test_select_best_model_handles_no_scored_models():
    out = select_best_model({"results": {"A": {}, "B": {}}})
    assert out["best_model"] is None and out["confidence_flag"] is True


def test_graph_routes_to_low_confidence_branch(monkeypatch):
    """Force every model's mase above threshold; the compiled graph must reach
    flag_low_confidence (and summarize on the low-MASE run). Ingest/forecast/
    evaluate are stubbed — this proves the *routing*, not the models."""
    import agent.graph as graph_mod

    visited = []

    def fake_ingest(state):
        return {"cleaned_df": None, "prices": None}

    def fake_run_all(state):
        return {"results": {}}

    def make_fake_evaluate(mase):
        return lambda state: {"results": {"Stub Model": {"mase": mase}}}

    monkeypatch.setattr(graph_mod, "ingest", fake_ingest)
    monkeypatch.setattr(graph_mod, "run_all_models", fake_run_all)
    monkeypatch.setattr(
        graph_mod, "flag_low_confidence",
        lambda state: visited.append("flag_low_confidence") or {},
    )
    monkeypatch.setattr(
        graph_mod, "flag_anomalies",
        lambda state: visited.append("flag_anomalies") or {},
    )
    monkeypatch.setattr(
        graph_mod, "summarize", lambda state: visited.append("summarize") or {}
    )

    base = {"view": "TEST", "mase_confidence_threshold": 1.0}

    monkeypatch.setattr(graph_mod, "evaluate_models", make_fake_evaluate(999.0))
    graph_mod.build_graph().invoke(dict(base))
    assert visited == ["flag_low_confidence"]

    # Phase 4: the confident branch runs flag_anomalies then summarize
    monkeypatch.setattr(graph_mod, "evaluate_models", make_fake_evaluate(0.5))
    graph_mod.build_graph().invoke(dict(base))
    assert visited == ["flag_low_confidence", "flag_anomalies", "summarize"]
