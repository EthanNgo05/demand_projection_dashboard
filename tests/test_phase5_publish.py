"""Phase 5: the `publish` node's file/log output + graph wiring.

`publish` is the terminal node: it writes outputs/agent_summary_{view}.json and
appends one AGENT line to logs.txt. These tests monkeypatch OUTPUT_DIR to a
tmp_path so they never touch the repo's real outputs/ or logs.txt. The log path
is derived from OUTPUT_DIR at call time, so patching OUTPUT_DIR relocates both.
"""

import json
import os

from agent.nodes.publish import publish


def test_publish_writes_expected_json(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.nodes.publish.OUTPUT_DIR", str(tmp_path))
    state = {
        "view": "ALL CUSTOMERS (combined)",
        "best_model": "XGBoost",
        "results": {"XGBoost": {"mae": 22.1}, "8-Week Moving Average": {"mae": 30.0}},
        "narrative": "Demand is flat.",
        "anomalies": ["- SKU-1 spiked"],
        "confidence_flag": False,
        "errors": [],
    }
    publish(state)

    out_path = tmp_path / "agent_summary_ALL_CUSTOMERS_(combined).json"
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["best_model"] == "XGBoost"
    assert payload["mae_by_model"]["XGBoost"] == 22.1
    assert payload["mae_by_model"]["8-Week Moving Average"] == 30.0
    assert payload["narrative"] == "Demand is flat."
    assert payload["anomalies"] == ["- SKU-1 spiked"]
    assert payload["confidence_flag"] is False
    # generated_at is stamped at write time.
    assert payload["generated_at"]


def test_publish_mangles_view_into_filename(tmp_path, monkeypatch):
    """Spaces -> underscores and '/' -> '-' so the view is a safe filename and
    matches dashboard._agent_summary_path's mangling exactly."""
    monkeypatch.setattr("agent.nodes.publish.OUTPUT_DIR", str(tmp_path))
    publish({"view": "AMAZON US/DC", "best_model": "XGBoost",
             "results": {"XGBoost": {"mae": 1.0}}})
    assert (tmp_path / "agent_summary_AMAZON_US-DC.json").exists()


def test_publish_appends_to_logs_not_overwrites(tmp_path, monkeypatch):
    log_path = tmp_path / "logs.txt"
    log_path.write_text("existing line\n")
    monkeypatch.setattr("agent.nodes.publish.OUTPUT_DIR", str(tmp_path / "outputs"))
    os.makedirs(tmp_path / "outputs")
    publish({"view": "ALL CUSTOMERS (combined)", "best_model": "XGBoost",
             "results": {"XGBoost": {"mae": 22.1}}, "narrative": "", "anomalies": []})
    contents = log_path.read_text()
    assert "existing line" in contents  # not clobbered
    assert "AGENT" in contents          # our line appended
    assert "best=XGBoost" in contents


def test_publish_handles_low_confidence_none_best(tmp_path, monkeypatch):
    """The flag_low_confidence path can arrive with best_model=None (no
    scoreable backtest). publish must still write a valid file, not crash."""
    monkeypatch.setattr("agent.nodes.publish.OUTPUT_DIR", str(tmp_path))
    publish({"view": "TINY VIEW", "best_model": None, "results": {},
             "narrative": "History too short.", "anomalies": [],
             "confidence_flag": True, "errors": []})
    payload = json.loads((tmp_path / "agent_summary_TINY_VIEW.json").read_text())
    assert payload["best_model"] is None
    assert payload["mae_by_model"] == {}
    assert payload["confidence_flag"] is True


def test_publish_creates_output_dir_if_missing(tmp_path, monkeypatch):
    nested = tmp_path / "does" / "not" / "exist"
    monkeypatch.setattr("agent.nodes.publish.OUTPUT_DIR", str(nested))
    os.makedirs(nested.parent.parent)  # leave the last two levels for publish
    # publish should makedirs(OUTPUT_DIR) itself.
    publish({"view": "V", "best_model": "XGBoost",
             "results": {"XGBoost": {"mae": 1.0}}})
    assert (nested / "agent_summary_V.json").exists()


def test_graph_includes_publish_as_terminal_node():
    """Both terminal paths (summarize, flag_low_confidence) must route through
    publish before END, and the graph must still compile."""
    from agent.graph import build_graph

    graph = build_graph()
    nodes = set(graph.get_graph().nodes)
    assert "publish" in nodes
