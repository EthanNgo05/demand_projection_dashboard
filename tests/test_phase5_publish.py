"""Phase 5: the `publish` node's file/log output + graph wiring.

`publish` is the terminal node: it writes outputs/agent_summary_{view}.json and
appends one AGENT line to logs/<date>/app.log. These tests monkeypatch
OUTPUT_DIR (for the JSON) and log_config.LOG_ROOT (for the audit line) to a
tmp_path so they never touch the repo's real outputs/ or logs/.
"""

import json
import os

import numpy as np
import pandas as pd

from agent.nodes.publish import _window_excluded_skus, publish


def _demand_with_out_of_window_sku():
    """A view frame where CW1897's only demand predates the 8-week window
    (as of the 2026-07-09 anchor) while CW9999 sells inside it."""
    rows = [["CW1897", "45L can", "NETDEPOT-JP",
             pd.Timestamp("2026-04-19"), 5.0, np.nan, 5, "Others - JP"]]
    for wk in pd.date_range("2026-05-10", "2026-06-28", freq="7D"):
        rows.append(["CW9999", "other can", "NETDEPOT-JP", wk, 20.0, np.nan, 20,
                     "Others - JP"])
    return pd.DataFrame(rows, columns=[
        "SKU", "Description", "Customer", "WeekDate", "POS", "Orders",
        "Projection", "Customer Grouping"])


def test_window_excluded_skus_flags_out_of_window():
    """The 8-week moving average drops a SKU whose only sales predate its
    window; that SKU must be reported so it isn't silently omitted."""
    state = {
        "view": "Others - JP",
        "best_model": "8-Week Moving Average",
        "today_ts": pd.Timestamp("2026-07-09"),
        "cleaned_df": _demand_with_out_of_window_sku(),
    }
    excluded = _window_excluded_skus(state)
    skus = {r["SKU"] for r in excluded}
    assert skus == {"CW1897"}, skus
    assert excluded[0]["Description"] == "45L can"


def test_window_excluded_empty_for_all_history_winner():
    """An all-history winner (XGBoost/Holt) forecasts every SKU with any demand,
    so there is nothing outside its window to report."""
    state = {
        "view": "Others - JP",
        "best_model": "XGBoost",
        "today_ts": pd.Timestamp("2026-07-09"),
        "cleaned_df": _demand_with_out_of_window_sku(),
    }
    assert _window_excluded_skus(state) == []


def test_window_excluded_absent_state_is_safe():
    """No cleaned_df / today_ts (e.g. the hand-built states other tests use) must
    degrade to an empty list, never raise."""
    assert _window_excluded_skus({"view": "V", "best_model": "XGBoost"}) == []


def test_publish_writes_expected_json(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.nodes.publish.OUTPUT_DIR", str(tmp_path))
    state = {
        "view": "All customers (combined)",
        "best_model": "XGBoost",
        "results": {"XGBoost": {"mase": 0.85}, "8-Week Moving Average": {"mase": 1.10}},
        "narrative": "Demand is flat.",
        "anomalies": ["- SKU-1 spiked"],
        "confidence_flag": False,
        "errors": [],
    }
    publish(state)

    out_path = tmp_path / "agent_summary_All_customers_(combined).json"
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["best_model"] == "XGBoost"
    assert payload["mase_by_model"]["XGBoost"] == 0.85
    assert payload["mase_by_model"]["8-Week Moving Average"] == 1.10
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
             "results": {"XGBoost": {"mase": 1.0}}})
    assert (tmp_path / "agent_summary_AMAZON_US-DC.json").exists()


def test_publish_appends_to_logs_not_overwrites(tmp_path, monkeypatch):
    from log_config import dated_log_path
    monkeypatch.setattr("log_config.LOG_ROOT", str(tmp_path / "logs"))
    # Seed today's log so we can prove publish appends rather than overwrites.
    log_path = dated_log_path("app.log")  # today's folder under tmp_path
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("existing line\n")
    monkeypatch.setattr("agent.nodes.publish.OUTPUT_DIR", str(tmp_path / "outputs"))
    os.makedirs(tmp_path / "outputs")
    publish({"view": "All customers (combined)", "best_model": "XGBoost",
             "results": {"XGBoost": {"mase": 0.85}}, "narrative": "", "anomalies": []})
    with open(log_path, encoding="utf-8") as f:
        contents = f.read()
    assert "existing line" in contents  # not clobbered
    assert "AGENT" in contents          # our line appended
    assert "best=XGBoost" in contents
    assert "mase=" in contents


def test_publish_handles_low_confidence_none_best(tmp_path, monkeypatch):
    """The flag_low_confidence path can arrive with best_model=None (no
    scoreable backtest). publish must still write a valid file, not crash."""
    monkeypatch.setattr("agent.nodes.publish.OUTPUT_DIR", str(tmp_path))
    publish({"view": "TINY VIEW", "best_model": None, "results": {},
             "narrative": "History too short.", "anomalies": [],
             "confidence_flag": True, "errors": []})
    payload = json.loads((tmp_path / "agent_summary_TINY_VIEW.json").read_text())
    assert payload["best_model"] is None
    assert payload["mase_by_model"] == {}
    assert payload["confidence_flag"] is True


def test_publish_creates_output_dir_if_missing(tmp_path, monkeypatch):
    nested = tmp_path / "does" / "not" / "exist"
    monkeypatch.setattr("agent.nodes.publish.OUTPUT_DIR", str(nested))
    os.makedirs(nested.parent.parent)  # leave the last two levels for publish
    # publish should makedirs(OUTPUT_DIR) itself.
    publish({"view": "V", "best_model": "XGBoost",
             "results": {"XGBoost": {"mase": 1.0}}})
    assert (nested / "agent_summary_V.json").exists()


def test_graph_includes_publish_as_terminal_node():
    """Both terminal paths (summarize, flag_low_confidence) must route through
    publish before END, and the graph must still compile."""
    from agent.graph import build_graph

    graph = build_graph()
    nodes = set(graph.get_graph().nodes)
    assert "publish" in nodes
