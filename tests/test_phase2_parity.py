"""Parity: the agent's numbers must match dashboard.compute_view exactly.

The most important test in this project (per 02-deterministic-pipeline-nodes.md):
both paths call the same underlying fit_regression, so assertions are
exact-match, no tolerance. Runs the combined view AND an individual Customer
Grouping — the combined view is the one most likely to diverge (combined
label + breakdown_df handling).
"""

import pandas as pd
import pytest

from agent.config import ALL_CUSTOMERS_VIEW, MODEL_OPTIONS
from agent.graph import build_graph

import dashboard  # noqa: E402  (imports streamlit in bare mode — no runtime needed)

# Slow: backtests every model across the two views. Skipped unless --runslow.
pytestmark = pytest.mark.slow

TODAY = pd.Timestamp("2026-07-01")  # pinned so both paths see identical anchors

assert ALL_CUSTOMERS_VIEW == dashboard.ALL_CUSTOMERS_VIEW


def _agent_results(view, sample_raw_path):
    final_state = build_graph().invoke(
        {
            "view": view,
            "today_ts": TODAY,
            "raw_path": sample_raw_path,
            "price_path": None,  # price parity is exercised from Phase 3 on
        }
    )
    # Phase 4 wires the LLM reasoning nodes into the same graph; with no API
    # key (the default test env) they degrade into state["errors"] by design.
    # This parity test validates the deterministic pipeline, so ignore those.
    pipeline_errors = [e for e in final_state.get("errors", []) if "LLM call failed" not in e]
    assert not pipeline_errors, pipeline_errors
    return final_state["results"]


@pytest.mark.parametrize("view", [ALL_CUSTOMERS_VIEW, "AMAZON-DC"])
def test_parity_all_models(view, sample_raw_path, sample_cleaned_df):
    results = _agent_results(view, sample_raw_path)
    for label, path in MODEL_OPTIONS.items():
        dash_summary, dash_weekly, _ = dashboard.compute_view(
            sample_cleaned_df, view, TODAY, path
        )
        agent_summary = results[label]["summary_df"]
        agent_weekly = results[label]["weekly_df"]
        pd.testing.assert_frame_equal(
            dash_summary.reset_index(drop=True),
            agent_summary.reset_index(drop=True),
            check_dtype=False,
        )
        pd.testing.assert_frame_equal(
            dash_weekly.reset_index(drop=True),
            agent_weekly.reset_index(drop=True),
            check_dtype=False,
        )


def test_combined_view_carries_breakdown(sample_raw_path):
    # proves breakdown_df parity: without it the summary lacks this column
    results = _agent_results(ALL_CUSTOMERS_VIEW, sample_raw_path)
    for label, r in results.items():
        assert "Top Volume Customer Groups" in r["summary_df"].columns, label
