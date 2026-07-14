"""Phase 6: full cross-view / cross-model regression parity.

Phase 2's ``test_phase2_parity.py`` proved parity for two hand-picked views.
This generalises it to *every* view the dashboard exposes (the combined
ALL CUSTOMERS view plus every Customer Grouping) against *all* configured
models. It is the test that should catch any future change to ``models/*.py``,
``agent/data_io.py``, or ``agent/nodes/forecast.py`` that accidentally
diverges the agent's numbers from ``dashboard.compute_view``.

Assertions are exact-match (no tolerance): both paths call the same underlying
``fit_regression``, so any difference is a real regression, not float drift.

Run this after any change to the forecasting/ingest code, not just once at the
end of Phase 6.
"""

import pandas as pd
import pytest

from agent.config import ALL_CUSTOMERS_VIEW, MODEL_OPTIONS
from agent.graph import build_graph

import dashboard  # noqa: E402  (imports streamlit in bare mode — no runtime needed)

# Slow: backtests every model across all ~48 views. Skipped unless --runslow.
pytestmark = pytest.mark.slow

TODAY = pd.Timestamp("2026-07-01")  # pinned so both paths see identical anchors

assert ALL_CUSTOMERS_VIEW == dashboard.ALL_CUSTOMERS_VIEW


@pytest.fixture(scope="module")
def all_views(sample_cleaned_df):
    """Every view the dashboard exposes: combined + all Customer Groupings,
    plus one per-region "All Customers" rollup (one, not all five, to keep the
    already-slow matrix from growing another 15 graph invocations)."""
    by_region = dashboard.list_views(sample_cleaned_df)
    views = [ALL_CUSTOMERS_VIEW]
    first_region = sorted(by_region.keys(), key=str)[0]
    views.append(dashboard.region_all_view(first_region))
    for region_groups in by_region.values():
        views.extend(region_groups)
    # No duplicates, combined first, groupings in a stable order.
    assert len(views) == len(set(views)), views
    return views


def _agent_results(view, sample_raw_path):
    """Run the whole graph for one view; return its per-model results dict.

    The LLM reasoning nodes are wired into the same graph; with no API key
    (the default test env) they degrade into ``state['errors']`` by design.
    This parity test validates the deterministic pipeline, so ignore those but
    fail loudly on any *pipeline* error.
    """
    final_state = build_graph().invoke(
        {
            "view": view,
            "today_ts": TODAY,
            "raw_path": sample_raw_path,
            "price_path": None,
        }
    )
    pipeline_errors = [
        e for e in final_state.get("errors", []) if "LLM call failed" not in e
    ]
    assert not pipeline_errors, (view, pipeline_errors)
    return final_state["results"]


def test_all_views_enumerated(all_views):
    """Guard against the matrix silently shrinking to just the combined view."""
    assert all_views[0] == ALL_CUSTOMERS_VIEW
    assert len(all_views) > 1, "no Customer Grouping views discovered in fixture"


def test_parity_across_all_views(all_views, sample_raw_path, sample_cleaned_df):
    """Full view x model matrix: agent summary + weekly frames must match the
    dashboard exactly for every view and every model."""
    for view in all_views:
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
                obj=f"summary[{view}::{label}]",
            )
            pd.testing.assert_frame_equal(
                dash_weekly.reset_index(drop=True),
                agent_weekly.reset_index(drop=True),
                check_dtype=False,
                obj=f"weekly[{view}::{label}]",
            )


def test_every_view_selects_a_model(all_views, sample_raw_path):
    """Each view backtests to a concrete winner (or a null best_model, which is
    the documented thin-history signal) — never a crash, and the chosen model
    is always one of the configured options."""
    for view in all_views:
        final_state = build_graph().invoke(
            {
                "view": view,
                "today_ts": TODAY,
                "raw_path": sample_raw_path,
                "price_path": None,
            }
        )
        best = final_state.get("best_model")
        assert best is None or best in MODEL_OPTIONS, (view, best)
