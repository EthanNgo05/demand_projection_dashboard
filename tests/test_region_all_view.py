"""Per-region "All Customers" rollup views ("All Customers - <region>").

Fast checks that the synthetic region view filters exactly the region's
customer groups — in the dashboard (compute_view) and in the agent's shared
view_frame helper — without invoking the slow full-graph parity suites
(which also cover one region view when run with --runslow).
"""

import pandas as pd

from agent import config as agent_config
from agent.data_io import view_frame

import dashboard

TODAY = pd.Timestamp("2026-07-01")  # matches the fixture workbook's snapshot


# First configured model file, without touching st.session_state
# (dashboard.pipeline_path() is a Streamlit-session helper).
MODEL_PATH = next(iter(dashboard.MODEL_OPTIONS.values()))


def _first_region(df):
    P = dashboard.load_pipeline(MODEL_PATH)
    regions = sorted(
        {str(P.region_for_group(g)) for g in df["Customer Grouping"].dropna().unique()}
    )
    assert regions, "fixture has no customer groups"
    return regions[0], P


def test_prefix_helpers_round_trip():
    view = dashboard.region_all_view("AU (ACR)")
    assert view == "All Customers - AU (ACR)"
    assert dashboard.region_from_view(view) == "AU (ACR)"
    assert agent_config.region_from_view(view) == "AU (ACR)"
    # Real groupings and the global combined view must NOT parse as region views.
    assert dashboard.region_from_view("AMAZON-DC") is None
    assert dashboard.region_from_view(dashboard.ALL_CUSTOMERS_VIEW) is None


def test_view_frame_region_rollup(sample_cleaned_df):
    region, P = _first_region(sample_cleaned_df)
    view = dashboard.region_all_view(region)
    sub = view_frame(sample_cleaned_df, view, P)

    in_region = sample_cleaned_df["Customer Grouping"].map(
        lambda g: str(P.region_for_group(g))
    ) == region
    expected = sample_cleaned_df[in_region]
    assert not expected.empty, f"fixture has no rows in region {region}"
    pd.testing.assert_frame_equal(sub, expected)

    # The other view kinds are untouched.
    assert view_frame(sample_cleaned_df, agent_config.ALL_CUSTOMERS_VIEW, P) is sample_cleaned_df
    group = expected["Customer Grouping"].iloc[0]
    assert set(view_frame(sample_cleaned_df, group, P)["Customer Grouping"]) == {group}


def test_compute_view_region_rollup_matches_manual_filter(sample_cleaned_df):
    """compute_view on a region view must aggregate exactly the union of the
    region's customer groups, and carry the combined-view breakdown column."""
    region, P = _first_region(sample_cleaned_df)
    view = dashboard.region_all_view(region)

    summary, weekly, agg = dashboard.compute_view(
        sample_cleaned_df, view, TODAY, MODEL_PATH
    )

    manual = sample_cleaned_df[
        sample_cleaned_df["Customer Grouping"].map(
            lambda g: str(P.region_for_group(g))
        ) == region
    ]
    pd.testing.assert_frame_equal(
        agg.reset_index(drop=True),
        P.aggregate_to_sku_week(manual).reset_index(drop=True),
    )
    assert summary is not None and not summary.empty
    # breakdown_df=sub mirrors the ALL CUSTOMERS view -> per-region top groups.
    assert "Top Volume Customer Groups" in summary.columns
    # Every forecast row is labeled with the synthetic view string.
    assert set(summary["Customer Grouping"].unique()) == {view}
