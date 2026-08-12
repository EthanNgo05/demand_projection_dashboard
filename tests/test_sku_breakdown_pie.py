"""Figure-level assertions for the SKU-detail customer-group donut.

This chart REPLACED a table, so the test that matters is that nothing the table
showed was lost on the way: every column it carried has to survive as a hover row.
The rest pins the folding rule and the two defects a donut is prone to (clipped
outside labels, colours that follow rank instead of identity). No Streamlit and no
fixture workbook — the figure is inspected as a Plotly object.
"""
import numpy as np
import pandas as pd
import pytest

from dashboard_app import charts
from dashboard_app.config import (
    C_OTHER, EIGHT_WK_AVG_COL, MIXED_SOURCE, RISK_COL,
)

UPD = "Updated Projection Average"


def _breakdown(n=3, updated=None, source="POS"):
    """A by-customer frame for one SKU, largest group first."""
    updated = updated if updated is not None else [100.0 * (n - i) for i in range(n)]
    groups = [f"GROUP-{i:02d}" for i in range(n)]
    return pd.DataFrame({
        "SKU": ["AAA"] * n,
        "Customer Grouping": groups,
        "Data Source": [source] * n,
        EIGHT_WK_AVG_COL: [u * 1.1 for u in updated],
        "Current Projection Average": [u * 0.9 for u in updated],
        UPD: updated,
        "Projection Difference": [u * 0.1 for u in updated],
        RISK_COL: [u * 2.0 for u in updated],
    })


def _trace(fig):
    return fig.data[0]


# --------------------------------------------------------------------------- #
# The hover carries what the table used to                                     #
# --------------------------------------------------------------------------- #
def test_hover_names_every_column_the_table_showed():
    """The donut replaced a table; its columns have to reappear as hover rows.

    Named EXACTLY as the columns are named elsewhere on the page — one name per
    quantity is the rule the whole app follows.
    """
    fig = charts.customer_share_donut(_breakdown())
    template = _trace(fig).hovertemplate
    for col in ["Current Projection Average", "Projection Difference",
                EIGHT_WK_AVG_COL, RISK_COL, "Data Source", UPD]:
        assert col in template, f"{col} vanished with the table: {template!r}"
    # The share column the table carried is now Plotly's own slice percentage.
    assert "%{percent}" in template
    assert "Share of Updated Forecast" not in template


def test_hover_is_not_unified_x():
    """_base_layout's hovermode is for time series; a pie has no x axis."""
    fig = charts.customer_share_donut(_breakdown())
    assert fig.layout.hovermode != "x unified"


def test_customdata_is_one_preformatted_row_per_slice():
    fig = charts.customer_share_donut(_breakdown(n=3))
    custom = _trace(fig).customdata
    assert len(custom) == 3
    # Money goes through the app's fmt_dollar, differences are signed, and every
    # cell is a string — Plotly format specs never see these values.
    assert all(isinstance(cell, str) for row in custom for cell in row)
    assert custom[0][0] == "270"           # Current Projection Average, 300 * 0.9
    assert custom[0][1] == "+30"           # Projection Difference, signed
    assert custom[0][-1] == "POS"          # Data Source


def test_missing_figures_read_as_an_em_dash_not_nan():
    """A hover row showing "nan" looks like a bug; a 0 would be a lie."""
    bd = _breakdown(n=2)
    bd.loc[0, RISK_COL] = np.nan
    bd.loc[0, "Data Source"] = None
    custom = _trace(charts.customer_share_donut(bd)).customdata
    assert "—" in custom[0]
    assert not any("nan" in cell.lower() for row in custom for cell in row)


def test_a_frame_without_prices_simply_drops_those_rows():
    """Revenue Risk is absent until a price file is loaded; the donut still draws."""
    bd = _breakdown(n=2).drop(columns=[RISK_COL])
    fig = charts.customer_share_donut(bd)
    assert RISK_COL not in _trace(fig).hovertemplate
    assert len(_trace(fig).customdata[0]) == 4


# --------------------------------------------------------------------------- #
# Folding: the palette is eight slots and is never cycled                      #
# --------------------------------------------------------------------------- #
def test_a_long_tail_folds_into_one_grey_other_slice():
    bd = _breakdown(n=11)
    fig = charts.customer_share_donut(bd)
    trace = _trace(fig)
    assert len(trace.labels) == 9, "eight groups plus one tail bucket"
    assert trace.labels[-1] == "Other (3 groups)"
    assert trace.marker.colors[-1] == C_OTHER, "the tail never takes a real slot"
    # The bucket totals what it swallowed, so the slices still sum to the SKU.
    assert trace.values[-1] == pytest.approx(bd[UPD].nsmallest(3).sum())
    assert sum(trace.values) == pytest.approx(bd[UPD].sum())


def test_the_folded_bucket_totals_the_groups_it_hides():
    bd = _breakdown(n=11)
    tail = bd.nsmallest(3, UPD)
    custom = _trace(charts.customer_share_donut(bd)).customdata[-1]
    assert custom[0] == f"{tail['Current Projection Average'].sum():,.0f}"
    assert custom[3] == charts.fmt_dollar(tail[RISK_COL].sum(), signed=True)


def test_a_tail_of_one_group_keeps_its_own_name():
    """Folding a single group into "Other" would hide an identity for nothing."""
    trace = _trace(charts.customer_share_donut(_breakdown(n=9)))
    assert len(trace.labels) == 9
    assert trace.labels[-1] == "GROUP-08"
    assert trace.marker.colors[-1] == C_OTHER, "there is still no ninth slot"


def test_the_bucket_reports_mixed_when_the_folded_groups_disagree():
    bd = _breakdown(n=11)
    bd.loc[bd.index[-1], "Data Source"] = "Orders"
    assert _trace(charts.customer_share_donut(bd)).customdata[-1][-1] == MIXED_SOURCE


# --------------------------------------------------------------------------- #
# Nothing to draw                                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bd", [
    None,
    pd.DataFrame(),
    _breakdown(n=3, updated=[0.0, 0.0, 0.0]),
    _breakdown(n=2, updated=[np.nan, np.nan]),
    _breakdown(n=2).drop(columns=[UPD]),
])
def test_nothing_to_chart_returns_none(bd):
    """The caller says so in words rather than showing an empty circle."""
    assert charts.customer_share_donut(bd) is None


def test_zero_groups_are_dropped_rather_than_drawn_as_slivers():
    fig = charts.customer_share_donut(_breakdown(n=3, updated=[50.0, 0.0, 10.0]))
    assert list(_trace(fig).labels) == ["GROUP-00", "GROUP-02"]


# --------------------------------------------------------------------------- #
# Layout defects a donut is prone to                                           #
# --------------------------------------------------------------------------- #
def test_donut_reserves_room_for_its_outside_labels():
    trace = _trace(charts.customer_share_donut(_breakdown()))
    assert trace.textposition == "outside"
    assert trace.automargin is True, (
        "outside labels sit beyond the pie's box; without automargin the topmost "
        "label is clipped off the figure"
    )


def test_the_hole_carries_the_sku_total():
    """The same number as the section's "Updated Forecast (avg/wk)" tile."""
    bd = _breakdown(n=4)
    fig = charts.customer_share_donut(bd)
    text = fig.layout.annotations[0].text
    assert f"{bd[UPD].sum():,.0f}" in text


def test_slices_stay_ordered_largest_first():
    trace = _trace(charts.customer_share_donut(_breakdown(n=4)))
    assert trace.sort is False, "the frame is already ordered; Plotly must not reorder"
    assert list(trace.values) == sorted(trace.values, reverse=True)


def test_every_drawn_group_gets_its_own_colour():
    """Eight slots, at most eight real slices, so no two may collide.

    The map is deliberately built from the DRAWN groups only. Building it from all
    of a SKU's groups would push a late-sorting group past slot 8, where
    categorical_color_map's backstop pins every remaining key to the last colour —
    two adjacent slices painted identically.
    """
    trace = _trace(charts.customer_share_donut(_breakdown(n=11)))
    real = list(trace.marker.colors)[:-1]   # excluding the grey tail bucket
    assert len(set(real)) == len(real) == 8
    assert C_OTHER not in real


def test_colour_does_not_follow_rank():
    """Reordering the frame must not repaint anything — colour keys on the name."""
    bd = _breakdown(n=4)
    forward = _trace(charts.customer_share_donut(bd))
    shuffled = _trace(charts.customer_share_donut(bd.iloc[::-1]))
    assert dict(zip(forward.labels, forward.marker.colors)) == \
        dict(zip(shuffled.labels, shuffled.marker.colors))
