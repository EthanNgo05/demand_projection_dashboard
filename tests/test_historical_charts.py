"""Figure-level assertions for the Historical Summary charts.

These pin defects that are invisible without a test: a redundant date in a hover
box, a clipped label, an axis that interpolates half-years. Nothing here needs a
browser or a database — the figures are inspected as Plotly objects.
"""
import numpy as np
import pandas as pd
import pytest

from dashboard_app import historical_charts as hc
from dashboard_app import historical_metrics as hm

LCW = pd.Timestamp("2026-07-19")


def _weeks(n, end=LCW):
    return [end - pd.Timedelta(weeks=i) for i in range(n - 1, -1, -1)]


@pytest.fixture
def frame():
    """A small priced frame with two regions and three years of weeks."""
    weeks = _weeks(160)
    rows = []
    for region, group, sku, price in [("US (LBC+NJ)", "US Retail", "AAA", 10.0),
                                      ("EU (SH-CTS)", "EU Web", "BBB", 20.0)]:
        rows.append(pd.DataFrame({
            "Customer Grouping": group, "Region": region, "SKU": sku,
            "Description": "Widget", "WeekDate": weeks,
            "demand": [10.0] * len(weeks), "revenue": [10.0 * price] * len(weeks),
            "List Price": price, "SKU Type": "Product",
        }))
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# Hover: the date belongs in the unified header, once — not on every row       #
# --------------------------------------------------------------------------- #
# hovermode="x unified" (set by charts._base_layout) already prints the x value at
# the top of the hover box. Repeating it per trace buried the actual numbers behind
# a wall of identical dates. The label:value text itself must STAY: an explicit
# hovertemplate suppresses Plotly's automatic trace-name row.

def _templates(fig):
    return [t.hovertemplate for t in fig.data
            if getattr(t, "hovertemplate", None)]


def _x_unified_figures(frame):
    """Every Historical Summary figure that uses a unified x hover."""
    return {
        "weekly_trend": hc.weekly_trend_chart(hm.weekly_totals(frame)),
        "monthly_yoy": hc.monthly_yoy_chart(hm.monthly_totals(frame),
                                            current_year=2026),
        "seasonality": hc.seasonality_chart(hm.seasonality_frame(frame)),
        "stacked_area": hc.stacked_area(hm.weekly_by_dimension(frame, "Region"),
                                        "Region"),
    }


def test_unified_hover_charts_do_not_repeat_the_x_value(frame):
    for name, fig in _x_unified_figures(frame).items():
        assert fig.layout.hovermode == "x unified", f"{name} changed hover mode"
        for template in _templates(fig):
            assert "%{x" not in template, (
                f"{name} repeats the x value on a hover row; the unified header "
                f"already shows it once. Template: {template!r}"
            )


def test_unified_hover_rows_still_name_their_series(frame):
    """Dropping %{x} must not drop the label — that would leave a bare number."""
    for name, fig in _x_unified_figures(frame).items():
        for template in _templates(fig):
            head = template.split(":")[0]
            assert head and "%{" not in head, (
                f"{name} hover row has no literal label before its value, so a "
                f"hovered row would show a swatch and a number with no identity. "
                f"Template: {template!r}"
            )


def test_stacked_area_hover_is_region_and_amount_only(frame):
    """The reported case: 'AU (ACR): $1,646', with the date only in the header."""
    fig = hc.stacked_area(hm.weekly_by_dimension(frame, "Region"), "Region")
    for template in _templates(fig):
        assert template.count("<br>") == 0, f"expected one line: {template!r}"
        assert "%{y" in template and "%{x" not in template


def test_year_stays_in_the_hover_where_it_identifies_the_series(frame):
    """In monthly-YoY and seasonality the year IS the identity, so it must remain."""
    monthly = hc.monthly_yoy_chart(hm.monthly_totals(frame), current_year=2026)
    assert any(t.startswith("2026:") for t in _templates(monthly))
    assert any(t.startswith("2025:") for t in _templates(monthly))
    seasonal = hc.seasonality_chart(hm.seasonality_frame(frame))
    assert all(t[:4].isdigit() for t in _templates(seasonal))


def test_heatmap_keeps_its_cell_identity_in_the_hover(frame):
    """A heatmap has no unified header, so %{y} %{x} is the only cell identity."""
    fig = hc.month_year_heatmap(hm.month_year_matrix(frame))
    template = fig.data[0].hovertemplate
    assert "%{y}" in template and "%{x}" in template


# --------------------------------------------------------------------------- #
# Donut label clipping                                                        #
# --------------------------------------------------------------------------- #
def test_donut_reserves_room_for_its_outside_labels(frame):
    start, end = hm.window_bounds(hm.WINDOW_52W, LCW)
    fig = hc.share_donut(hm.by_dimension(frame, "Region", start, end), "Region")
    trace = fig.data[0]
    assert trace.textposition == "outside"
    assert trace.automargin is True, (
        "outside labels sit beyond the pie's box; without automargin the topmost "
        "label is clipped off the figure"
    )
    assert fig.layout.margin.t >= 90, "title needs clearance from the top label"


# --------------------------------------------------------------------------- #
# Movers: the chart must not name a period it no longer measures                #
# --------------------------------------------------------------------------- #
def test_movers_chart_does_not_hardcode_52_weeks(frame):
    """yoy_movers follows the analysis window now, so the labels can't say '52'."""
    start, end = hm.window_bounds(hm.WINDOW_13W, LCW)
    fig = hc.movers_chart(hm.yoy_movers(frame, start, end, n=10))
    text = (fig.layout.xaxis.title.text or "") + "".join(_templates(fig))
    assert "52" not in text, f"movers still names a fixed 52 weeks: {text!r}"


# --------------------------------------------------------------------------- #
# Heatmap year axis                                                           #
# --------------------------------------------------------------------------- #
def test_heatmap_year_axis_is_categorical(frame):
    """Numeric-looking strings on a linear axis produced 2022.5 / 2023.5 ticks."""
    fig = hc.month_year_heatmap(hm.month_year_matrix(frame))
    assert fig.layout.xaxis.type == "category"


def test_heatmap_x_values_are_whole_years(frame):
    fig = hc.month_year_heatmap(hm.month_year_matrix(frame))
    for value in fig.data[0].x:
        assert "." not in str(value), f"non-integer year label: {value!r}"
        assert str(value).isdigit() and len(str(value)) == 4


def test_heatmap_cell_labels_are_positioned_by_index(frame):
    """The regression the categorical axis introduced, and the reason it was missed.

    On a category axis an annotation coordinate is a category INDEX, and Plotly
    resolves it numeric-first: x="2023" is read as the coordinate 2023, roughly 2000
    index units off the grid. That stretched the x autorange to 2026, squashed the
    real columns into a sliver and stacked every label at the right edge. Asserting
    the trace's x values (above) could not see it — the defect lived in the
    annotations, which nothing looked at.
    """
    matrix = hm.month_year_matrix(frame)
    fig = hc.month_year_heatmap(matrix)
    n_years, n_months = len(matrix.columns), 12
    assert fig.layout.annotations, "every populated cell should carry its value"
    for a in fig.layout.annotations:
        assert isinstance(a.x, (int, np.integer)) and not isinstance(a.x, bool), (
            f"annotation x must be a category index, got {a.x!r} "
            f"({type(a.x).__name__})"
        )
        assert isinstance(a.y, (int, np.integer)) and not isinstance(a.y, bool), (
            f"annotation y must be a category index, got {a.y!r}"
        )
        assert 0 <= a.x < n_years, f"x={a.x} is outside {n_years} year columns"
        assert 0 <= a.y < n_months, f"y={a.y} is outside 12 month rows"


def test_heatmap_labels_cannot_stretch_the_axis_range(frame):
    """No annotation may sit beyond the grid, whatever coordinate convention wins.

    The autorange pass includes annotation positions, so one stray label is enough
    to compress every real column into an unreadable strip.
    """
    matrix = hm.month_year_matrix(frame)
    fig = hc.month_year_heatmap(matrix)
    assert max(a.x for a in fig.layout.annotations) < len(matrix.columns)


def test_heatmap_leaves_room_for_its_year_labels(frame):
    fig = hc.month_year_heatmap(hm.month_year_matrix(frame))
    assert fig.layout.xaxis.automargin is True
    assert fig.layout.margin.b >= 30, "year ticks were clipped at b=10"


# --------------------------------------------------------------------------- #
# Range presets                                                               #
# --------------------------------------------------------------------------- #
def test_range_presets_reach_beyond_three_years():
    """History is pinned rather than rolling, so it now exceeds 3 years."""
    import inspect
    source = inspect.getsource(hc.history_range_control)
    assert '"4 Years"' in source
    assert '"All"' in source, "All must remain — it is what tracks the true floor"


# --------------------------------------------------------------------------- #
# Empty-input tolerance (a filter can legitimately select nothing)            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("build", [
    lambda f: hc.weekly_trend_chart(hm.weekly_totals(f)),
    lambda f: hc.monthly_yoy_chart(hm.monthly_totals(f)),
    lambda f: hc.seasonality_chart(hm.seasonality_frame(f)),
    lambda f: hc.stacked_area(hm.weekly_by_dimension(f, "Region"), "Region"),
    lambda f: hc.month_year_heatmap(hm.month_year_matrix(f)),
    lambda f: hc.share_donut(hm.by_dimension(f, "Region", LCW, LCW), "Region"),
])
def test_charts_tolerate_an_empty_frame(build):
    empty = pd.DataFrame(columns=["WeekDate", "Region", "SKU", "demand", "revenue",
                                  "Description", "Customer Grouping"])
    fig = build(empty)
    assert fig is not None
    assert not np.any([getattr(t, "x", None) is not None and len(t.x or []) > 1
                       for t in fig.data]), "expected an empty/placeholder figure"
