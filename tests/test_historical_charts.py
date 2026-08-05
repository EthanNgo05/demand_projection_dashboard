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
        "weekly_year_overlay": hc.weekly_year_overlay(hm.weekly_by_year(frame)),
        "monthly_year_overlay": hc.monthly_year_overlay(hm.monthly_totals(frame)),
        "dimension_lines": hc.dimension_lines(
            hm.weekly_by_dimension(frame, "Region"), "Region"),
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


def test_dimension_lines_hover_is_region_and_amount_only(frame):
    """The reported case: 'AU (ACR): $1,646', with the date only in the header."""
    fig = hc.dimension_lines(hm.weekly_by_dimension(frame, "Region"), "Region")
    for template in _templates(fig):
        assert template.count("<br>") == 0, f"expected one line: {template!r}"
        assert "%{y" in template and "%{x" not in template


def test_year_stays_in_the_hover_where_it_identifies_the_series(frame):
    """On both year overlays the year IS the series identity, so it must remain."""
    monthly = hc.monthly_year_overlay(hm.monthly_totals(frame))
    assert any(t.startswith("2026:") for t in _templates(monthly))
    assert any(t.startswith("2025:") for t in _templates(monthly))
    weekly = hc.weekly_year_overlay(hm.weekly_by_year(frame))
    assert all(t[:4].isdigit() for t in _templates(weekly))


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
    """yoy_movers follows the tab's own date range, so the labels can't say '52'."""
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
# Year overlays                                                                #
# --------------------------------------------------------------------------- #
def test_year_colour_follows_the_year_not_its_rank():
    """The anti-repaint rule, and the easiest thing here to regress.

    The map is built over EVERY year in the frame and the charts merely filter which
    of them they draw. Rebuilding it from a selection would slide each surviving year
    onto a new palette slot, so unticking 2023 would recolour 2024 — a line the
    planner never touched appearing to change identity.
    """
    everything = hc.year_color_map([2023, 2024, 2025, 2026])
    assert len(set(everything.values())) == 4, "years must not share a hue"
    # Keys are strings, because that is what the chart builders look up.
    assert set(everything) == {"2023", "2024", "2025", "2026"}
    # Deterministic: same years in, same hues out, whatever order they arrive in.
    assert hc.year_color_map([2026, 2023, 2025, 2024]) == everything

    # And the reason callers must NOT rebuild it per selection: dropping the oldest
    # year slides every survivor down a slot. This is the failure the view avoids by
    # building the map once over hm.available_years and only filtering at draw time.
    rebuilt = hc.year_color_map([2024, 2025, 2026])
    assert rebuilt["2024"] == everything["2023"], (
        "expected the documented slide — if this ever stops holding, the comment on "
        "year_color_map about why the map is built over all years needs revisiting"
    )


def test_filtering_years_does_not_repaint_the_survivors(frame):
    """End-to-end version of the rule above, through the chart builder."""
    weekly = hm.weekly_by_year(frame)
    years = sorted(weekly["Year"].unique())
    colors = hc.year_color_map(years)

    full = hc.weekly_year_overlay(weekly, colors=colors, years=years)
    trimmed = hc.weekly_year_overlay(weekly, colors=colors, years=years[1:])
    before = {t.name: t.line.color for t in full.data}
    after = {t.name: t.line.color for t in trimmed.data}
    assert set(after) == {str(y) for y in years[1:]}, "wrong years drawn"
    for name, color in after.items():
        assert color == before[name], (
            f"{name} was repainted when an earlier year was deselected"
        )


def test_weekly_overlay_puts_every_year_on_one_calendar_axis(frame):
    """The overlay only works if all years share an x range — one reference year."""
    fig = hc.weekly_year_overlay(hm.weekly_by_year(frame))
    assert len(fig.data) > 1, "fixture should span several years"
    seen_years = {pd.Timestamp(x).year for t in fig.data for x in t.x}
    assert seen_years <= {hm._ALIGN_YEAR, hm._ALIGN_YEAR + 1}, (
        f"weeks leaked out of the reference year: {sorted(seen_years)} — a week "
        f"starting Dec 31 may spill one day into the next year, nothing more"
    )
    # Month names only: the reference year is an axis device and must never show.
    assert fig.layout.xaxis.tickformat == "%b"


def test_weekly_overlay_direct_labels_every_year(frame):
    """Secondary encoding, not decoration.

    The categorical palette's worst adjacent pair sits at ΔE 8.4 (the bottom of the
    legal CVD band) and slot 4 falls under 3:1 on a light surface. Both are only
    permitted with a second, non-colour cue — here the legend plus one end-of-line
    label per year.
    """
    fig = hc.weekly_year_overlay(hm.weekly_by_year(frame))
    assert fig.layout.showlegend is True
    labelled = {a.text.strip() for a in fig.layout.annotations}
    assert labelled == {t.name for t in fig.data}, (
        "every drawn year needs its own end label"
    )


def test_year_overlays_share_one_colour_per_year(frame):
    """A year must be the same hue on both trend charts, or the tab reads as two."""
    years = hm.available_years(frame)
    colors = hc.year_color_map(years)
    weekly = hc.weekly_year_overlay(hm.weekly_by_year(frame), colors=colors,
                                    years=years)
    monthly = hc.monthly_year_overlay(hm.monthly_totals(frame), colors=colors,
                                      years=years)
    line_colors = {t.name: t.line.color for t in weekly.data}
    bar_colors = {t.name: t.marker.color for t in monthly.data}
    assert line_colors == bar_colors


def test_overlays_render_an_empty_panel_when_no_year_is_selected(frame):
    """Deselecting every year is a legitimate state, not a crash."""
    for fig in (hc.weekly_year_overlay(hm.weekly_by_year(frame), years=[]),
                hc.monthly_year_overlay(hm.monthly_totals(frame), years=[])):
        assert not fig.data, "nothing should be plotted"
        assert fig.layout.annotations, "the empty panel must say why it is empty"


def test_dimension_lines_are_not_filled(frame):
    """The stack was replaced precisely because the fills buried the lines."""
    fig = hc.dimension_lines(hm.weekly_by_dimension(frame, "Region"), "Region")
    assert fig.data, "fixture should produce regions"
    for trace in fig.data:
        assert not getattr(trace, "stackgroup", None), "traces must not stack"
        assert getattr(trace, "fill", None) in (None, "none"), (
            f"{trace.name} is filled to the baseline, which is what this replaced"
        )
        assert trace.line.color, "each region carries its own hue on the line itself"


# --------------------------------------------------------------------------- #
# Empty-input tolerance (a filter can legitimately select nothing)            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("build", [
    lambda f: hc.weekly_year_overlay(hm.weekly_by_year(f)),
    lambda f: hc.monthly_year_overlay(hm.monthly_totals(f)),
    lambda f: hc.dimension_lines(hm.weekly_by_dimension(f, "Region"), "Region"),
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
