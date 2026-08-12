"""Plotly figures for the Historical Summary view.

Kept out of charts.py, which stays untouched: that module's figures all share one
shape (actuals flowing into a forecast, with a "forecast ->" divider), and these do
not -- there is no forecast anywhere in this view. They do reuse its
``_base_layout`` (passing ``forecast_start=None``, which it already supports) so
fonts, margins, grid colour and hover behaviour match the rest of the app exactly.

Every figure here is a pure function of a frame: the tabs own their own controls, so
nothing in this module reads or writes Streamlit state.

Colour follows the dataviz method: the form is chosen from the data's job, then
colour is assigned by the job it does.

* Magnitude comparisons (top SKUs, top customers) are ONE hue -- ten bars in ten
  colours would imply ten identities where there is only one measure.
* Identity breakdowns (region, SKU type) use ``config.C_CATEGORICAL``, assigned in
  fixed slot order via ``config.categorical_color_map`` so filtering never repaints
  the categories that survive.
* Years are an identity breakdown too, and take the same categorical palette via
  ``year_color_map``. They were previously drawn on the sequential blue ramp, which
  encodes their ORDER but leaves adjacent years low-contrast -- the trade the trend
  tab now makes the other way, since its whole job is telling years apart. The
  palette's worst adjacent pair sits at the bottom of the acceptable CVD band and one
  slot falls under 3:1 on a light surface, so the year overlays pair the hue with a
  legend AND a direct end-of-line label rather than relying on colour alone.
* Growth/decline uses the app's established green/up, red/down -- never alone,
  always with sorted position and a signed direct label.

No chart here uses two y-axes.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard_app.charts import _base_layout
from dashboard_app.config import (
    C_ACTUAL, C_DECLINE, C_GRID, C_GROWTH, C_OTHER, C_SEPARATOR,
    C_SEQUENTIAL_BLUE, categorical_color_map, fmt_dollar,
)
from dashboard_app.historical_metrics import OTHER_LABEL

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Both live in config.py beside the categorical palette: the SKU-detail donut in
# charts.py needs the same two values, and charts.py cannot import from here (the
# dependency runs the other way -- this module imports _base_layout from it).
_SEPARATOR = C_SEPARATOR
_OTHER_COLOR = C_OTHER


def _money(v):
    return fmt_dollar(v, decimals=0)


def _is_money(value_col):
    return value_col == "revenue"


def _value_label(value_col):
    """Axis / hover label -- title case, with the unit spelled out."""
    return "Revenue (USD)" if _is_money(value_col) else "Units"


def _value_word(value_col):
    """Prose form for chart titles. Separate from _value_label because
    ``"Revenue (USD)".lower()`` produces "revenue (usd)", which reads as a typo."""
    return "revenue" if _is_money(value_col) else "units"


# Prose forms for dimension names. A blanket ``.lower()`` turns "SKU Type" into
# "sku type"; these are the readable forms, plural where a title needs one.
_DIM_WORD = {
    "Region": "region",
    "SKU Type": "SKU type",
    "SKU": "SKU",
    "Customer Grouping": "customer group",
}
_DIM_PLURAL = {
    "Region": "regions",
    "SKU Type": "SKU types",
    "SKU": "SKUs",
    "Customer Grouping": "customer groups",
}


def _dim_word(dim, plural=False):
    table = _DIM_PLURAL if plural else _DIM_WORD
    return table.get(dim, dim.lower())


def _fmt_value(v, value_col):
    if v is None or pd.isna(v):
        return "—"
    return _money(v) if _is_money(value_col) else f"{v:,.0f}"


def _empty_figure(message):
    """A titled blank panel. Better than a bare exception or a silent gap when a
    filter combination legitimately selects nothing."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, xref="paper", yref="paper",
                       x=0.5, y=0.5, font=dict(size=13))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def _color_for(labels):
    """Category -> hex, with OTHER_LABEL pinned to grey and excluded from the
    categorical slots so a real category never loses a slot to the tail bucket."""
    real = [str(x) for x in labels if str(x) != OTHER_LABEL]
    mapping = categorical_color_map(real)
    mapping[OTHER_LABEL] = _OTHER_COLOR
    return mapping


# --------------------------------------------------------------------------- #
# Range control (history only)                                                 #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 1. Trend & seasonality -- years overlaid on one Jan-Dec axis                  #
# --------------------------------------------------------------------------- #
def year_color_map(years):
    """{year -> hex} over the categorical palette, oldest in the first slot.

    Callers MUST build this from every year in the frame, not from the years
    currently selected. Colour follows the entity, never its rank: if the map were
    rebuilt from a selection, deselecting one year would shift every later year onto
    a new slot and repaint lines the planner had not touched.

    Delegates to ``categorical_color_map``, which sorts its keys -- years sort
    correctly as strings, so oldest lands in slot 1 and the assignment is stable.
    """
    return categorical_color_map([str(y) for y in years])


def _month_axis(fig):
    """A Jan-Dec calendar axis for a frame re-dated into ``_ALIGN_YEAR``.

    One tick per month, month names only. The reference year is an implementation
    detail of the overlay and must never appear on screen, which is exactly what
    ``tickformat="%b"`` guarantees.
    """
    fig.update_xaxes(title=None, dtick="M1", tickformat="%b", ticklabelmode="period")
    return fig


def weekly_year_overlay(weekly, value_col="revenue", colors=None, years=None):
    """Every selected year's weekly total, overlaid on a shared Jan-Dec axis.

    Replaces the old single-line ``weekly_trend_chart`` (one continuous line across
    all history) and the old ``seasonality_chart`` (the same curves against a bare
    week number). Plotting years against a real calendar axis answers both questions
    at once, which is why there is now one chart here rather than two.

    ``years`` restricts which years are drawn WITHOUT changing their colours --
    ``colors`` is keyed on the full set. Each line also carries its year as a
    right-edge label: the categorical palette's worst adjacent pair sits at the
    bottom of the acceptable CVD band, and one slot falls under 3:1 on a light
    surface, so identity must not rest on the hue alone.
    """
    if weekly is None or weekly.empty:
        return _empty_figure("No history in this selection.")
    present = sorted(weekly["Year"].unique())
    drawn = [y for y in present if years is None or y in set(years)]
    if not drawn:
        return _empty_figure("No years selected. Pick at least one to plot.")
    colors = colors or year_color_map(present)
    label = _value_label(value_col)

    fig = go.Figure()
    for year in drawn:
        sub = weekly[weekly["Year"] == year].sort_values("SeasonDate")
        if sub.empty:
            continue
        color = colors[str(year)]
        fig.add_trace(go.Scatter(
            x=sub["SeasonDate"], y=sub[value_col], mode="lines", name=str(year),
            line=dict(color=color, width=2),
            # Year kept (it identifies the line); the date dropped -- hovermode
            # "x unified" already prints it once in the box header.
            hovertemplate=(f"{year}: " +
                           ("%{y:$,.0f}" if _is_money(value_col) else "%{y:,.0f}") +
                           "<extra></extra>"),
        ))
        last = sub.iloc[-1]
        fig.add_annotation(
            x=last["SeasonDate"], y=last[value_col], text=f" {year}",
            showarrow=False, xanchor="left", yanchor="middle",
            font=dict(size=11, color=color),
        )
    fig = _base_layout(fig, f"Weekly {_value_word(value_col)} by year",
                       None, y_title=label)
    # Room on the right for the end labels, which sit outside the plotting area.
    fig.update_layout(showlegend=True, margin=dict(r=70))
    _month_axis(fig)
    if _is_money(value_col):
        fig.update_yaxes(tickprefix="$")
    return fig


def monthly_year_overlay(monthly, value_col="revenue", colors=None, years=None):
    """Month totals with every selected year as its own bar in the group.

    Was a two-series current-vs-prior chart on the app's blue/grey actual-vs-
    reference pairing. That pairing only says anything when there are exactly two
    series, so with an arbitrary year selection it gives way to the same categorical
    hue each year carries on the line chart above -- one year, one colour, both
    charts.
    """
    if monthly is None or monthly.empty:
        return _empty_figure("No monthly history in this selection.")
    present = sorted(int(y) for y in monthly["Year"].unique())
    drawn = [y for y in present if years is None or y in set(years)]
    if not drawn:
        return _empty_figure("No years selected. Pick at least one to plot.")
    colors = colors or year_color_map(present)
    label = _value_label(value_col)

    fig = go.Figure()
    for year in drawn:
        sub = monthly[monthly["Year"] == year].set_index("MonthNum")[value_col]
        fig.add_trace(go.Bar(
            x=_MONTHS, y=[sub.get(m, None) for m in range(1, 13)], name=str(year),
            marker=dict(color=colors[str(year)], cornerradius=4),
            # The year is series IDENTITY (every year shares each month slot), so it
            # stays; %{x} was the month, which the unified header already shows.
            hovertemplate=(f"{year}: " +
                           ("%{y:$,.0f}" if _is_money(value_col) else "%{y:,.0f}") +
                           "<extra></extra>"),
        ))
    fig = _base_layout(fig, f"{_value_word(value_col).capitalize()} by month and year",
                       None, y_title=label)
    # bargap/bargroupgap give the 2px-equivalent breathing room between adjacent
    # bars that the mark spec asks for -- and here they are also the secondary
    # encoding that keeps neighbouring years apart when their hues are close.
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08,
                      showlegend=True)
    if _is_money(value_col):
        fig.update_yaxes(tickprefix="$")
    return fig


# --------------------------------------------------------------------------- #
# 2. Mix & breakdown                                                           #
# --------------------------------------------------------------------------- #
def share_donut(totals, dim, value_col="revenue", title=None):
    """Share of the window by one dimension.

    A donut is legitimate here because the parts genuinely sum to a meaningful
    whole and there are few of them; slices are direct-labelled with name and
    percent, so identity never rests on colour alone.
    """
    if totals is None or totals.empty:
        return _empty_figure(f"No {_dim_word(dim)} data in this window.")
    labels = [str(x) for x in totals[dim]]
    colors = _color_for(labels)
    label = _value_label(value_col)
    fig = go.Figure(go.Pie(
        labels=labels, values=totals[value_col], hole=0.55, sort=False,
        marker=dict(colors=[colors[x] for x in labels],
                    line=dict(color=_SEPARATOR, width=2)),
        textinfo="label+percent", textposition="outside",
        # Outside labels sit beyond the pie's own box, so without automargin the
        # topmost one was clipped off the top of the figure. automargin lets Plotly
        # grow the margins to fit the labels instead of cropping them.
        automargin=True,
        hovertemplate=("%{label}<br>" + label + ": " +
                       ("%{value:$,.0f}" if _is_money(value_col)
                        else "%{value:,.0f}") +
                       "<br>Share: %{percent}<extra></extra>"),
    ))
    fig.update_layout(
        title=dict(text=title or
                   f"{_value_word(value_col).capitalize()} share by {_dim_word(dim)}",
                   font=dict(size=16)),
        # Taller, with real top/side room: automargin can only spend margin that
        # exists, and the title needs clearance from whichever label lands highest.
        # Full width now that this is the only donut on the tab.
        height=460, margin=dict(l=40, r=40, t=90, b=30),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def ranked_bars(totals, dim, value_col="revenue", title=None, color=C_ACTUAL):
    """Horizontal bars ranking one dimension by a single measure.

    ONE hue on purpose: every bar measures the same thing, so ten colours would
    invent ten identities. Largest sits at the top (Plotly draws the first
    category at the bottom, hence the reversal).
    """
    if totals is None or totals.empty:
        return _empty_figure(f"No {_dim_word(dim)} data in this window.")
    ordered = totals.sort_values(value_col, ascending=True)
    label = _value_label(value_col)
    text = [_fmt_value(v, value_col) for v in ordered[value_col]]
    fig = go.Figure(go.Bar(
        x=ordered[value_col], y=[str(x) for x in ordered[dim]], orientation="h",
        marker=dict(color=color, cornerradius=4),
        text=text, textposition="auto",
        hovertemplate=("%{y}<br>" + label + ": " +
                       ("%{x:$,.0f}" if _is_money(value_col) else "%{x:,.0f}") +
                       "<extra></extra>"),
    ))
    fig = _base_layout(
        fig,
        title or f"Top {_dim_word(dim, plural=True)} by {_value_word(value_col)}",
        None, y_title=None)
    fig.update_layout(showlegend=False, bargap=0.3,
                      height=max(340, 34 * len(ordered) + 130))
    fig.update_xaxes(title=label, tickprefix="$" if _is_money(value_col) else None)
    fig.update_yaxes(rangemode="normal", tickformat=None, automargin=True)
    return fig


def dimension_lines(weekly_dim, dim, value_col="revenue", title=None):
    """How the mix shifts over time -- one plain line per category.

    Was a stacked area, on the reasoning that the parts sum to the total. In practice
    the fills buried the lines: a stacked band's height is its own value but its
    POSITION is the sum of everything beneath it, so only the bottom category could
    be read against the axis, and a mid-sized region's week-to-week movement was
    invisible under the region above it. Unstacked lines give every category the same
    baseline and make them directly comparable, which is the question this chart is
    actually asked. The running total is not lost -- the donut beside it is the
    whole-period share.

    No fill, by the same argument: a filled line chart re-introduces exactly the
    occlusion the stack had.
    """
    if weekly_dim is None or weekly_dim.empty:
        return _empty_figure(f"No {_dim_word(dim)} history in this selection.")
    label = _value_label(value_col)
    categories = [str(x) for x in weekly_dim[dim].unique()]
    colors = _color_for(categories)
    # Largest first, so the legend reads in the order the eye ranks the lines.
    order = (weekly_dim.groupby(dim)[value_col].sum(min_count=1)
             .sort_values(ascending=False).index)
    fig = go.Figure()
    for cat in order:
        sub = weekly_dim[weekly_dim[dim] == cat].sort_values("WeekDate")
        fig.add_trace(go.Scatter(
            x=sub["WeekDate"], y=sub[value_col], name=str(cat),
            mode="lines", line=dict(width=2, color=colors[str(cat)]),
            # Region + amount only. With one row per region in a unified hover,
            # repeating the week date on every row buried the actual numbers.
            hovertemplate=(str(cat) + ": " +
                           ("%{y:$,.0f}" if _is_money(value_col) else "%{y:,.0f}") +
                           "<extra></extra>"),
        ))
    fig = _base_layout(
        fig,
        title or f"Weekly {_value_word(value_col)} by {_dim_word(dim)}",
        None, y_title=label)
    if _is_money(value_col):
        fig.update_yaxes(tickprefix="$")
    return fig


# --------------------------------------------------------------------------- #
# 3. Movers & concentration                                                    #
# --------------------------------------------------------------------------- #
def movers_chart(movers, n=10):
    """Year-over-year gainers and decliners as diverging bars around zero.

    Green/red is not colourblind-safe on its own, so it never carries the meaning
    alone here: gainers and decliners sort into contiguous blocks either side of
    the zero line, and every bar is direct-labelled with a signed dollar figure.
    """
    if movers is None or movers.empty:
        return _empty_figure("No year-over-year movement to show.")
    ordered = movers.sort_values("delta", ascending=True)
    colors = [C_GROWTH if d >= 0 else C_DECLINE for d in ordered["delta"]]
    labels = [f"{fmt_dollar(d, signed=True)}" for d in ordered["delta"]]
    fig = go.Figure(go.Bar(
        x=ordered["delta"], y=ordered["SKU"].astype(str), orientation="h",
        marker=dict(color=colors, cornerradius=4),
        text=labels, textposition="auto",
        customdata=np.stack([
            ordered["Description"].fillna("").astype(str),
            ordered["current"], ordered["prior"],
        ], axis=-1),
        # Neither side names a week count or a span: the comparison follows whatever
        # the caller passed (see historical_metrics.yoy_movers), which for this view
        # is the selected year against the same span one calendar year earlier. The
        # tab prints the actual comparison dates above the chart.
        hovertemplate=("%{y} — %{customdata[0]}<br>"
                       "Selected period: %{customdata[1]:$,.0f}<br>"
                       "Same period last year: %{customdata[2]:$,.0f}<br>"
                       "Change: %{x:$,.0f}<extra></extra>"),
    ))
    fig = _base_layout(fig, f"Biggest year-over-year movers (top {n} each way)",
                       None, y_title=None)
    fig.update_layout(showlegend=False, bargap=0.3,
                      height=max(360, 30 * len(ordered) + 130))
    fig.update_xaxes(title="Revenue change vs the same period last year",
                     tickprefix="$",
                     zeroline=True, zerolinewidth=1, zerolinecolor=C_GRID)
    fig.update_yaxes(rangemode="normal", tickformat=None, automargin=True)
    return fig


def pareto_chart(pareto_df):
    """Cumulative revenue share against SKU rank -- how concentrated the business is.

    One y-axis only. The obvious "improvement" here is a second axis carrying
    per-SKU revenue bars; that is the dual-axis anti-pattern and is deliberately
    not done. The 80% guide makes the classic reading immediate.
    """
    if pareto_df is None or pareto_df.empty:
        return _empty_figure("No priced revenue to rank in this window.")
    fig = go.Figure(go.Scatter(
        x=pareto_df["rank"], y=pareto_df["cum_share"], mode="lines",
        line=dict(color=C_ACTUAL, width=2), name="Cumulative share",
        customdata=np.stack([pareto_df["SKU"].astype(str),
                             pareto_df["revenue"]], axis=-1),
        hovertemplate=("Rank %{x} — %{customdata[0]}<br>"
                       "SKU revenue: %{customdata[1]:$,.0f}<br>"
                       "Cumulative: %{y:.1f}%<extra></extra>"),
    ))
    fig.add_hline(y=80, line_width=1, line_dash="dot",
                  line_color="rgba(100,116,139,0.7)")
    fig.add_annotation(x=0, xref="paper", y=80, yanchor="bottom", xanchor="left",
                       text="80% of revenue", showarrow=False,
                       font=dict(size=11, color="rgba(100,116,139,0.95)"))
    n80 = int((pareto_df["cum_share"] <= 80).sum()) + 1
    total = len(pareto_df)
    fig = _base_layout(
        fig,
        f"Revenue concentration — {n80} of {total} SKUs make 80%",
        None, y_title="Cumulative share of revenue (%)",
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="SKUs, highest revenue first")
    fig.update_yaxes(range=[0, 101], ticksuffix="%", tickformat=".0f")
    return fig


# --------------------------------------------------------------------------- #
# 4. Seasonal heatmap                                                          #
# --------------------------------------------------------------------------- #
def _cell_text_color(value, vmax):
    """Ink for a heatmap cell, chosen from the CELL's own fill rather than the page
    theme -- the cells tile the plot area, so the surface never shows through and
    the theme is irrelevant to their legibility."""
    if value is None or pd.isna(value) or not vmax:
        return "rgba(0,0,0,0)"
    return "#ffffff" if (value / vmax) > 0.55 else "#0b0b0b"


def month_year_heatmap(matrix, value_col="revenue"):
    """Month x year grid -- seasonal pockets and anomalies at a glance.

    Sequential single-hue ramp (never a rainbow), light = low, dark = high.
    """
    if matrix is None or matrix.empty or matrix.isna().all().all():
        return _empty_figure("Not enough history for a seasonal heatmap.")
    years = [str(y) for y in matrix.columns]
    values = matrix.to_numpy(dtype="float64")
    vmax = np.nanmax(values) if np.isfinite(values).any() else 0.0
    label = _value_label(value_col)
    colorscale = [[i / (len(C_SEQUENTIAL_BLUE) - 1), c]
                  for i, c in enumerate(C_SEQUENTIAL_BLUE)]

    fig = go.Figure(go.Heatmap(
        z=values, x=years, y=_MONTHS, colorscale=colorscale,
        xgap=2, ygap=2,   # the 2px surface gap between adjacent fills
        hoverongaps=False,
        colorbar=dict(title=label, tickprefix="$" if _is_money(value_col) else None),
        hovertemplate=("%{y} %{x}<br>" + label + ": " +
                       ("%{z:$,.0f}" if _is_money(value_col) else "%{z:,.0f}") +
                       "<extra></extra>"),
    ))
    # Direct labels: the grid is small enough that every cell can carry its value,
    # which also covers the contrast-relief rule for the lighter ramp steps.
    #
    # Positioned by INDEX, not by category name. On a `category` axis an annotation
    # coordinate is a category index, and Plotly resolves it numeric-FIRST: a
    # numeric-looking name like "2023" is read as the coordinate 2023 rather than as
    # the label of column 0. Passing the year string here put every label at
    # x = 2023..2026 in index units, stretched the autorange to 2026, and squashed
    # the real columns into a sliver at the left. ("Jan" survived that bug only
    # because it isn't numeric.) Indices are what the axis actually speaks.
    for col in range(len(years)):
        for row in range(len(_MONTHS)):
            v = values[row][col]
            if pd.isna(v):
                continue
            fig.add_annotation(
                x=col, y=row, text=_fmt_value(v, value_col), showarrow=False,
                font=dict(size=10, color=_cell_text_color(v, vmax)),
            )
    fig.update_layout(
        title=dict(text=f"{_value_word(value_col).capitalize()} by month and year",
                   font=dict(size=16)),
        # b=30 (not 10) leaves the year tick labels room along the bottom -- the same
        # clipping share_donut's automargin fixed.
        height=520, margin=dict(l=10, r=10, t=80, b=30),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        # A heatmap carries a colorbar, not a series legend.
        showlegend=False,
    )
    # Years are CATEGORIES, not a number line. They are passed as strings, but Plotly
    # autodetects a linear axis from numeric-looking strings and then interpolates
    # half-steps — which is where "2022.5", "2023.5" came from. Forcing the axis type
    # gives exactly one centred tick per year column.
    fig.update_xaxes(type="category", automargin=True)
    fig.update_yaxes(autorange="reversed")
    return fig
