"""Plotly figures for the Historical Summary view.

Kept out of charts.py, which stays untouched: that module's figures all share one
shape (actuals flowing into a forecast, with a "forecast ->" divider), and these
eleven do not -- there is no forecast anywhere in this view. They do reuse its
``_base_layout`` (passing ``forecast_start=None``, which it already supports) so
fonts, margins, grid colour and hover behaviour match the rest of the app exactly.

Colour follows the dataviz method: the form is chosen from the data's job, then
colour is assigned by the job it does.

* Magnitude comparisons (top SKUs, top customers) are ONE hue -- ten bars in ten
  colours would imply ten identities where there is only one measure.
* Identity breakdowns (region, SKU type) use ``config.C_CATEGORICAL``, assigned in
  fixed slot order via ``config.categorical_color_map`` so filtering never repaints
  the categories that survive.
* Year-over-year pairs reuse the app's existing semantics: blue = actual/current,
  grey = the reference line being compared against.
* Ordered years (the seasonality overlay) use the sequential blue ramp, oldest
  lightest, so recency reads off the colour without needing the legend.
* Growth/decline uses the app's established green/up, red/down -- never alone,
  always with sorted position and a signed direct label.

No chart here uses two y-axes.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_app.charts import _base_layout
from dashboard_app.config import (
    C_ACTUAL, C_CATEGORICAL, C_DECLINE, C_GRID, C_GROWTH, C_ORIGINAL,
    C_SEQUENTIAL_BLUE, categorical_color_map, fmt_dollar,
)
from dashboard_app.historical_metrics import OTHER_LABEL

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Translucent neutral: lightens against a dark surface and darkens against a light
# one, so one value separates stacked fills in both themes (the app's trace colours
# are deliberately theme-invariant -- see charts.py).
_SEPARATOR = "rgba(128,128,128,0.45)"

# Grey used for a fold-to-tail bucket, so "Other" never impersonates a real
# category by borrowing a categorical slot.
_OTHER_COLOR = C_ORIGINAL


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
def history_range_control(frame, key, default="2 Years"):
    """Date-range picker for the trend charts.

    Deliberately separate from ``charts.chart_range_control`` (which is untouched):
    that control exists to trim history while always keeping the forecast horizon
    visible, and this view has no forecast. Presets here start at 1 year because
    the questions these charts answer -- seasonality, year-over-year -- need at
    least two seasons on screen.

    Returns a (start, end) pair of Timestamps, or None when there is no data.
    """
    if frame is None or frame.empty:
        return None
    presets = {
        "1 Year": pd.DateOffset(years=1),
        "2 Years": pd.DateOffset(years=2),
        "3 Years": pd.DateOffset(years=3),
        "All": None,
        "Custom…": "custom",
    }
    data_min = pd.to_datetime(frame["WeekDate"]).min()
    data_max = pd.to_datetime(frame["WeekDate"]).max()

    choice = st.selectbox(
        "Date range", list(presets), index=list(presets).index(default),
        key=f"{key}_preset", help="How much history to show in this chart.",
    )
    if choice == "Custom…":
        picked = st.date_input(
            "Custom range",
            value=(max(data_min, data_max - pd.DateOffset(years=1)).date(),
                   data_max.date()),
            min_value=data_min.date(), max_value=data_max.date(),
            key=f"{key}_custom",
            help="Click the calendar or type dates. Pick a start and an end.",
        )
        if isinstance(picked, (tuple, list)) and len(picked) == 2:
            return pd.Timestamp(picked[0]), pd.Timestamp(picked[1])
        return data_min, data_max
    if choice == "All":
        return data_min, data_max
    return max(data_min, data_max - presets[choice]), data_max


# --------------------------------------------------------------------------- #
# 1. Trend & seasonality                                                       #
# --------------------------------------------------------------------------- #
def weekly_trend_chart(weekly, value_col="revenue"):
    """Total demand week by week -- the shape of the business over time.

    One series, so no legend box: the title names it (dataviz -- a legend for a
    single series is noise).
    """
    if weekly is None or weekly.empty:
        return _empty_figure("No history in this selection.")
    label = _value_label(value_col)
    hover = ("%{x|%b %d, %Y}<br>" +
             (f"{label}: %{{y:$,.0f}}" if _is_money(value_col)
              else f"{label}: %{{y:,.0f}}") + "<extra></extra>")
    fig = go.Figure(go.Scatter(
        x=weekly["WeekDate"], y=weekly[value_col], mode="lines",
        line=dict(color=C_ACTUAL, width=2), name=label, hovertemplate=hover,
    ))
    fig = _base_layout(fig, f"Weekly {_value_word(value_col)}", None, y_title=label)
    fig.update_layout(showlegend=False)
    if _is_money(value_col):
        fig.update_yaxes(tickprefix="$")
    return fig


def monthly_yoy_chart(monthly, value_col="revenue", current_year=None):
    """This year against last, month by month -- grouped bars, not stacked.

    Two series only (current + prior), reusing the app's blue = actual / grey =
    reference pairing rather than two categorical slots, so the comparison reads
    the same way it does on every projection chart.
    """
    if monthly is None or monthly.empty:
        return _empty_figure("No monthly history in this selection.")
    years = sorted(monthly["Year"].unique())
    current_year = current_year or years[-1]
    prior_year = current_year - 1
    label = _value_label(value_col)

    def _series(year):
        sub = monthly[monthly["Year"] == year].set_index("MonthNum")[value_col]
        return [sub.get(m, None) for m in range(1, 13)]

    fig = go.Figure()
    for year, color in ((prior_year, C_ORIGINAL), (current_year, C_ACTUAL)):
        if year not in years:
            continue
        fig.add_trace(go.Bar(
            x=_MONTHS, y=_series(year), name=str(year),
            marker=dict(color=color, cornerradius=4),
            hovertemplate=(f"{year} %{{x}}<br>{label}: " +
                           ("%{y:$,.0f}" if _is_money(value_col) else "%{y:,.0f}") +
                           "<extra></extra>"),
        ))
    fig = _base_layout(
        fig,
        f"{_value_word(value_col).capitalize()} by month — "
        f"{current_year} vs {prior_year}",
        None, y_title=label)
    # bargap/bargroupgap give the 2px-equivalent breathing room between adjacent
    # bars that the mark spec asks for.
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08)
    if _is_money(value_col):
        fig.update_yaxes(tickprefix="$")
    return fig


def seasonality_chart(seasonal, value_col="revenue"):
    """Every year overlaid on a shared week-of-year axis -- when the season hits.

    Years are ORDERED, so they take the sequential blue ramp (oldest lightest)
    rather than categorical slots: recency then reads straight off the colour.
    """
    if seasonal is None or seasonal.empty:
        return _empty_figure("No seasonal history in this selection.")
    label = _value_label(value_col)
    years = sorted(seasonal["Year"].unique())
    # Spread the years across the darker half of the ramp so even the oldest line
    # clears the surface; the newest year lands on the darkest step.
    steps = np.linspace(3, len(C_SEQUENTIAL_BLUE) - 1, num=max(len(years), 1))
    fig = go.Figure()
    for year, step in zip(years, steps):
        sub = seasonal[seasonal["Year"] == year].sort_values("WeekOfYear")
        is_latest = year == years[-1]
        fig.add_trace(go.Scatter(
            x=sub["WeekOfYear"], y=sub[value_col], mode="lines", name=str(year),
            line=dict(color=C_SEQUENTIAL_BLUE[int(round(step))],
                      width=3 if is_latest else 2),
            hovertemplate=(f"{year} · week %{{x}}<br>{label}: " +
                           ("%{y:$,.0f}" if _is_money(value_col) else "%{y:,.0f}") +
                           "<extra></extra>"),
        ))
    fig = _base_layout(fig,
                       f"Seasonality — {_value_word(value_col)} by week of year",
                       None, y_title=label)
    fig.update_xaxes(title="Week of year")
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
        hovertemplate=("%{label}<br>" + label + ": " +
                       ("%{value:$,.0f}" if _is_money(value_col)
                        else "%{value:,.0f}") +
                       "<br>Share: %{percent}<extra></extra>"),
    ))
    fig.update_layout(
        title=dict(text=title or
                   f"{_value_word(value_col).capitalize()} share by {_dim_word(dim)}",
                   font=dict(size=16)),
        height=420, margin=dict(l=10, r=10, t=80, b=10),
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


def stacked_area(stacked, dim, value_col="revenue", title=None):
    """How the mix shifts over time. Stacked because the parts sum to the total."""
    if stacked is None or stacked.empty:
        return _empty_figure(f"No {_dim_word(dim)} history in this selection.")
    label = _value_label(value_col)
    categories = [str(x) for x in stacked[dim].unique()]
    colors = _color_for(categories)
    # Largest total at the bottom of the stack: the biggest band gets the stable
    # baseline, so the smaller ones above it stay readable.
    order = (stacked.groupby(dim)[value_col].sum(min_count=1)
             .sort_values(ascending=False).index)
    fig = go.Figure()
    for cat in order:
        sub = stacked[stacked[dim] == cat].sort_values("WeekDate")
        fig.add_trace(go.Scatter(
            x=sub["WeekDate"], y=sub[value_col], name=str(cat),
            mode="lines", stackgroup="one",
            line=dict(width=1.5, color=_SEPARATOR),
            fillcolor=colors[str(cat)],
            hovertemplate=("%{x|%b %d, %Y}<br>" + str(cat) + ": " +
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
        hovertemplate=("%{y} — %{customdata[0]}<br>"
                       "Last 52 wks: %{customdata[1]:$,.0f}<br>"
                       "Prior 52 wks: %{customdata[2]:$,.0f}<br>"
                       "Change: %{x:$,.0f}<extra></extra>"),
    ))
    fig = _base_layout(fig, f"Biggest year-over-year movers (top {n} each way)",
                       None, y_title=None)
    fig.update_layout(showlegend=False, bargap=0.3,
                      height=max(360, 30 * len(ordered) + 130))
    fig.update_xaxes(title="Revenue change vs prior 52 weeks", tickprefix="$",
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
    for col, year in enumerate(years):
        for row, month in enumerate(_MONTHS):
            v = values[row][col]
            if pd.isna(v):
                continue
            fig.add_annotation(
                x=year, y=month, text=_fmt_value(v, value_col), showarrow=False,
                font=dict(size=10, color=_cell_text_color(v, vmax)),
            )
    fig.update_layout(
        title=dict(text=f"{_value_word(value_col).capitalize()} by month and year",
                   font=dict(size=16)),
        height=520, margin=dict(l=10, r=10, t=80, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        # A heatmap carries a colorbar, not a series legend.
        showlegend=False,
    )
    fig.update_yaxes(autorange="reversed")
    return fig
