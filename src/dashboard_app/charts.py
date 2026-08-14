"""Plotly chart builders and the per-chart date-range control."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_app.config import (
    C_ACTUAL, C_UPDATED, C_ORIGINAL, C_GRID, C_OTHER, C_SEPARATOR,
    EIGHT_WK_AVG_COL, MIXED_SOURCE, RISK_COL, categorical_color_map, fmt_dollar,
)
from dashboard_app.summaries import historical_window


# --------------------------------------------------------------------------- #
# Charts                                                                      #
# --------------------------------------------------------------------------- #
# App font stack — kept in sync with .streamlit/config.toml so chart text matches
# the rest of the UI. Only family/size are set on the figure; text COLORS are left
# unset on purpose so Streamlit's built-in plotly theme (theme="streamlit", the
# st.plotly_chart default) recolors them to match whichever theme is actually
# displayed — it swaps placeholder colors on the frontend based on the live
# background, which is the only reliable way to stay legible in both light and
# dark (a server-side st.context.theme.type read can lag the displayed theme and
# would paint near-white titles onto a light page).
_CHART_FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def _base_layout(fig, title, forecast_start, y_title="Units (POS / Orders)"):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        font=dict(family=_CHART_FONT, size=13),
        margin=dict(l=10, r=10, t=80, b=10),
        height=420,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=0.98, x=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        # namelength=-1 shows the full trace name in the unified-hover box (Plotly's
        # default of 15 chars truncated "Updated forecast (from POS)" to "Updated fore…").
        hoverlabel=dict(namelength=-1),
    )
    # Translucent grid/divider colors read correctly on both light and dark
    # surfaces, so these stay explicit. tickformat=",.0f" prints plain grouped
    # integers (83,895) instead of Plotly's default SI abbreviation (83.895k).
    fig.update_xaxes(gridcolor=C_GRID, title=None)
    fig.update_yaxes(gridcolor=C_GRID, rangemode="tozero", title=y_title,
                     tickformat=",.0f")
    if forecast_start is not None:
        fig.add_vline(
            x=forecast_start, line_width=1, line_dash="dot",
            line_color="rgba(100,116,139,0.7)",
        )
        fig.add_annotation(
            x=forecast_start, yref="paper", y=0.93, yanchor="bottom",
            text="forecast →", showarrow=False,
            font=dict(size=11, color="rgba(100,116,139,0.95)"),
            xshift=4,
        )
    return fig


def _clip_to_range(df, date_range):
    """Clip a trace frame to a chart date-range window on WeekDate (Y auto-fits).

    date_range is None (no clipping — current behavior) or a (start, end) pair of
    Timestamps. Empty frames pass through untouched.
    """
    if date_range is None or df.empty:
        return df
    s, e = date_range
    return df[(df["WeekDate"] >= s) & (df["WeekDate"] <= e)]


# --------------------------------------------------------------------------- #
# Hover revenue-difference helpers                                            #
# --------------------------------------------------------------------------- #
# Setting an explicit hovertemplate suppresses "x unified" mode's automatic
# trace-name label, so each data line's template must carry its OWN
# "<label>: <value>" text (otherwise a hovered row shows just a colored swatch +
# number, with no telling which line is which). The "Revenue Risk: $X" figure is
# NOT a <br> line on the Original-projection trace — that would tie it to the grey
# swatch — but a separate, swatch-less companion trace (see _revenue_risk_trace)
# so it reads as its own bottom row, color-coded green (>=0) / red (<0).

_RISK_POS = "#16a34a"   # green — plan gap adds revenue (matches Exceptions card)
_RISK_NEG = "#dc2626"   # red   — plan gap removes revenue


def _hover_template(label):
    """A unified-hover template showing ``<label>: <value>`` (comma-grouped).
    ``<extra></extra>`` suppresses Plotly's secondary trace-name box."""
    return f"{label}: %{{y:,.0f}}<extra></extra>"


def _as_price_map(prices):
    """Normalize a SKU->price input (dict, pandas Series, or None) to
    ``{str(sku): float}``. Returns ``{}`` when nothing usable is supplied, so a
    caller with no list prices simply gets plain (value-only) hovers."""
    if prices is None:
        return {}
    if isinstance(prices, (pd.Series, dict)):
        items = prices.items()
    else:
        return {}
    out = {}
    for k, v in items:
        val = pd.to_numeric(v, errors="coerce")
        if not pd.isna(val):
            out[str(k)] = float(val)
    return out


def _rev_by_week(frame, unit_col, price_map):
    """Per-week revenue = Σ(units × that SKU's price), as a WeekDate-indexed
    Series. Empty when no price map (so the caller shows plain hovers)."""
    if not price_map or frame is None or frame.empty:
        return pd.Series(dtype="float64")
    f = frame[["SKU", "WeekDate", unit_col]].copy()
    f["_rev"] = (pd.to_numeric(f[unit_col], errors="coerce")
                 * f["SKU"].astype(str).map(price_map))
    return f.groupby("WeekDate")["_rev"].sum(min_count=1)


def _revenue_risk_trace(week_series, risk_by_week):
    """An invisible companion Scatter that renders "Revenue Risk: $X" as its OWN
    swatch-less row in the unified hover — a ``<br>`` on the Original line would
    tie the figure to the grey swatch. Points sit at y=0 (the axis already
    includes 0 via rangemode="tozero", so no visual/range change) which sorts the
    row to the bottom; the dollar figure is color-coded green (>=0) / red (<0) via
    an inline ``<span>``. Returns ``None`` when there is no risk to show (no prices
    loaded, or no week where both compared lines have a value)."""
    if risk_by_week is None or len(risk_by_week) == 0:
        return None
    lookup = dict(risk_by_week)
    xs, cd = [], []
    for w in week_series:
        v = lookup.get(pd.Timestamp(w))
        if v is None or pd.isna(v):
            continue
        colour = _RISK_POS if v >= 0 else _RISK_NEG
        xs.append(w)
        cd.append([f"<span style='color:{colour}'>{fmt_dollar(v, signed=True)}</span>"])
    if not xs:
        return None
    return go.Scatter(
        x=xs, y=[0] * len(xs), mode="markers",
        marker=dict(color="rgba(0,0,0,0)", size=0.1), showlegend=False,
        customdata=cd,
        hovertemplate="Revenue Risk: %{customdata[0]}<extra></extra>",
    )


def chart_range_control(agg, weekly, lcw, key):
    """Compact date-range picker rendered right above a chart.

    Returns a (view_start, view_end) pair of Timestamps used to clip that chart's
    traces so its Y-axis auto-fits the visible window. Each chart gets its own
    control (unique `key`) and thus its own independent range.

    Presets trim history only — the forecast horizon always stays visible.
    "Custom…" reveals a calendar / typeable range picker.
    """
    RANGE_PRESETS = {
        "1 Month":  pd.DateOffset(months=1),
        "3 Months": pd.DateOffset(months=3),
        "6 Months": pd.DateOffset(months=6),
        "9 Months": pd.DateOffset(months=9),
        "1 Year":   pd.DateOffset(years=1),
        "2 Years":  pd.DateOffset(years=2),
        "3 Years":  pd.DateOffset(years=3),
        "All":      None,
        "Custom…":  "custom",
    }
    data_min = pd.to_datetime(agg["WeekDate"]).min()
    horizon_end = pd.to_datetime(weekly["WeekDate"]).max()

    preset = st.selectbox(
        "Date range", list(RANGE_PRESETS),
        index=list(RANGE_PRESETS).index("6 Months"),
        key=f"{key}_preset",
        help="How much history to show. The forecast always stays visible.",
    )
    if preset == "Custom…":
        default_start = max(data_min, horizon_end - pd.DateOffset(months=6))
        picked = st.date_input(
            "Custom range",
            value=(default_start.date(), horizon_end.date()),
            min_value=data_min.date(), max_value=horizon_end.date(),
            key=f"{key}_custom",
            help="Click the calendar or type dates. Pick a start and an end.",
        )
        # date_input returns a single date mid-selection; apply once both ends chosen.
        if isinstance(picked, (tuple, list)) and len(picked) == 2:
            return pd.Timestamp(picked[0]), pd.Timestamp(picked[1])
        return data_min, horizon_end
    if preset == "All":
        return data_min, horizon_end
    # Preset controls history start; forecast ALWAYS stays visible.
    return max(data_min, lcw - RANGE_PRESETS[preset]), horizon_end


def aggregate_chart(agg, summary, weekly, anchors, view, date_range=None,
                    prices=None):
    """Total actual demand (historical window) flowing into total forecast (15 wks).

    Historical demand uses each SKU's forecast source (POS or Orders) so the
    actual total is comparable to the forecast total. When date_range is given,
    the plotted traces are clipped to that window so the Y-axis rescales to fit.
    ``prices`` (a SKU->list-price map/Series) adds a "Revenue Risk: $X" row to
    the Original-projection line's hover — the totals are summed across SKUs, so
    revenue is computed per SKU (units × that SKU's price) BEFORE the groupby and
    then summed.
    """
    lb, lcw, ffw = anchors
    pm = _as_price_map(prices)

    hist = historical_window(agg, summary, anchors)
    hist_tot = hist.groupby("WeekDate")["demand"].sum(min_count=1).reset_index()

    fc = weekly.copy()
    fc["WeekDate"] = pd.to_datetime(fc["WeekDate"])
    fc_tot = fc.groupby("WeekDate")["projected_pos"].sum().reset_index()

    # Original projection: plot straight from the spreadsheet's Projection column
    # across the SAME span shown for actuals + forecast (history start through the
    # forecast horizon), so the grey line runs the full width of the chart rather
    # than only over the 15 forecast weeks. Weeks with no Projection are dropped
    # (the line simply connects the weeks that have a value); no recomputation.
    horizon_end = pd.to_datetime(weekly["WeekDate"]).max()
    sys_proj = agg[
        (agg["WeekDate"] >= lb) & (agg["WeekDate"] <= horizon_end)
    ].dropna(subset=["Projection"])
    sys_tot = sys_proj.groupby("WeekDate")["Projection"].sum().reset_index()

    # Revenue Risk = the plan-gap valued at each SKU's list price and summed (so
    # it matches the difference of the two plotted total lines): Actual − plan in
    # history weeks, forecast − plan in forecast weeks. Subtraction aligns on
    # WeekDate → NaN where either side is absent → dropna; the two ranges are
    # disjoint so concat gives one risk-per-week series for the Original line.
    rev_orig = _rev_by_week(sys_proj, "Projection", pm)
    risk_by_week = pd.concat([
        (_rev_by_week(hist, "demand", pm) - rev_orig).dropna(),
        (_rev_by_week(fc, "projected_pos", pm) - rev_orig).dropna(),
    ])

    # Clip every plotted trace to the chosen chart window so the Y-axis auto-fits
    # the visible weeks (does not affect the summary/forecast math).
    hist_tot = _clip_to_range(hist_tot, date_range)
    fc_tot = _clip_to_range(fc_tot, date_range)
    sys_tot = _clip_to_range(sys_tot, date_range)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_tot["WeekDate"], y=hist_tot["demand"], name="Actual demand",
        mode="lines+markers", line=dict(color=C_ACTUAL, width=3),
        marker=dict(size=6), hovertemplate=_hover_template("Actual demand"),
    ))
    if not hist_tot.empty and not fc_tot.empty:
        fig.add_trace(go.Scatter(
            x=[hist_tot["WeekDate"].iloc[-1], fc_tot["WeekDate"].iloc[0]],
            y=[hist_tot["demand"].iloc[-1], fc_tot["projected_pos"].iloc[0]],
            mode="lines", showlegend=False,
            line=dict(color=C_UPDATED, width=2, dash="dot"), hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=fc_tot["WeekDate"], y=fc_tot["projected_pos"], name="Updated forecast",
        mode="lines+markers", line=dict(color=C_UPDATED, width=3, dash="dash"),
        marker=dict(size=6), hovertemplate=_hover_template("Updated forecast"),
    ))
    if not sys_tot.empty:
        fig.add_trace(go.Scatter(
            x=sys_tot["WeekDate"], y=sys_tot["Projection"], name="Original projection",
            mode="lines+markers", line=dict(color=C_ORIGINAL, width=2, dash="dot"),
            marker=dict(size=5),
            hovertemplate=_hover_template("Original projection"),
        ))
    rr = _revenue_risk_trace(sys_tot["WeekDate"], risk_by_week)
    if rr is not None:
        fig.add_trace(rr)
    return _base_layout(fig, f"Total weekly demand — {view}", ffw)


def _sku_title(sku, desc):
    """``"SKU — Description"``, or the bare SKU when there is no description.

    Truthiness, not ``isinstance(desc, str)``: ``agent.data_io._clean`` strips the
    warehouse's fixed-width padding, so a description that was nothing but spaces
    now arrives as ``""`` — which is a str, and would have titled the chart
    ``"BT1028 — "``. (It is deliberately left as ``""`` rather than turned into NaN
    at ingestion: the models group by ["SKU", "Description"] with dropna=True, so
    NaN would drop the SKU from the forecast entirely.)
    """
    desc = desc.strip() if isinstance(desc, str) else ""
    return f"{sku} — {desc}" if desc else str(sku)


def sku_chart(sku, desc, source, agg, weekly, anchors, date_range=None,
              prices=None):
    """Per-SKU: actuals (historical window, from its source) + updated forecast + original proj.

    When date_range is given, the plotted traces are clipped to that window so the
    Y-axis rescales to fit the visible weeks. ``prices`` (a SKU->list-price
    map/Series) adds the "Revenue Risk: $X" row to the Original-projection hover.

    An ``agg`` carrying a pre-resolved ``demand`` column (see
    ``summaries.resolve_demand``) is plotted from THAT rather than from ``source``.
    For a rolled-up SKU whose customers mix POS and Orders no single column is the
    actual demand — ``source`` is then ``MIXED_SOURCE`` — and picking one would draw
    an actuals line below the forecast the tiles beside it report.
    """
    lb, lcw, ffw = anchors
    pm = _as_price_map(prices)

    a = agg[agg["SKU"].astype(str) == str(sku)].sort_values("WeekDate")
    if "demand" in a.columns:
        col = "demand"
        label = "POS + Orders" if source == MIXED_SOURCE else source
    else:
        col = "Orders" if source == "Orders" else "POS"
        label = source
    hist = a[(a["WeekDate"] >= lb) & (a["WeekDate"] <= lcw)].dropna(subset=[col])
    # Original projection: straight from the spreadsheet's Projection column,
    # across the SAME span shown for actuals + forecast (history start through the
    # forecast horizon), so the grey line runs the full width of the chart. Weeks
    # with no Projection are dropped; no recomputation.
    horizon_end = pd.to_datetime(weekly["WeekDate"]).max()
    sys_proj = a[
        (a["WeekDate"] >= lb) & (a["WeekDate"] <= horizon_end)
    ].dropna(subset=["Projection"])

    fc = weekly[weekly["SKU"].astype(str) == str(sku)].copy()
    fc["WeekDate"] = pd.to_datetime(fc["WeekDate"])

    # Revenue Risk (this SKU's price × plan-gap), shown on the Original line: it
    # appears once per hover and is Actual − plan in history weeks, forecast − plan
    # in forecast weeks. Empty when no prices supplied.
    rev_orig = _rev_by_week(sys_proj, "Projection", pm)
    risk_by_week = pd.concat([
        (_rev_by_week(hist, col, pm) - rev_orig).dropna(),
        (_rev_by_week(fc, "projected_pos", pm) - rev_orig).dropna(),
    ])

    # Clip every plotted trace to the chosen chart window so the Y-axis auto-fits.
    hist = _clip_to_range(hist, date_range)
    fc = _clip_to_range(fc, date_range)
    sys_proj = _clip_to_range(sys_proj, date_range)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["WeekDate"], y=hist[col], name=f"Actual {label}",
        mode="lines+markers", line=dict(color=C_ACTUAL, width=3),
        marker=dict(size=7), hovertemplate=_hover_template(f"Actual {label}"),
    ))
    if not hist.empty and not fc.empty:
        fig.add_trace(go.Scatter(
            x=[hist["WeekDate"].iloc[-1], fc["WeekDate"].iloc[0]],
            y=[hist[col].iloc[-1], fc["projected_pos"].iloc[0]],
            mode="lines", showlegend=False,
            line=dict(color=C_UPDATED, width=2, dash="dot"), hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=fc["WeekDate"], y=fc["projected_pos"],
        name=f"Updated forecast (from {label})",
        mode="lines+markers", line=dict(color=C_UPDATED, width=3, dash="dash"),
        marker=dict(size=7),
        hovertemplate=_hover_template(f"Updated forecast (from {label})"),
    ))
    if not sys_proj.empty:
        fig.add_trace(go.Scatter(
            x=sys_proj["WeekDate"], y=sys_proj["Projection"],
            name="Original projection", mode="lines+markers",
            line=dict(color=C_ORIGINAL, width=2, dash="dot"), marker=dict(size=5),
            hovertemplate=_hover_template("Original projection"),
        ))
    rr = _revenue_risk_trace(sys_proj["WeekDate"], risk_by_week)
    if rr is not None:
        fig.add_trace(rr)
    return _base_layout(fig, _sku_title(sku, desc), ffw,
                        y_title=f"Units ({label})")


def actuals_vs_plan_chart(sku, desc, source, agg, anchors, date_range=None,
                          weekly=None, prices=None):
    """Per-SKU actuals vs the system's original projection — the Exceptions view's
    model-agnostic chart. By default there is NO updated-forecast line (this view
    runs no model), only the two series the view compares: actual POS or Orders
    sell-through and the original/system projection (the plan of record).

    ``agg`` is a per-SKU-week frame (``SKU, WeekDate, POS, Orders, Projection``)
    already scoped to the one Customer Grouping. When ``date_range`` is given the
    plotted traces are clipped to that window so the Y-axis rescales to fit. When a
    per-SKU ``weekly`` frame (``SKU, WeekDate, projected_pos``) is passed — e.g. after
    "Calculate Optimal Projection" — an orange dashed **Optimized forecast** line and
    the actual→forecast connector are added, matching ``sku_chart``. ``prices`` (a
    SKU->list-price map/Series) adds the "Revenue Risk: $X" row to the
    Original-projection hover.
    """
    lb, lcw, ffw = anchors
    col = "Orders" if source == "Orders" else "POS"
    pm = _as_price_map(prices)

    a = agg[agg["SKU"].astype(str) == str(sku)].sort_values("WeekDate")
    hist = a[(a["WeekDate"] >= lb) & (a["WeekDate"] <= lcw)].dropna(subset=[col])
    # Original projection: straight from the Projection column, from history start
    # through the last week that carries a projection (its own forward horizon —
    # no model frame needed), so the grey line runs the full width of the chart.
    sys_proj = a[a["WeekDate"] >= lb].dropna(subset=["Projection"])

    fc = None
    if weekly is not None and not weekly.empty:
        fc = weekly[weekly["SKU"].astype(str) == str(sku)].copy()
        fc["WeekDate"] = pd.to_datetime(fc["WeekDate"])
        fc = fc.sort_values("WeekDate")

    # Revenue Risk (this SKU's price × plan-gap), shown on the Original line: it
    # appears once per hover and is Actual − plan in history weeks, forecast − plan
    # in forecast weeks (the latter only when an Optimized forecast is overlaid).
    # Empty when no prices supplied.
    rev_orig = _rev_by_week(sys_proj, "Projection", pm)
    risk_parts = [(_rev_by_week(hist, col, pm) - rev_orig).dropna()]
    if fc is not None:
        risk_parts.append((_rev_by_week(fc, "projected_pos", pm) - rev_orig).dropna())
    risk_by_week = pd.concat(risk_parts)

    hist = _clip_to_range(hist, date_range)
    sys_proj = _clip_to_range(sys_proj, date_range)
    if fc is not None:
        fc = _clip_to_range(fc, date_range)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["WeekDate"], y=hist[col], name=f"Actual {source}",
        mode="lines+markers", line=dict(color=C_ACTUAL, width=3),
        marker=dict(size=7), hovertemplate=_hover_template(f"Actual {source}"),
    ))
    if not sys_proj.empty:
        fig.add_trace(go.Scatter(
            x=sys_proj["WeekDate"], y=sys_proj["Projection"],
            name="Original projection", mode="lines+markers",
            line=dict(color=C_ORIGINAL, width=2, dash="dot"), marker=dict(size=5),
            hovertemplate=_hover_template("Original projection"),
        ))
    if fc is not None and not fc.empty:
        if not hist.empty:
            fig.add_trace(go.Scatter(
                x=[hist["WeekDate"].iloc[-1], fc["WeekDate"].iloc[0]],
                y=[hist[col].iloc[-1], fc["projected_pos"].iloc[0]],
                mode="lines", showlegend=False,
                line=dict(color=C_UPDATED, width=2, dash="dot"), hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=fc["WeekDate"], y=fc["projected_pos"], name="Optimized forecast",
            mode="lines+markers", line=dict(color=C_UPDATED, width=3, dash="dash"),
            marker=dict(size=7),
            hovertemplate=_hover_template("Optimized forecast"),
        ))
    rr = _revenue_risk_trace(sys_proj["WeekDate"], risk_by_week)
    if rr is not None:
        fig.add_trace(rr)
    return _base_layout(fig, _sku_title(sku, desc), ffw,
                        y_title=f"Units ({source})")


# --------------------------------------------------------------------------- #
# Customer-group share of one SKU's updated forecast                          #
# --------------------------------------------------------------------------- #
# The SKU-detail section's "where does this SKU's volume come from?" figure. It
# replaced a wide table whose real question was a share — a comparison a donut
# answers at a glance and a table makes the reader compute. A donut is legitimate
# here for the same reason it is on the Historical Summary: the parts genuinely
# sum to a meaningful whole (the SKU's updated forecast, which IS the sum of its
# per-customer fits — see CLAUDE.md's "One grain") and there are few of them.
#
# Every column the table used to carry moves into the hover, one row each, named
# EXACTLY as the column is named everywhere else on the page. Values are formatted
# in Python rather than by a Plotly format spec, so a missing figure reads "—"
# instead of "nan" and money goes through the app's own fmt_dollar (same approach
# as _revenue_risk_trace above).
_BREAKDOWN_HOVER_FIELDS = (
    ("Current Projection Average", "units"),
    ("Projection Difference", "signed"),
    (EIGHT_WK_AVG_COL, "units"),
    (RISK_COL, "money"),
    ("Data Source", "text"),
)


def _fmt_breakdown(value, kind):
    """One hover cell, preformatted. Blank/NaN is always an em dash — a hover row
    reading "nan" looks like a bug, and a zero would be a lie."""
    if kind == "text":
        return str(value) if isinstance(value, str) and value else "—"
    v = pd.to_numeric(value, errors="coerce")
    if pd.isna(v):
        return "—"
    if kind == "money":
        return fmt_dollar(v, signed=True)
    return f"{v:+,.0f}" if kind == "signed" else f"{v:,.0f}"


def _fold_breakdown(rows, col, kind):
    """The folded-tail bucket's value for one column. Numbers add up (every one of
    these is additive across customer groups — that is the one-grain invariant);
    ``Data Source`` collapses to the shared label, or MIXED_SOURCE where the folded
    groups disagree, matching compute.roll_up_summary's rule at SKU grain."""
    if kind == "text":
        vals = {str(v) for v in rows[col].dropna()} if col in rows.columns else set()
        if not vals:
            return None
        return vals.pop() if len(vals) == 1 else MIXED_SOURCE
    return pd.to_numeric(rows[col], errors="coerce").sum(min_count=1)


def customer_share_donut(bd, upd_col="Updated Projection Average", top_n=8):
    """Each customer group's share of ONE SKU's updated forecast.

    ``bd`` is the by-customer frame already sliced to the SKU. Returns ``None``
    when there is nothing positive to draw (no rows, or every group forecast at
    zero), so the caller can say so in words rather than show an empty circle.

    Groups past ``top_n`` fold into a single grey tail slice: the categorical
    palette is eight slots and is never cycled (see ``config.C_CATEGORICAL``). A
    tail of exactly one group keeps its own name — folding a single group into
    "Other" would hide an identity to no purpose — but still takes the tail grey,
    since there is no ninth slot for it.

    Slice colours come from ``categorical_color_map`` over the DRAWN groups, so a
    colour keys on the group's name rather than on its rank — reordering the frame
    repaints nothing. It is built from the drawn groups only, not from all of the
    SKU's: a late-sorting group past slot 8 would hit that helper's backstop, which
    pins every remaining key to the last colour, and two slices would come out
    identical. Across SKUs the assignment is not stable, and cannot be — a
    different SKU sells to a different set of groups.
    """
    group_col = "Customer Grouping"
    if bd is None or bd.empty or upd_col not in bd.columns \
            or group_col not in bd.columns:
        return None
    d = bd.copy()
    d[upd_col] = pd.to_numeric(d[upd_col], errors="coerce")
    # A pie cannot draw a negative or absent value, and a 0% slice is only clutter.
    d = d[d[upd_col] > 0].sort_values(upd_col, ascending=False)
    if d.empty:
        return None

    fields = [(c, k) for c, k in _BREAKDOWN_HOVER_FIELDS if c in d.columns]
    head, tail = d.head(top_n), d.iloc[top_n:]
    cmap = categorical_color_map(head[group_col].astype(str))

    labels, values, custom, colors = [], [], [], []
    for _, r in head.iterrows():
        group = str(r[group_col])
        labels.append(group)
        values.append(float(r[upd_col]))
        custom.append([_fmt_breakdown(r[c], k) for c, k in fields])
        colors.append(cmap[group])
    if not tail.empty:
        labels.append(str(tail[group_col].iloc[0]) if len(tail) == 1
                      else f"Other ({len(tail)} groups)")
        values.append(float(tail[upd_col].sum()))
        custom.append([_fmt_breakdown(_fold_breakdown(tail, c, k), k)
                       for c, k in fields])
        colors.append(C_OTHER)

    rows = "".join(f"<br>{col}: %{{customdata[{i}]}}"
                   for i, (col, _) in enumerate(fields))
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.55, sort=False,
        marker=dict(colors=colors, line=dict(color=C_SEPARATOR, width=2)),
        textinfo="label+percent", textposition="outside",
        # Outside labels sit beyond the pie's own box; without automargin the
        # topmost one is clipped off the figure (see share_donut).
        automargin=True,
        customdata=custom,
        # %{percent} is computed over the drawn slices, which after folding is the
        # true share — it replaces the "Share of Updated Forecast" column exactly.
        hovertemplate=(f"<b>%{{label}}</b><br>Share of updated forecast: "
                       f"%{{percent}}<br>{upd_col}: %{{value:,.0f}} / wk"
                       f"{rows}<extra></extra>"),
    ))
    # The hole carries the SKU total — the same number as the section's "Updated
    # Forecast (avg/wk)" tile, so the tie the caption claims is visible in one look.
    fig.add_annotation(
        text=(f"<b>{sum(values):,.0f}</b>"
              "<br><span style='font-size:12px'>units / wk</span>"),
        showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5,
    )
    fig.update_layout(
        font=dict(family=_CHART_FONT, size=13),
        # No title: the section's "#### Customer group breakdown" heading names it.
        # Real side/top room all the same — automargin can only spend margin that
        # exists, and the outside labels need somewhere to go.
        height=460, margin=dict(l=40, r=40, t=60, b=30),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig
