"""Plotly chart builders and the per-chart date-range control."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_app.config import C_ACTUAL, C_UPDATED, C_ORIGINAL, C_GRID
from dashboard_app.summaries import historical_window


# --------------------------------------------------------------------------- #
# Charts                                                                      #
# --------------------------------------------------------------------------- #
# App font stack — kept in sync with .streamlit/config.toml so chart text matches
# the rest of the UI.
_CHART_FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def _theme_colors():
    """Pick chart text/grid/surface colors for the active Streamlit theme.

    Charts use a transparent background so they sit on the page surface; Plotly's
    default text (~#444) is unreadable on the dark surface, so we set text/grid
    colors explicitly per mode. Degrades to the light palette if the theme context
    isn't available. Trace colors (C_ACTUAL/C_UPDATED/C_ORIGINAL) are unchanged.
    """
    mode = getattr(getattr(st, "context", None), "theme", None)
    mode = getattr(mode, "type", "light") or "light"
    if mode == "dark":
        return dict(
            text="#e5e5e5", muted="#a1a1aa",
            grid="rgba(148,163,184,0.16)", divider="rgba(148,163,184,0.55)",
            hover_bg="rgba(31,31,35,0.95)", hover_border="rgba(148,163,184,0.35)",
        )
    return dict(
        text="#334155", muted="#64748b",
        grid=C_GRID, divider="rgba(100,116,139,0.7)",
        hover_bg="rgba(255,255,255,0.96)", hover_border="rgba(148,163,184,0.35)",
    )


def _base_layout(fig, title, forecast_start, y_title="Units (POS / Orders)"):
    t = _theme_colors()
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=t["text"])),
        font=dict(family=_CHART_FONT, color=t["text"], size=13),
        margin=dict(l=10, r=10, t=80, b=10),
        height=420,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=0.98, x=0,
                    font=dict(size=12, color=t["text"])),
        hoverlabel=dict(bgcolor=t["hover_bg"], bordercolor=t["hover_border"],
                        font=dict(family=_CHART_FONT, size=12, color=t["text"])),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(gridcolor=t["grid"], title=None)
    fig.update_yaxes(gridcolor=t["grid"], rangemode="tozero", title=y_title)
    if forecast_start is not None:
        fig.add_vline(
            x=forecast_start, line_width=1, line_dash="dot",
            line_color=t["divider"],
        )
        fig.add_annotation(
            x=forecast_start, yref="paper", y=0.93, yanchor="bottom",
            text="forecast →", showarrow=False,
            font=dict(size=11, color=t["muted"]),
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


def aggregate_chart(agg, summary, weekly, anchors, view, date_range=None):
    """Total actual demand (historical window) flowing into total forecast (15 wks).

    Historical demand uses each SKU's forecast source (POS or Orders) so the
    actual total is comparable to the forecast total. When date_range is given,
    the plotted traces are clipped to that window so the Y-axis rescales to fit.
    """
    lb, lcw, ffw = anchors

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

    # Clip every plotted trace to the chosen chart window so the Y-axis auto-fits
    # the visible weeks (does not affect the summary/forecast math).
    hist_tot = _clip_to_range(hist_tot, date_range)
    fc_tot = _clip_to_range(fc_tot, date_range)
    sys_tot = _clip_to_range(sys_tot, date_range)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_tot["WeekDate"], y=hist_tot["demand"], name="Actual demand",
        mode="lines+markers", line=dict(color=C_ACTUAL, width=3),
        marker=dict(size=6),
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
        marker=dict(size=6),
    ))
    if not sys_tot.empty:
        fig.add_trace(go.Scatter(
            x=sys_tot["WeekDate"], y=sys_tot["Projection"], name="Original projection",
            mode="lines+markers", line=dict(color=C_ORIGINAL, width=2, dash="dot"),
            marker=dict(size=5),
        ))
    return _base_layout(fig, f"Total weekly demand — {view}", ffw)


def sku_chart(sku, desc, source, agg, weekly, anchors, date_range=None):
    """Per-SKU: actuals (historical window, from its source) + updated forecast + original proj.

    When date_range is given, the plotted traces are clipped to that window so the
    Y-axis rescales to fit the visible weeks.
    """
    lb, lcw, ffw = anchors
    col = "Orders" if source == "Orders" else "POS"

    a = agg[agg["SKU"].astype(str) == str(sku)].sort_values("WeekDate")
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

    # Clip every plotted trace to the chosen chart window so the Y-axis auto-fits.
    hist = _clip_to_range(hist, date_range)
    fc = _clip_to_range(fc, date_range)
    sys_proj = _clip_to_range(sys_proj, date_range)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["WeekDate"], y=hist[col], name=f"Actual {source}",
        mode="lines+markers", line=dict(color=C_ACTUAL, width=3),
        marker=dict(size=7),
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
        name=f"Updated forecast (from {source})",
        mode="lines+markers", line=dict(color=C_UPDATED, width=3, dash="dash"),
        marker=dict(size=7),
    ))
    if not sys_proj.empty:
        fig.add_trace(go.Scatter(
            x=sys_proj["WeekDate"], y=sys_proj["Projection"],
            name="Original projection", mode="lines+markers",
            line=dict(color=C_ORIGINAL, width=2, dash="dot"), marker=dict(size=5),
        ))
    title = f"{sku} — {desc}" if isinstance(desc, str) else str(sku)
    return _base_layout(fig, title, ffw, y_title=f"Units ({source})")
