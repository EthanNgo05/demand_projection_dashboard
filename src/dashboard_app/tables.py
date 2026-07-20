"""Summary-table styling and the on-screen search filter."""
import pandas as pd
import streamlit as st

from dashboard_app.config import PRICE_COL, RISK_COL, fmt_dollar
from dashboard_app.summaries import resolve_avg_col


# --------------------------------------------------------------------------- #
# Summary table styling                                                       #
# --------------------------------------------------------------------------- #
def style_summary(summary_df):
    """Format numbers and colour the up/down columns (up green / down red)."""
    df = summary_df.copy()
    int_cols = [c for c in [
        "Weeks with data", "Initial Projection Average",
        "Updated Projection Average", "Projection Difference",
    ] if c in df.columns]
    fmt = {c: "{:,.0f}" for c in int_cols}
    avg_col = resolve_avg_col(df)
    if avg_col in df.columns:
        fmt[avg_col] = "{:,.1f}"
    if PRICE_COL in df.columns:
        fmt[PRICE_COL] = lambda v: fmt_dollar(v, decimals=2)
    if RISK_COL in df.columns:
        fmt[RISK_COL] = lambda v: fmt_dollar(v, decimals=0)

    def colour_diff(v):
        if pd.isna(v):
            return ""
        if v > 0:
            return "color:#15803d;font-weight:600"
        if v < 0:
            return "color:#b91c1c;font-weight:600"
        return "color:#64748b"

    sty = df.style.format(fmt, na_rep="—")
    # Colour both the unit difference and the dollar revenue risk by direction.
    diff_cols = [c for c in ["Projection Difference", RISK_COL] if c in df.columns]
    if diff_cols:
        sty = sty.map(colour_diff, subset=diff_cols)
    return sty


def search_filter(df, key, columns=None, placeholder="e.g. a SKU, region, or customer"):
    """Render a search box above a table and return ``df`` filtered to matches.

    Matches the typed query as a case-insensitive substring against every
    column (or just ``columns`` if given). An empty query returns ``df``
    unchanged. Each table needs a unique ``key``. Downloads should stay on the
    unfiltered frame; this only narrows what's shown on screen.
    """
    query = st.text_input("🔍 Search", key=key, placeholder=placeholder).strip()
    if not query:
        return df
    cols = [c for c in (columns or list(df.columns)) if c in df.columns]
    mask = pd.Series(False, index=df.index)
    for c in cols:
        mask |= df[c].astype(str).str.contains(
            query, case=False, na=False, regex=False
        )
    out = df[mask]
    st.caption(f"{len(out):,} of {len(df):,} rows match “{query}”.")
    return out
