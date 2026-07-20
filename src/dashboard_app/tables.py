"""Summary-table styling and the Excel-style per-column checklist filters."""
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


# --------------------------------------------------------------------------- #
# Excel-style per-column filters (searchable checkbox dropdown per column)     #
# --------------------------------------------------------------------------- #
_SMALL_LIST = 15   # lists this short show every value up-front; bigger ones search
_CAP = 200         # max checkboxes rendered at once (keeps big columns snappy)


def _cb_key(wkey, opt):
    return f"{wkey}__cb__{opt}"


def _set_many(wkey, options, value):
    """Callback: check/uncheck a batch of options (runs before the rerun renders
    the checkboxes, so setting their session_state keys is safe)."""
    for o in options:
        st.session_state[_cb_key(wkey, o)] = value


def _popover_checklist(label, options, wkey):
    """One column's filter: a popover button opening a searchable checkbox list.

    Values are OR-ed within the column. Small lists render in full; large ones
    show a search box (and only the values you've already checked) until you
    type, so the dropdown stays fast and uncluttered. Each checkbox owns its
    state via a stable session-state key, so a choice survives reruns and being
    searched out of view. Returns the set of checked values (empty = no filter).
    """
    options = list(options)

    def selected():
        return [o for o in options if st.session_state.get(_cb_key(wkey, o), False)]

    n = len(selected())
    with st.popover(f"{label} ({n})" if n else label, use_container_width=True):
        query = st.text_input(
            f"Search {label}", key=f"{wkey}__q", placeholder="Type to search…",
            label_visibility="collapsed",
        ).strip().lower()

        if query:
            matches = [o for o in options if query in str(o).lower()]
        elif len(options) <= _SMALL_LIST:
            matches = options
        else:
            matches = None  # big list, no query: show only what's already picked

        batch = matches if matches is not None else options
        c_all, c_clear = st.columns(2)
        c_all.button("Select all", key=f"{wkey}__all", use_container_width=True,
                     on_click=_set_many, args=(wkey, batch, True))
        c_clear.button("Clear", key=f"{wkey}__clear", use_container_width=True,
                       on_click=_set_many, args=(wkey, batch, False))

        if matches is None:
            for o in sorted(selected(), key=str):
                st.checkbox(str(o), key=_cb_key(wkey, o))
            st.caption(f"Type to search {len(options):,} values.")
        else:
            for o in matches[:_CAP]:
                st.checkbox(str(o), key=_cb_key(wkey, o))
            if len(matches) > _CAP:
                st.caption(f"Showing {_CAP:,} of {len(matches):,} — refine the search.")
            elif not matches:
                st.caption("No matches.")

    return set(selected())


def _clear_all(specs):
    """Callback: uncheck every value across every column filter for this table."""
    for _, wkey, values in specs:
        for o in values.dropna().unique():
            st.session_state[_cb_key(wkey, o)] = False


def filter_table(df, key, P):
    """Render one searchable checkbox filter per column and return the filtered df.

    Excel semantics: OR within a column, AND across columns. The dropdowns
    cross-filter — each column only offers values that still yield rows given the
    others' selections, so an empty combination can't be picked. A derived
    ``Region`` filter (from ``Customer Grouping`` via the pipeline's
    ``region_for_group``) is offered alongside the real columns. ``key``
    namespaces the widgets; each table needs a unique one.
    """
    # (label, widget-key, per-row values) — every column, plus a derived Region.
    specs = [(col, f"{key}::{col}", df[col]) for col in df.columns]
    if "Customer Grouping" in df.columns:
        region = df["Customer Grouping"].map(lambda g: str(P.region_for_group(g)))
        after = [s[0] for s in specs].index("Customer Grouping") + 1
        specs.insert(after, ("Region", f"{key}::__region__", region))
    specs = [s for s in specs if s[2].nunique(dropna=True) > 1]  # worth filtering
    if not specs:
        return df

    # Current selections, read straight from session_state (persist across reruns).
    sel = {wkey: {o for o in values.dropna().unique()
                  if st.session_state.get(_cb_key(wkey, o), False)}
           for _, wkey, values in specs}

    def available(target_wkey):
        """Values of the target column still reachable under every OTHER column's
        selection — Excel's narrowed dropdown."""
        mask = pd.Series(True, index=df.index)
        target_vals = None
        for _, wkey, values in specs:
            if wkey == target_wkey:
                target_vals = values
                continue
            if sel[wkey]:
                mask &= values.isin(sel[wkey])
        return set(target_vals[mask].dropna().unique())

    # Clamp away any checked value that no longer yields rows (can happen after
    # unchecking/re-checking), so an empty combination can't persist.
    for _, wkey, values in specs:
        reachable = available(wkey)
        for o in list(sel[wkey]):
            if o not in reachable:
                st.session_state[_cb_key(wkey, o)] = False
                sel[wkey].discard(o)

    n_active = sum(1 for _, wkey, _ in specs if sel[wkey])
    if n_active:
        st.button(
            f"✕ Clear all filters ({n_active})", key=f"{key}__clearall",
            on_click=_clear_all, args=(specs,),
        )

    # Render the filter buttons in tidy rows so a wide table doesn't overflow.
    per_row = 4
    selections = {}
    for start in range(0, len(specs), per_row):
        row = specs[start:start + per_row]
        cols = st.columns(per_row)
        for col, (label, wkey, values) in zip(cols, row):
            with col:
                selections[wkey] = _popover_checklist(
                    label, sorted(available(wkey), key=str), wkey
                )

    mask = pd.Series(True, index=df.index)
    for _, wkey, values in specs:
        if selections[wkey]:
            mask &= values.isin(selections[wkey])

    out = df[mask]
    if len(out) != len(df):
        st.caption(f"{len(out):,} of {len(df):,} rows match the filters.")
    return out


@st.fragment
def render_filtered_table(df, key, P):
    """Render the per-column filters + the styled table in an isolated fragment.

    The fragment scopes a checkbox click to just this block, so filtering never
    reruns the whole dashboard (charts, agent, KPIs, data-quality tables) — it
    stays quick and clean, like Excel. ``df`` (the unfiltered frame) is captured
    as a fragment arg and reused verbatim on each rerun.
    """
    st.dataframe(
        style_summary(filter_table(df, key, P)),
        width="stretch", hide_index=True,
    )
