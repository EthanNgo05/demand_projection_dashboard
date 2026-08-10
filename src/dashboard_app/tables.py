"""Summary-table styling and the Excel-style add-filter-chip table filters."""
import re

import pandas as pd
import streamlit as st

from dashboard_app.config import (
    MODEL_USED_COL, PRICE_COL, RISK_COL, fmt_dollar,
    KPI_HELP, KPI_TEXT_FIELDS, ONHAND_COL, TREND_COL, WOS_COL, kpi_sort,
)
from dashboard_app.keyskus import (
    key_sku_mask, mark_key_sku, sku_chip_column_config,
)
from dashboard_app.watchlist import (
    STAR_PREFIX, active_pairs, mark_starred_sku, starred_mask,
)


# --------------------------------------------------------------------------- #
# Summary table styling                                                       #
# --------------------------------------------------------------------------- #
def style_summary(summary_df):
    """Format numbers and colour the up/down columns (up green / down red)."""
    df = summary_df.copy()
    int_cols = [c for c in [
        "Weeks with data", "Current Projection Average",
        "Updated Projection Average", "Projection Difference",
    ] if c in df.columns]
    fmt = {c: "{:,.0f}" for c in int_cols}
    # Format every descriptive-average column to one decimal. Single-group views
    # carry one; the Optimal Projections combined view carries two (All-Time
    # and 8-Week POS/Orders Average). The Exceptions view stores its 8-week
    # average as a whole number (integer dtype) so it ties out with Projection
    # Difference / Revenue Risk — render those without a spurious decimal.
    #
    # Substring, not suffix: the view-total table qualifies its column
    # "... POS/Orders Average (model fit)" (see dashboard._render_quick_view), and
    # that one must format identically. No other column carries the phrase.
    for c in df.columns:
        if "POS/Orders Average" in c and pd.api.types.is_numeric_dtype(df[c]):
            fmt[c] = "{:,.0f}" if pd.api.types.is_integer_dtype(df[c]) else "{:,.1f}"
    # The Exceptions view's signed percent deviation: two decimals when the value
    # has a fractional part, a whole number when it doesn't, always suffixed "%".
    if "% Deviation" in df.columns:
        def _fmt_pct(v):
            if pd.isna(v):
                return "—"
            return f"{int(v):,}%" if v == int(v) else f"{v:,.2f}%"
        fmt["% Deviation"] = _fmt_pct
    if PRICE_COL in df.columns:
        fmt[PRICE_COL] = lambda v: fmt_dollar(v, decimals=2)
    if RISK_COL in df.columns:
        fmt[RISK_COL] = lambda v: fmt_dollar(v, decimals=0)

    # Mid-tone green/red that keep adequate contrast on BOTH light and dark
    # surfaces (Styler emits fixed inline colors that Streamlit does not recolor,
    # so a single theme-neutral pair is used rather than a per-theme guess). Colour
    # is cosmetic — the sign in the formatted number already carries the meaning.
    def colour_diff(v):
        if pd.isna(v):
            return ""
        if v > 0:
            return "color:#16a34a;font-weight:600"
        if v < 0:
            return "color:#dc2626;font-weight:600"
        return "color:#6b7280"

    sty = df.style.format(fmt, na_rep="—")
    # Colour both the unit difference and the dollar revenue risk by direction.
    diff_cols = [c for c in ["Projection Difference", RISK_COL] if c in df.columns]
    if diff_cols:
        sty = sty.map(colour_diff, subset=diff_cols)
    return sty


# --------------------------------------------------------------------------- #
# Excel-style add-filter chips (multiselect / date range per field)            #
# --------------------------------------------------------------------------- #
# Only these fields are ever offered as filters — no continuous-number columns
# (e.g. Revenue Risk) that would make a useless hundreds-long value list.
_ADD_PLACEHOLDER = "➕ Add filter…"
# Label of the key-SKU filter chip. Public so page-level sections can ask whether a
# table has been narrowed to key items (see ``key_only_active``).
KEY_FILTER_LABEL = "Key SKU"
# Recognised week/date columns across the summary and data-quality tables.
_DATE_COLS = ["First_WeekDate", "Last_WeekDate",
              "First Projected Week", "Last Projected Week",
              "First Missing Week", "Last Missing Week"]


def _ms_key(wkey):
    """Session-state key holding a checklist/active-in field's multiselect list."""
    return f"{wkey}__ms"


def _build_fields(df, key, P):
    """Whitelist of filterable fields for ``df``, in a fixed order.

    Only Starred / Key SKU / SKU / Customer / Data Source / Model Used / Region /
    Date range / Active In are ever offered, and only when the underlying column
    exists (and, for checklists, varies). Each field is a dict describing how to read
    options and build a mask; ``kind`` is ``checklist``, ``active_in``, ``date``,
    ``starred`` or ``key_sku``.
    """
    fields = []

    def add_checklist(label, series):
        if series is not None and series.nunique(dropna=True) > 1:
            fields.append({
                "label": label, "wkey": f"{key}::{label}", "kind": "checklist",
                "values": series,
                "options": sorted(series.dropna().unique(), key=str),
            })

    # Watchlist ("Starred") filter — a single toggle that narrows to rows on the
    # active watchlist. Membership is computed live (no column); only offered when
    # it actually varies (some but not all rows starred), so it would narrow.
    starred = starred_mask(df)
    if starred is not None and starred.any() and not starred.all():
        fields.append({
            "label": "Starred", "wkey": f"{key}::Starred",
            "kind": "starred", "values": starred,
        })

    # Key SKU filter — the toggle counterpart to the blue "Key" chip. Same shape as
    # Starred (a live mask, no column), and offered on the same terms: only when some
    # but not all rows are key SKUs, so it would narrow. Rolled-up Exceptions grains
    # put "12 SKUs" in the SKU cell and so match nothing — the field drops out there.
    key_mask = key_sku_mask(df)
    if key_mask.any() and not key_mask.all():
        fields.append({
            "label": KEY_FILTER_LABEL, "wkey": f"{key}::{KEY_FILTER_LABEL}",
            "kind": "key_sku", "values": key_mask,
        })

    if "SKU" in df.columns:
        add_checklist("SKU", df["SKU"])

    cust_col = next((c for c in ("Customer Grouping", "Customer")
                     if c in df.columns), None)
    if cust_col:
        add_checklist("Customer", df[cust_col])

    if "Data Source" in df.columns:
        add_checklist("Data Source", df["Data Source"])

    # Model Used: only present on the Optimized Projections table, where each
    # customer group carries its own backtest-winning model.
    if MODEL_USED_COL in df.columns:
        add_checklist("Model Used", df[MODEL_USED_COL])

    # Region: an explicit Region/Region Code column if present, else derived from
    # the customer grouping via the loaded pipeline (summary/KPI tables).
    if "Region" in df.columns:
        add_checklist("Region", df["Region"])
    elif "Region Code" in df.columns:
        add_checklist("Region", df["Region Code"])
    elif P is not None and "Customer Grouping" in df.columns:
        add_checklist("Region",
                      df["Customer Grouping"].map(lambda g: str(P.region_for_group(g))))

    date_cols = [c for c in _DATE_COLS if c in df.columns]
    if date_cols:
        parsed = {c: pd.to_datetime(df[c], errors="coerce") for c in date_cols}
        allv = pd.concat(parsed.values())
        lo, hi = allv.min(), allv.max()
        if pd.notna(lo) and pd.notna(hi) and lo.date() != hi.date():
            firsts = [c for c in date_cols if "First" in c] or date_cols
            lasts = [c for c in date_cols if "Last" in c] or date_cols
            fields.append({
                "label": "Date range", "wkey": f"{key}::Date", "kind": "date",
                "first": pd.concat([parsed[c] for c in firsts], axis=1).min(axis=1),
                "last": pd.concat([parsed[c] for c in lasts], axis=1).max(axis=1),
                "min_d": lo.date(), "max_d": hi.date(),
            })

    if "Active in" in df.columns:
        codes = sorted({x.strip() for s in df["Active in"].dropna()
                        for x in str(s).split(",") if x.strip()})
        if len(codes) > 1:
            fields.append({
                "label": "Active In", "wkey": f"{key}::ActiveIn",
                "kind": "active_in", "values": df["Active in"], "options": codes,
            })

    return fields


def _selection(field):
    """Read a field's current selection from session_state (persists across
    reruns): a set of checked values/codes, or a ``(start, end)`` date tuple."""
    wkey = field["wkey"]
    if field["kind"] == "date":
        cur = st.session_state.get(f"{wkey}__di")
        if isinstance(cur, (tuple, list)) and len(cur) == 2:
            return (cur[0], cur[1])
        return None
    if field["kind"] in ("starred", "key_sku"):
        return bool(st.session_state.get(f"{wkey}__on", False))
    # checklist / active_in: the multiselect stores its picked values as a list.
    return set(st.session_state.get(_ms_key(wkey), []))


def _field_mask(df, field, selection):
    """Boolean row mask for one field's selection (empty selection → all True)."""
    kind = field["kind"]
    if kind in ("starred", "key_sku"):
        if not selection:
            return pd.Series(True, index=df.index)
        return field["values"]
    if kind == "checklist":
        if not selection:
            return pd.Series(True, index=df.index)
        return field["values"].isin(selection)
    if kind == "active_in":
        if not selection:
            return pd.Series(True, index=df.index)
        sel = set(selection)
        return field["values"].apply(
            lambda s: bool({x.strip() for x in str(s).split(",")} & sel)
        )
    # date: keep rows whose [first, last] interval overlaps the picked window.
    if not selection:
        return pd.Series(True, index=df.index)
    start, end = selection
    if start == field["min_d"] and end == field["max_d"]:
        return pd.Series(True, index=df.index)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    overlap = (field["last"] >= start_ts) & (field["first"] <= end_ts)
    return overlap.fillna(True)  # keep rows with unknown dates


def _multiselect_field(label, options, wkey):
    """A filter chip: a multiselect listing the field's currently-reachable values.

    Values are OR-ed within the field. ``st.multiselect`` is natively multi-select
    with built-in type-to-search (fine for the ~700-SKU list) and shows the picks
    as removable tags. Its own session-state key (``_ms_key``) persists the choice
    across reruns. Returns the set of picked values (empty = no filter)."""
    picked = st.multiselect(
        label, list(options), key=_ms_key(wkey),
        placeholder=f"All {label.lower()} — type to search",
    )
    return set(picked)


def _popover_daterange(label, field):
    """A date-range filter chip. Returns the picked ``(start, end)`` or None."""
    wkey, dikey = field["wkey"], f"{field['wkey']}__di"
    lo, hi = field["min_d"], field["max_d"]
    cur = st.session_state.get(dikey)
    narrowed = isinstance(cur, (tuple, list)) and len(cur) == 2 and tuple(cur) != (lo, hi)
    with st.popover(f"{label} ✓" if narrowed else label, use_container_width=True):
        kwargs = {} if dikey in st.session_state else {"value": (lo, hi)}
        val = st.date_input(label, min_value=lo, max_value=hi, key=dikey,
                            label_visibility="collapsed", **kwargs)
    if isinstance(val, (tuple, list)) and len(val) == 2:
        return (val[0], val[1])
    return None  # mid-selection (only a start picked) → treat as no filter


def _popover_starred(label, field):
    """A watchlist filter chip: one checkbox that narrows to starred rows."""
    onkey = f"{field['wkey']}__on"
    on = bool(st.session_state.get(onkey, False))
    with st.popover(f"{label} ✓" if on else label, use_container_width=True):
        st.checkbox("On the active watchlist", key=onkey)
    return bool(st.session_state.get(onkey, False))


def _popover_key_sku(label, field):
    """A key-SKU filter chip: one checkbox that narrows to key items.

    Flipping it also forces a FULL app rerun. The chips live inside an
    ``@st.fragment`` (render_filtered_table / render_selectable_table), so by default
    a click reruns only that table — but page-level sections keyed off this filter
    (the Exceptions view's on-plan and not-in-demand-data lists, via
    ``key_only_active``) sit outside the fragment and would otherwise lag one
    interaction behind the table they describe.
    """
    onkey = f"{field['wkey']}__on"
    appliedkey = f"{field['wkey']}__applied"
    on = bool(st.session_state.get(onkey, False))
    with st.popover(f"{label} ✓" if on else label, use_container_width=True):
        st.checkbox("Key items only", key=onkey)
    on = bool(st.session_state.get(onkey, False))
    if st.session_state.get(appliedkey) != on:
        st.session_state[appliedkey] = on
        st.rerun(scope="app")
    return on


def _add_filter(key, active_key):
    """Callback: activate the field chosen in the "Add filter" selectbox."""
    choice = st.session_state.get(f"{key}__add")
    if choice and choice != _ADD_PLACEHOLDER:
        active = list(st.session_state.get(active_key, []))
        if choice not in active:
            active.append(choice)
        st.session_state[active_key] = active
    st.session_state[f"{key}__add"] = _ADD_PLACEHOLDER  # reset for the next add


def _remove_filter(active_key, label, wkey, kind):
    """Callback: drop a filter chip and clear whatever it had selected."""
    st.session_state[active_key] = [
        l for l in st.session_state.get(active_key, []) if l != label
    ]
    if kind == "date":
        st.session_state.pop(f"{wkey}__di", None)
    elif kind in ("starred", "key_sku"):
        st.session_state.pop(f"{wkey}__on", None)
        st.session_state.pop(f"{wkey}__applied", None)
    else:  # checklist / active_in
        st.session_state.pop(_ms_key(wkey), None)


def filter_table(df, key, P=None):
    """Add-filter-chip filtering: start clean, add only the fields you want.

    An "Add filter" picker activates a field; each active filter shows as a row
    — a multiselect (or a date-range / starred / key-SKU popover) plus a ✕ to remove
    it. Excel semantics (OR within a field, AND across fields) with cross-filtering,
    so the active multiselects only offer values that still yield rows. Only the
    whitelist Starred / Key SKU / SKU / Customer / Data Source / Model Used / Region /
    Date range / Active In is offered. ``key`` namespaces the widgets.
    """
    fields = _build_fields(df, key, P)
    if not fields:
        return df
    by_label = {f["label"]: f for f in fields}
    labels_in_order = [f["label"] for f in fields]

    active_key = f"{key}__active"
    active = [l for l in st.session_state.get(active_key, []) if l in by_label]
    st.session_state[active_key] = active  # sanitised (columns change per view)
    active_fields = [by_label[l] for l in active]

    # "Add filter" picker — only fields not already active.
    addable = [l for l in labels_in_order if l not in active]
    add_col, _ = st.columns([1, 2])
    with add_col:
        if addable:
            st.selectbox(
                "Add filter", [_ADD_PLACEHOLDER] + addable, key=f"{key}__add",
                label_visibility="collapsed", on_change=_add_filter,
                args=(key, active_key),
            )
        else:
            st.caption("All filters added.")

    # Current selections, read from session_state (persist across reruns).
    sel = {f["label"]: _selection(f) for f in active_fields}

    def available(target):
        """Options of the target field still reachable under every OTHER active
        filter — Excel's narrowed dropdown. (Date fields have no option list.)"""
        mask = pd.Series(True, index=df.index)
        for f in active_fields:
            if f["label"] == target["label"]:
                continue
            mask &= _field_mask(df, f, sel[f["label"]])
        if target["kind"] == "checklist":
            return set(target["values"][mask].dropna().unique())
        if target["kind"] == "active_in":
            codes = set()
            for s in target["values"][mask].dropna():
                codes |= {x.strip() for x in str(s).split(",") if x.strip()}
            return codes
        return None

    # Clamp away any picked value that no longer yields rows, so an empty
    # combination can't persist (date fields aren't clamped). Rewriting the
    # multiselect's stored list here — before the widget instantiates — is legal
    # and keeps the value ⊆ its options (which Streamlit requires).
    reachable_by = {}
    for f in active_fields:
        if f["kind"] in ("date", "starred", "key_sku"):
            continue
        reachable = available(f)
        reachable_by[f["label"]] = reachable
        kept = [o for o in sel[f["label"]] if o in reachable]
        if len(kept) != len(sel[f["label"]]):
            st.session_state[_ms_key(f["wkey"])] = kept
            sel[f["label"]] = set(kept)

    # Render active filters one per row — [ control ][✕] — so each multiselect has
    # room to show every picked value as a tag (date/starred keep compact popovers).
    selections = {}
    for f in active_fields:
        ctrl_col, x_col = st.columns([12, 1], vertical_alignment="bottom")
        with ctrl_col:
            if f["kind"] == "date":
                selections[f["label"]] = _popover_daterange(f["label"], f)
            elif f["kind"] == "starred":
                selections[f["label"]] = _popover_starred(f["label"], f)
            elif f["kind"] == "key_sku":
                selections[f["label"]] = _popover_key_sku(f["label"], f)
            else:
                selections[f["label"]] = _multiselect_field(
                    f["label"], sorted(reachable_by[f["label"]], key=str), f["wkey"]
                )
        with x_col:
            st.button("✕", key=f"{f['wkey']}__rm",
                      help=f"Remove the {f['label']} filter",
                      on_click=_remove_filter,
                      args=(active_key, f["label"], f["wkey"], f["kind"]))

    mask = pd.Series(True, index=df.index)
    for f in active_fields:
        mask &= _field_mask(df, f, selections[f["label"]])

    out = df[mask]
    if len(out) != len(df):
        st.caption(f"{len(out):,} of {len(df):,} rows match the filters.")
    return out


def key_only_active(*keys):
    """True when the Key-SKU filter chip is switched on for any of ``keys``.

    Lets a view render key-SKU-specific sections (the Exceptions view's on-plan and
    not-in-demand-data lists) in step with a table's filter, rather than adding a
    second page-level control that could disagree with the chip.
    """
    return any(
        bool(st.session_state.get(f"{k}::{KEY_FILTER_LABEL}__on", False)) for k in keys
    )


@st.fragment
def render_filtered_table(df, key, P=None, *, style=True, column_config=None):
    """Render the add-filter chips + the table in an isolated fragment.

    The fragment scopes a filter click to just this block, so filtering never
    reruns the whole dashboard — it stays quick and clean, like Excel. ``style``
    applies the summary formatting/colouring (summary & KPI tables); pass
    ``style=False`` for the data-quality tables, which render plainly. ``df``
    (the unfiltered frame) is captured as a fragment arg and reused each rerun.
    ``column_config`` is forwarded to ``st.dataframe`` so callers can pin
    per-column widths (e.g. widen a free-text column so its text isn't clipped).
    Rows on the active watchlist are marked by a ``★`` prefix on their SKU cell, and
    key SKUs by a blue "Key" chip to its right (both display only — filtering runs on
    the undecorated frame).
    """
    filtered = filter_table(df, key, P)
    # ★ first: the prefix is part of the SKU string the chip config lists as an option.
    display = mark_starred_sku(filtered)
    display, sku_values = mark_key_sku(display)
    cfg = {**(column_config or {}),
           **(sku_chip_column_config(sku_values) if sku_values else {})}
    st.dataframe(
        style_summary(display) if style else display,
        width="stretch", hide_index=True, column_config=cfg or None,
    )


# --------------------------------------------------------------------------- #
# Click-to-expand exception tables (condensed rows + a detail card on select)  #
# --------------------------------------------------------------------------- #
def _fmt_detail_value(col, val):
    """Format one cell for the detail card, matching how style_summary renders
    the same column in the table (dollars, signed percent, comma'd integers)."""
    if pd.isna(val):
        return "—"
    if col == PRICE_COL:
        return fmt_dollar(val, decimals=2)
    if col == RISK_COL:
        return fmt_dollar(val, decimals=0)
    if col == "% Deviation":
        return f"{int(val):,}%" if val == int(val) else f"{val:,.2f}%"
    if col == TREND_COL:
        # Signed, so "is this growing or dying" reads without hunting for a colour.
        return f"{val:+,.1f}%"
    if col == ONHAND_COL:
        return f"{val:,.0f}"
    if col in (WOS_COL, "Container Impact"):
        return f"{val:,.1f}"
    int_cols = {
        "Weeks with data", "Current Projection Average",
        "Updated Projection Average", "Projection Difference",
    }
    if col in int_cols:
        return f"{val:,.0f}"
    if col.endswith("POS/Orders Average") and isinstance(val, (int, float)):
        return f"{val:,.0f}" if float(val).is_integer() else f"{val:,.1f}"
    return str(val)


def _tile_value(col, val):
    """``_fmt_detail_value`` plus the one KPI whose blank carries meaning.

    A blank trend is not "no data" — it means the earlier 8-week window had no
    sales, so there is no baseline to measure against. "New" says that; "—" would
    read as missing. Every other column keeps the shared "—" for NaN.
    """
    if col == TREND_COL and pd.isna(val):
        return "New"
    return _fmt_detail_value(col, val)


def _render_kpi_tiles(row, cols, card_key, extra=None, deltas=None, per_row=4):
    """Render a detail card's KPIs as the same shaded tiles the KPI row uses.

    Every card in the app funnels through here, which is the point: KPIs used to be
    flat ``**Label**\\n\\nvalue`` markdown in this card, shaded ``st.metric`` tiles
    beside the projections chart, and a hand-rolled coloured ``<span>`` in the
    Exceptions card — three treatments for one kind of thing. Emitting ``st.metric``
    means the tiles inherit the existing ``[data-testid="stMetric"]`` styling in
    dashboard.py's stylesheet, so they match *by construction* rather than by a
    second copy of the CSS that can drift.

    ``cols`` is the field set; ``config.kpi_sort`` decides the order, so a field
    lands in the same position no matter which view opened the card.

    Two optional hooks, both bound per view via ``functools.partial``:

    * ``extra``: ``callable(row) -> [(label, value, delta, help, kind), ...]`` for KPIs
      that are derived rather than columns (e.g. Projected Revenue = price ×
      forecast), appended after the column-backed tiles.
    * ``deltas``: ``{column: callable(row) -> str | None}`` to hang a secondary
      figure under an EXISTING tile — the small green/red line ``st.metric`` renders
      below the value. Used for the percentage under Projection Difference, which has
      to modify a tile rather than add one.

    Each tile row is wrapped in a keyed container so the stylesheet can equalise
    heights within the row — otherwise a value that wraps to three lines (a long
    model name) leaves its neighbours short and the grid reads ragged.
    """
    deltas = deltas or {}
    tiles = []
    for c in cols:
        fn = deltas.get(c)
        tiles.append((
            c, _tile_value(c, row[c]), fn(row) if fn else None, KPI_HELP.get(c),
            "text" if c in KPI_TEXT_FIELDS else "stat",
        ))
    if extra is not None:
        tiles.extend(extra(row))
    if not tiles:
        return
    # Order AFTER folding in the extras, not before: a derived tile is a KPI like any
    # other and belongs in its canonical slot. Sorting only the column-backed tiles
    # left Projected Revenue stranded at the end of the grid, away from List Price and
    # Revenue Risk — the two figures it is read against.
    order = {label: i for i, label in enumerate(kpi_sort([t[0] for t in tiles]))}
    tiles.sort(key=lambda t: order[t[0]])

    for start in range(0, len(tiles), per_row):
        chunk = tiles[start:start + per_row]
        # Pad the final row so a lone trailing tile stays column-width instead of
        # stretching across the card.
        with st.container(key=f"kpitiles-{card_key}-{start}"):
            slots = st.columns(per_row)
            for slot, (label, value, delta, help_txt, kind) in zip(slots, chunk):
                if kind == "text":
                    # Keyed wrapper -> CSS can size identity values as captions
                    # rather than headlines ("Holt-Winters (triple) exponential
                    # smoothing" is not a number and must not look like one).
                    slug = re.sub(r"[^0-9A-Za-z]+", "-", str(label)).strip("-")
                    with slot.container(key=f"kpitile-text-{card_key}-{start}-{slug}"):
                        st.metric(label, value, help=help_txt)
                else:
                    slot.metric(label, value, delta=delta, help=help_txt)


def _dismiss_card(sel_key, pos):
    """Callback: deselect this card's table row so the card closes AND its
    table checkbox clears together (runs before the rerun, so both are gone on
    the next render). Streamlit 1.58+ lets us set st.dataframe row selection
    through Session State (see the DataframeState docstring in
    streamlit/elements/arrow.py)."""
    state = st.session_state.get(sel_key)
    current = []
    if state and "selection" in state:
        current = list(state["selection"].get("rows", []))
    st.session_state[sel_key] = {"selection": {"rows": [r for r in current if r != pos]}}


def _render_row_detail(row, shown, detail_chart=None, key_base=None,
                       sel_key=None, close_label=None, close_pos=None, card_cols=None,
                       row_action=None, title_col="SKU", extra_kpis=None,
                       kpi_deltas=None):
    """Render a row's full detail in a bordered card beneath the table.

    Layout is the same for every view: the card's KPIs as one shaded tile grid
    (``_render_kpi_tiles``), then the chart full-width beneath. The projections card
    used to split into chart-left / metrics-right, which meant its KPIs lived in two
    places at once — the grid here AND that column — with ``Data Source`` in both.

    ``card_cols`` (if given) is the set of fields to show in the card, decoupled from
    the frame's full column set (which the condensed table, sorting, and the Excel
    download still need); ``config.kpi_sort`` orders them, so the list is a set, not
    a sequence. Without it the card falls back to listing EVERY non-hidden field, so
    other callers keep their behaviour. ``shown`` is kept for signature stability but
    no longer hides columns. ``extra_kpis`` is passed straight to
    ``_render_kpi_tiles`` for KPIs derived rather than read off the row. When
    ``detail_chart`` is given it is called with ``(row, key_base)`` to draw a chart
    below the tiles.
    A ✕ button (top-right) closes the card by deselecting its table row via
    ``sel_key``/``close_pos`` (so the row's checkbox clears too, matching an
    in-table deselect), letting the user close it without scrolling back up.
    ``row_action`` (if given) is a ``{label, help, danger, callback}`` dict rendered
    as a button at the bottom of the card; clicking it calls ``callback(row)`` (the
    callback owns any rerun — e.g. by opening a confirmation dialog)."""
    # The title-bearing column (``title_col``, default SKU) and Description live in
    # the card title, so they're dropped from the grid; the remaining columns start
    # with the first stat. At a rolled-up grain ``title_col`` is the group key
    # (e.g. Customer Grouping / Region) so it isn't repeated as a grid cell.
    _title_drop = {"SKU", "Description", title_col}
    if card_cols is not None:
        detail_cols = [
            c for c in card_cols
            if c in row.index and c not in _title_drop
            and not str(c).startswith("_")
        ]
    else:
        detail_cols = [
            c for c in row.index
            if c not in _title_drop and not str(c).startswith("_")
        ]
    # "Note" reads as a sentence, not a stat — peel it out of the 3-per-row grid and
    # render it full-width at the bottom (only when it carries text), so the grid's
    # last row stays a clean pair (e.g. List Price · Weeks with data).
    note_val = row["Note"] if "Note" in detail_cols else None
    if "Note" in detail_cols:
        detail_cols = [c for c in detail_cols if c != "Note"]
    show_note = note_val is not None and not pd.isna(note_val) and str(note_val) != ""

    desc = row["Description"] if "Description" in row.index else ""
    # Mark the card title with a ★ when this row is on the active watchlist, so an
    # opened card (in any view) shows membership without a dedicated column.
    cust = row.get("Customer Grouping") or row.get("Customer")
    star = STAR_PREFIX if (str(row.get("SKU", "")), str(cust)) in active_pairs() else ""
    # Keyed so the scoped CSS in render_selectable_table can tint + space each card.
    card_key = re.sub(r"[^0-9A-Za-z_-]+", "-", f"detailcard-{key_base}-{close_label}")
    with st.container(border=True, key=card_key):
        title_c, x_col = st.columns([12, 1])
        title_txt = f"{star}{row.get(title_col, '')}"
        title = f"**{title_txt}** — {desc}" if desc else f"**{title_txt}**"
        title_c.markdown(title)
        if sel_key is not None and close_pos is not None:
            x_col.button(
                "✕", key=f"{key_base}__close__{close_label}", help="Close this card",
                on_click=_dismiss_card, args=(sel_key, close_pos),
            )
        _render_kpi_tiles(row, detail_cols, card_key, extra=extra_kpis,
                          deltas=kpi_deltas)
        if show_note:
            st.markdown(f"**Note**\n\n{_fmt_detail_value('Note', note_val)}")
        if detail_chart is not None:
            detail_chart(row, key_base)
        if row_action is not None:
            btn_key = re.sub(r"[^0-9A-Za-z_-]+", "-",
                             f"{key_base}__rowaction__{close_label}")
            if row_action.get("danger"):
                # Scope the destructive red styling to this button's key wrapper,
                # matching the delete-watchlist confirm button elsewhere.
                st.markdown(
                    f"<style>.st-key-{btn_key} button{{background-color:#dc2626;"
                    "border-color:#dc2626;color:#fff;}"
                    f".st-key-{btn_key} button:hover{{background-color:#b91c1c;"
                    "border-color:#b91c1c;color:#fff;}</style>",
                    unsafe_allow_html=True,
                )
            if st.button(row_action["label"], key=btn_key,
                         help=row_action.get("help"), width="content"):
                row_action["callback"](row)


@st.fragment
def render_selectable_table(df, key, P=None, *, condensed_cols, style=True,
                            column_config=None, detail_chart=None, detail_cols=None,
                            row_action=None, title_col="SKU", extra_kpis=None,
                            kpi_deltas=None):
    """Like render_filtered_table, but shows only ``condensed_cols`` per row and
    reveals the full row in a detail card below when a row is clicked.

    The Exceptions tables use this so each row stays scannable (SKU / Customer /
    projection / revenue risk) while every other field is one click away. Select
    multiple rows to stack their detail cards side by side. Filtering still runs on
    the FULL frame, so the filter chips are unaffected; the detail lookup also uses
    the full frame, mapping each selected positional index back to ``filtered``.
    Wrapped in a fragment so a row click reruns only this block. ``detail_chart``,
    if given, is a ``(row, key_base)`` callback that draws a chart inside each card.
    Rows on the active watchlist are marked by a ``★`` prefix on their SKU cell, and
    key SKUs by a blue "Key" chip to its right (both display only — filtering and the
    detail lookup use the undecorated frame, so the decorated SKU never reaches either).
    ``row_action`` is forwarded to each detail card (see ``_render_row_detail``) so
    callers can add a per-row button (e.g. the watchlist's "Remove" affordance), and
    ``extra_kpis`` / ``kpi_deltas`` likewise for KPI tiles a view derives rather than
    reads off the row.
    """
    filtered = filter_table(df, key, P)
    display_cols = [c for c in condensed_cols if c in filtered.columns]
    display_df = mark_starred_sku(filtered[display_cols])
    display_df, sku_values = mark_key_sku(display_df)
    cfg = {**(column_config or {}),
           **(sku_chip_column_config(sku_values) if sku_values else {})}
    event = st.dataframe(
        style_summary(display_df) if style else display_df,
        width="stretch", hide_index=True, column_config=cfg or None,
        on_select="rerun", selection_mode="multi-row", key=f"{key}__sel",
    )
    # Positional indices persist across reruns, so drop any that a filter has since
    # pushed out of range (sorted so cards stack in table order, not click order).
    rows = event.selection.rows if event and event.selection else []
    rows = sorted(r for r in rows if r < len(filtered))

    # The dataframe selection is the single source of truth: each selected row gets
    # a card, and a card's ✕ deselects its row (see _dismiss_card) so closing a card
    # and unchecking its row are the same action.
    sel_key = f"{key}__sel"
    if rows:
        # Tint + space each detail card so stacked cards read as separate panels
        # rather than one long card. Scoped to the cards' keyed wrappers; the
        # translucent gray works on either theme.
        st.markdown(
            "<style>"
            '[class*="st-key-detailcard-"]{'
            "background-color:rgba(130,140,160,0.06);margin-bottom:0.9rem;}"
            '[class*="st-key-detailcard-"] '
            '[data-testid="stVerticalBlockBorderWrapper"],'
            '[data-testid="stVerticalBlockBorderWrapper"][class*="st-key-detailcard-"]'
            "{border-color:rgba(130,140,160,0.35);}"
            "</style>",
            unsafe_allow_html=True,
        )
        for r in rows:
            _render_row_detail(filtered.iloc[r], shown=display_cols,
                               detail_chart=detail_chart, key_base=key,
                               sel_key=sel_key, close_label=filtered.index[r],
                               close_pos=r, card_cols=detail_cols,
                               row_action=row_action, title_col=title_col,
                               extra_kpis=extra_kpis, kpi_deltas=kpi_deltas)
    else:
        st.caption("Select one or more rows to see their full details.")
