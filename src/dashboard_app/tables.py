"""Summary-table styling and the Excel-style add-filter-chip table filters."""
import functools
import re

import pandas as pd
import streamlit as st

from dashboard_app.config import (
    MODEL_USED_COL, PRICE_COL, RISK_COL, fmt_dollar,
    KPI_HELP, KPI_TEXT_FIELDS, ONHAND_COL, TREND_COL, WOS_COL, kpi_sort,
)
from dashboard_app.keyskus import (
    CHIP_LABEL, KEY_SKU_COL, is_key_sku, key_sku_mask, mark_key_sku,
    sku_chip_column_config,
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
# The four fields the two main projections tables (Quick's and Optimized's
# "Summary table by SKU and customer") show as a FIXED bar — always on screen, no
# add/remove step. Every other table keeps the add-filter picker, which is why this
# is a caller-supplied set rather than a change to the default. See ``filter_table``.
FIXED_FILTER_LABELS = ("SKU", "Customer", "Region", KEY_FILTER_LABEL)
# Recognised week/date columns across the summary and data-quality tables.
_DATE_COLS = ["First_WeekDate", "Last_WeekDate",
              "First Projected Week", "Last Projected Week",
              "First Missing Week", "Last Missing Week"]


def _ms_key(wkey):
    """Session-state key holding a checklist/active-in field's multiselect list."""
    return f"{wkey}__ms"


def _sku_label_map(df):
    """``{sku: "SKU — Description"}`` for the SKU filter's dropdown labels.

    Lets the reader search the SKU dropdown by product name as well as by number —
    the affordance Quick's standalone SKU picker had before the fixed bar replaced
    it. The stored VALUE stays the raw SKU; only the label carries the description.
    Empty (so the raw SKU shows) when the frame has no ``Description``.

    No ``.strip()`` here: the warehouse's fixed-width padding comes off at the one
    ingestion boundary (``agent.data_io._clean``), so re-asserting it at each render
    site would just be a second place to keep in step. A description that was nothing
    but padding arrives as ``""`` and falls through the truthiness test below.
    """
    if "SKU" not in df.columns or "Description" not in df.columns:
        return {}
    pairs = df[["SKU", "Description"]].drop_duplicates("SKU")
    return {
        str(s): f"{s} — {d}"
        for s, d in zip(pairs["SKU"].astype(str), pairs["Description"])
        if isinstance(d, str) and d
    }


def _build_fields(df, key, P, fixed=None):
    """Whitelist of filterable fields for ``df``, in a fixed order.

    Only Starred / Key SKU / SKU / Customer / Data Source / Model Used / Region /
    Date range / Active In are ever offered, and only when the underlying column
    exists (and, for checklists, varies). Each field is a dict describing how to read
    options and build a mask; ``kind`` is ``checklist``, ``active_in``, ``date``,
    ``starred`` or ``key_sku``.

    ``fixed`` restricts the build to those labels AND relaxes the "would this
    actually narrow anything?" gates. Those gates exist to keep the *add-filter menu*
    short — a single-valued field there is a menu entry that does nothing. A fixed bar
    makes the opposite promise: the caller has said these controls are always on
    screen, so one vanishing on a single-region view would read as a bug. The one
    gate that survives is Key SKU's ``.any()``: a button that can only ever empty the
    table is worse than no button.
    """
    fields = []
    want = None if fixed is None else set(fixed)

    def add_checklist(label, series, **extra):
        if series is None or (want is not None and label not in want):
            return
        # Fixed bar: build whenever the column exists. Add-filter menu: only when
        # the field would narrow something.
        if want is None and series.nunique(dropna=True) <= 1:
            return
        fields.append({
            "label": label, "wkey": f"{key}::{label}", "kind": "checklist",
            "values": series,
            "options": sorted(series.dropna().unique(), key=str),
            **extra,
        })

    # Watchlist ("Starred") filter — a single toggle that narrows to rows on the
    # active watchlist. Membership is computed live (no column); only offered when
    # it actually varies (some but not all rows starred), so it would narrow.
    starred = starred_mask(df)
    if (want is None or "Starred" in want) and starred is not None \
            and starred.any() and not starred.all():
        fields.append({
            "label": "Starred", "wkey": f"{key}::Starred",
            "kind": "starred", "values": starred,
        })

    # Key SKU filter — the toggle counterpart to the blue "Key" chip. Same shape as
    # Starred (a live mask, no column), and offered on the same terms: only when some
    # but not all rows are key SKUs, so it would narrow. Rolled-up Exceptions grains
    # put "12 SKUs" in the SKU cell and so match nothing — the field drops out there.
    key_mask = key_sku_mask(df)
    if (want is None or KEY_FILTER_LABEL in want) and key_mask.any() \
            and (want is not None or not key_mask.all()):
        fields.append({
            "label": KEY_FILTER_LABEL, "wkey": f"{key}::{KEY_FILTER_LABEL}",
            "kind": "key_sku", "values": key_mask,
        })

    if "SKU" in df.columns:
        add_checklist("SKU", df["SKU"], labels=_sku_label_map(df))

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
    if date_cols and (want is None or "Date range" in want):
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

    if "Active in" in df.columns and (want is None or "Active In" in want):
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


def _multiselect_field(label, options, wkey, labels=None, on_change=None):
    """A filter chip: a multiselect listing the field's currently-reachable values.

    Values are OR-ed within the field. ``st.multiselect`` is natively multi-select
    with built-in type-to-search (fine for the ~700-SKU list) and shows the picks
    as removable tags. Its own session-state key (``_ms_key``) persists the choice
    across reruns. Returns the set of picked values (empty = no filter).

    ``labels`` maps a value to its display string (the SKU field passes
    ``"SKU — Description"``), so the search box matches on either while the stored
    value stays raw — the same value-vs-label split the SKU pickers use.
    ``on_change`` is forwarded to the widget; the fixed bar uses it to drop the
    table's positional row selection when the frame underneath it changes."""
    picked = st.multiselect(
        label, list(options), key=_ms_key(wkey),
        placeholder=f"All {label.lower()} — type to search",
        format_func=(lambda v: labels.get(str(v), str(v))) if labels else str,
        on_change=on_change,
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


def _button_key_sku(label, field, on_change=None):
    """The fixed bar's key-SKU control: a pressable button, not a popover checkbox.

    "See key SKUs by clicking a button" is one click; the popover form is three (open,
    tick, dismiss). Pressed state is the button's own ``type="primary"`` rather than a
    ``✓`` suffix, so it reads as a toggle at a glance.

    Writes the SAME ``__on`` session key ``_popover_key_sku`` does, so ``key_only_active``
    and any state saved under the other form keep working. The app-scoped rerun is there
    for the same reason the popover's is: the bar lives inside an ``@st.fragment``, and
    page-level sections keyed off this filter sit outside it.
    """
    onkey = f"{field['wkey']}__on"
    on = bool(st.session_state.get(onkey, False))
    if st.button(f"⭐ {label}s only", key=f"{field['wkey']}__btn",
                 type="primary" if on else "secondary", width="stretch",
                 help="Show only SKUs on the current key-SKU list"):
        st.session_state[onkey] = not on
        st.session_state[f"{field['wkey']}__applied"] = not on
        if on_change is not None:
            on_change()
        st.rerun(scope="app")
    return on


def _clear_row_selection(key):
    """Callback: drop a selectable table's open detail cards.

    ``render_selectable_table`` keys its selection POSITIONALLY, so row 3 of the
    unfiltered frame and row 3 of a filtered one are different (SKU, customer) pairs —
    leaving the selection alone when a filter changes would silently swap an open
    card's subject. Writing the selection through Session State is the documented
    affordance ``_dismiss_card`` already relies on.
    """
    st.session_state[f"{key}__sel"] = {"selection": {"rows": []}}


def sku_filter_narrowed(key):
    """True when the fixed bar's SKU filter has exactly one SKU picked.

    This is what lets ``render_selectable_table`` decide ``focused`` for itself: it is
    the "the reader has already narrowed to one thing" signal that a caller-side SKU
    dropdown used to supply through ``focus_single``.
    """
    return len(st.session_state.get(_ms_key(f"{key}::SKU"), [])) == 1


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


def filter_table(df, key, P=None, *, fixed=None):
    """Table filtering with Excel semantics: OR within a field, AND across fields.

    Two layouts over one field/mask model:

    * **Add-filter chips** (the default, every table but the two main projections
      ones). An "Add filter" picker activates a field; each active filter shows as a
      row — a multiselect (or a date-range / starred / key-SKU popover) plus a ✕ to
      remove it. Only the whitelist Starred / Key SKU / SKU / Customer / Data Source /
      Model Used / Region / Date range / Active In is offered.
    * **A fixed bar** (``fixed`` = a tuple of labels, normally ``FIXED_FILTER_LABELS``).
      Those fields, and only those, render on first paint with no add/remove step:
      the checklists side by side as dropdowns, the key-SKU toggle as a button
      beneath them.

    Both share the cross-filtering below, which is the property that matters: each
    dropdown offers only the values that still yield rows under every OTHER field's
    current pick, so Region = AU leaves no AMEA-only customer selectable. ``key``
    namespaces the widgets.
    """
    fields = _build_fields(df, key, P, fixed=fixed)
    if not fields:
        return df

    if fixed is not None:
        # Every built field is active, always — that is what "fixed" means. No
        # active-list bookkeeping and no picker.
        active_fields = fields
    else:
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

    selections = {}
    if fixed is not None:
        # Fixed bar: the dropdowns side by side on one row (they are read together —
        # "this SKU, at this customer, in this region"), the key-SKU button on its own
        # line beneath so a click can't be mistaken for a fourth dropdown.
        #
        # Every control drops the table's open detail cards: the selection is keyed by
        # POSITION, so a narrowed frame would leave a card describing a different
        # (SKU, customer) than the row that opened it.
        clear = functools.partial(_clear_row_selection, key)
        drops = [f for f in active_fields if f["kind"] == "checklist"]
        toggles = [f for f in active_fields if f["kind"] != "checklist"]
        # st.columns(0) raises, and a frame carrying none of SKU / Customer / Region
        # (only the key-SKU toggle survived) is a legal caller.
        for col, f in zip(st.columns(len(drops)) if drops else [], drops):
            with col:
                selections[f["label"]] = _multiselect_field(
                    f["label"], sorted(reachable_by[f["label"]], key=str), f["wkey"],
                    labels=f.get("labels"), on_change=clear,
                )
        for f in toggles:
            btn_col, _ = st.columns([1, 3])
            with btn_col:
                if f["kind"] == "key_sku":
                    selections[f["label"]] = _button_key_sku(f["label"], f,
                                                             on_change=clear)
                elif f["kind"] == "starred":
                    selections[f["label"]] = _popover_starred(f["label"], f)
                else:
                    selections[f["label"]] = _popover_daterange(f["label"], f)
    else:
        # Render active filters one per row — [ control ][✕] — so each multiselect has
        # room to show every picked value as a tag (date/starred keep compact popovers).
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
                        f["label"], sorted(reachable_by[f["label"]], key=str),
                        f["wkey"]
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


def _render_kpi_tiles(row, cols, card_key, extra=None, deltas=None, identity=None,
                      per_row=4):
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

    ``identity`` is the same tuple shape as ``extra`` returns, but already built and
    the same for every view — the Watchlist / Key SKU flags ``_render_row_detail``
    derives. It is a list rather than a callback because there is nothing per-view to
    bind: every card answers those two questions the same way.

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
    if identity:
        tiles.extend(identity)
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
    Every card carries the row's two identity flags — watchlist membership and key-SKU
    status — as a ★ / blue "Key" badge on the title AND as a pair of tiles, so the card
    answers both questions whichever way the reader looks.
    A ✕ button (top-right) closes the card by deselecting its table row via
    ``sel_key``/``close_pos`` (so the row's checkbox clears too, matching an
    in-table deselect), letting the user close it without scrolling back up. Callers
    that render a card with no table to deselect from (``focus_single``) pass neither,
    which suppresses the button.
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
    # Mark the card title with a ★ when this row is on the active watchlist, and a
    # blue "Key" badge when it is a key SKU — the card's echo of the two decorations
    # the tables carry (watchlist.mark_starred_sku's ★ prefix left of the SKU,
    # keyskus' chip right of it), so an opened card reads the same way as its row.
    #
    # Both also become explicit KPI tiles below, because a badge can only say YES:
    # the absence of a ★ is indistinguishable from a card that never draws one, and
    # "is this a key item?" is a question a planner asks of every SKU, not only of
    # the ones that happen to be flagged. Skipped whole at a rolled-up grain (no SKU
    # column), where neither flag has a subject to be true of.
    cust = row.get("Customer Grouping") or row.get("Customer")
    has_sku = "SKU" in row.index
    starred = has_sku and (str(row.get("SKU", "")), str(cust)) in active_pairs()
    key_sku = has_sku and is_key_sku(row.get("SKU", ""))
    star = STAR_PREFIX if starred else ""
    identity = [
        ("Watchlist", f"{STAR_PREFIX}Starred" if starred else "Not starred", None,
         KPI_HELP.get("Watchlist"), "text"),
        (KEY_SKU_COL, "Yes" if key_sku else "No", None,
         KPI_HELP.get(KEY_SKU_COL), "text"),
    ] if has_sku else None
    # Keyed so the scoped CSS in render_selectable_table can tint + space each card.
    card_key = re.sub(r"[^0-9A-Za-z_-]+", "-", f"detailcard-{key_base}-{close_label}")
    with st.container(border=True, key=card_key):
        title_c, x_col = st.columns([12, 1])
        title_txt = f"{star}{row.get(title_col, '')}"
        title = f"**{title_txt}** — {desc}" if desc else f"**{title_txt}**"
        if key_sku:
            title = f"{title} :blue-badge[{CHIP_LABEL}]"
        title_c.markdown(title)
        if sel_key is not None and close_pos is not None:
            x_col.button(
                "✕", key=f"{key_base}__close__{close_label}", help="Close this card",
                on_click=_dismiss_card, args=(sel_key, close_pos),
            )
        _render_kpi_tiles(row, detail_cols, card_key, extra=extra_kpis,
                          deltas=kpi_deltas, identity=identity)
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
                            kpi_deltas=None, focus_single=False, fixed=None):
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

    ``fixed`` is forwarded to ``filter_table`` — pass ``FIXED_FILTER_LABELS`` for the
    always-on SKU / Customer / Region / Key-SKU bar the two main projections tables use.

    ``focus_single`` says "the caller has already narrowed to one thing": when it is
    set AND the filters leave exactly one row, the table is not rendered at all and
    that row's card opens on its own. A one-row table is not a choice — it is a click
    the reader has to make to see something they have already asked for, and its five
    condensed columns are all repeated by the card's tiles. Under ``fixed`` the same
    signal is DERIVED — one SKU picked in the bar means the same thing a caller-side
    SKU dropdown used to mean — so those callers pass nothing. It is off by default,
    so a table that happens to filter down to one row still behaves normally.
    """
    filtered = filter_table(df, key, P, fixed=fixed)
    # Deliberately NOT `focus_single and len(df) == 1`: the filter chips run after the
    # caller's picker, so it is `filtered` that decides. A chip that empties the frame
    # leaves focused False and falls through to the ordinary empty-table path.
    narrowed = focus_single or (fixed is not None and sku_filter_narrowed(key))
    focused = narrowed and len(filtered) == 1
    display_cols = [c for c in condensed_cols if c in filtered.columns]
    if focused:
        rows = [0]
    else:
        display_df = mark_starred_sku(filtered[display_cols])
        display_df, sku_values = mark_key_sku(display_df)
        cfg = {**(column_config or {}),
               **(sku_chip_column_config(sku_values) if sku_values else {})}
        event = st.dataframe(
            style_summary(display_df) if style else display_df,
            width="stretch", hide_index=True, column_config=cfg or None,
            on_select="rerun", selection_mode="multi-row", key=f"{key}__sel",
        )
        # Positional indices persist across reruns, so drop any that a filter has
        # since pushed out of range (sorted so cards stack in table order, not click
        # order).
        rows = event.selection.rows if event and event.selection else []
        rows = sorted(r for r in rows if r < len(filtered))

    # The dataframe selection is the single source of truth: each selected row gets
    # a card, and a card's ✕ deselects its row (see _dismiss_card) so closing a card
    # and unchecking its row are the same action. Under `focused` there is no
    # dataframe and so no selection to be the source of anything: the card is passed
    # no sel_key, which also suppresses its ✕ (closing it would leave the reader with
    # no way back — the table that would re-open it is gone).
    sel_key = None if focused else f"{key}__sel"
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
                               close_pos=None if focused else r,
                               card_cols=detail_cols,
                               row_action=row_action, title_col=title_col,
                               extra_kpis=extra_kpis, kpi_deltas=kpi_deltas)
    else:
        st.caption("Select one or more rows to see their full details.")
