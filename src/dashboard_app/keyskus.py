"""Key-SKU membership + the blue "Key" chip used across every table.

A key SKU is one the planning team flags as important: ``sql/key_skus.sql``
(``KeyItem = 'Yes'`` in ``dmd.week_of_supply_parameters``) → ``extract_key_skus.py``
→ ``raw_inputs/key_skus/key_skus_<date>.xlsx`` → ``data_io.read_key_skus``. Nothing
here touches the database; membership is read from the newest snapshot on disk.

Key-SKU status used to be a *place* — the Exceptions view's "Key SKUs" tab. It is now
an *attribute*: every display table marks its key rows with a blue "Key" chip inside
the SKU cell, every table can be filtered to key items only (``tables._build_fields``),
and every export carries a ``Key SKU`` true/false column instead of the chip.

The chip sits to the RIGHT of the SKU name, mirroring the watchlist's ``★ `` prefix on
the left, so a starred key SKU reads ★ → SKU → Key.

The split matters: the chip is a display-only decoration in a throwaway copy (same
contract as ``watchlist.mark_starred_sku``'s ★ prefix — it must never reach filtering,
the detail-card lookup, or a workbook), while ``with_key_sku_column`` writes real
booleans into the frames that are handed to the Excel writers.
"""
import os

import pandas as pd
import streamlit as st

from dashboard_app.datasources import discover_key_skus_file, load_key_skus
from dashboard_app.watchlist import STAR_PREFIX

# The blue chip rendered to the RIGHT of the SKU, inside the SKU cell — the
# counterpart to the watchlist's ``★ `` prefix on the left.
CHIP_LABEL = "Key"

# Colour for the SKU's own tag. MultiselectColumn renders every list item as a tag
# and its colour list has no "leave this one unstyled" entry, so the SKU gets an
# explicitly neutral translucent gray — the same one the detail cards use in
# tables.py, chosen because it reads as unstyled on both light and dark themes.
SKU_TAG_NEUTRAL = "rgba(130,140,160,0.15)"

# Boolean column written into exports in place of the chip.
KEY_SKU_COL = "Key SKU"


def current_key_skus():
    """The newest key-SKU snapshot as a frozenset, or an empty frozenset.

    Empty when no snapshot has been extracted yet — every caller then degrades to
    "nothing is a key SKU" (no chips, no filter field offered, all-False exports)
    rather than erroring. The Exceptions view surfaces the fetch button for that case.
    """
    path = discover_key_skus_file()
    if not path:
        return frozenset()
    return load_key_skus(path, os.path.getmtime(path))


def _normalised_skus(series):
    """SKU cells reduced to the raw SKU string used in the key-SKU list.

    Table cells carry display decoration that the snapshot does not: the watchlist's
    ``★ `` prefix (watchlist.mark_starred_sku) and the trailing ``*`` the data-quality
    tables stamp on. Both are stripped so a decorated row still matches.
    """
    return (series.astype(str).str.strip()
            .str.removeprefix(STAR_PREFIX).str.rstrip("*").str.strip())


def _normalise_sku(value):
    """Scalar twin of ``_normalised_skus`` — same strip/prefix/suffix rules.

    Kept as a separate function rather than routing one value through a Series:
    the detail card asks about ONE SKU per render, and building a pandas object to
    answer that is more machinery than the question deserves. The two must agree,
    so any change to the decoration rules belongs in both.
    """
    return str(value).strip().removeprefix(STAR_PREFIX).rstrip("*").strip()


def is_key_sku(sku, key_skus=None):
    """Whether one SKU string is a key SKU (``★ `` / trailing ``*`` tolerated).

    The frame-wide question is ``key_sku_mask``; this is the single-row form the
    detail cards need, and it exists so callers stop reaching for the raw
    ``current_key_skus()`` frozenset — membership against that set is only correct
    for an UNDECORATED SKU, and a card's row can carry either form.
    """
    if key_skus is None:
        key_skus = current_key_skus()
    if not key_skus:
        return False
    return _normalise_sku(sku) in key_skus


def key_sku_mask(df, key_skus=None):
    """Boolean Series (index-aligned to ``df``) marking rows whose SKU is a key SKU.

    All-False when ``df`` has no ``SKU`` column or the key-SKU list is empty/absent.
    Rolled-up Exceptions grains put a count label (``"12 SKUs"``) in the SKU cell,
    which matches nothing — so those tables get no chips and no filter field.
    """
    if df is None or "SKU" not in getattr(df, "columns", []):
        return pd.Series(False, index=getattr(df, "index", None), dtype=bool)
    if key_skus is None:
        key_skus = current_key_skus()
    if not key_skus:
        return pd.Series(False, index=df.index, dtype=bool)
    return _normalised_skus(df["SKU"]).isin(key_skus)


def mark_key_sku(df, key_skus=None):
    """``(frame, sku_values)`` — a copy of ``df`` whose SKU cells become tag lists.

    A key SKU's cell becomes ``[sku, "Key"]`` so the blue chip renders to the RIGHT
    of the SKU name; every other cell becomes ``[sku]``. ``sku_values`` is the sorted
    list of SKU cell values present, which ``sku_chip_column_config`` needs as its
    option list — colours attach to options, so a value missing from that list cannot
    be styled.

    Returns ``(df, None)`` unchanged when no row is a key SKU, leaving the SKU column
    as ordinary text: no table grows tag styling it has no chip to justify.

    Display only. The caller keeps filtering, the detail-card row lookup and every
    export on the undecorated frame — a tag list would break all three.
    """
    mask = key_sku_mask(df, key_skus)
    if not bool(mask.any()):
        return df, None
    out = df.copy()
    values = out["SKU"].astype(str)
    out["SKU"] = [[v, CHIP_LABEL] if m else [v] for v, m in zip(values, mask)]
    return out, sorted(set(values))


def sku_chip_column_config(sku_values, label=None):
    """``column_config`` rendering the SKU column as [neutral SKU tag][blue Key chip].

    ``st.dataframe`` escapes HTML and ``TextColumn`` has no Markdown support, so a
    tag list is the only way to get a real coloured chip beside the SKU — which is
    also why the SKU itself renders as a tag rather than plain text.

    Colours are passed as a list positionally aligned to ``options``: every SKU
    neutral, ``"Key"`` blue. Streamlit 1.58's ``MultiselectColumn`` claims to accept
    per-option dicts but re-wraps them (``{"value": {"value": ...}}``), so the
    parallel-list form is the only one that actually works.

    Accepted cost: as a tag column the SKU no longer sorts or searches as plain text
    in the dataframe toolbar. The filter chips (``tables.filter_table``) run on the
    undecorated frame and are unaffected.
    """
    sku_values = list(sku_values)
    return {
        "SKU": st.column_config.MultiselectColumn(
            label, options=[*sku_values, CHIP_LABEL],
            color=[*([SKU_TAG_NEUTRAL] * len(sku_values)), "blue"],
            disabled=True,
        )
    }


def with_key_sku_column(df, key_skus=None):
    """Copy of ``df`` with a boolean ``Key SKU`` column after ``SKU`` — for exports.

    A no-op returning ``df`` unchanged when it has no ``SKU`` column (or already
    carries the column), so export call sites can wrap unconditionally. Unlike the
    chip this is always added, including when nothing matches: a column of ``FALSE``
    is a real answer, an absent column is an ambiguous one.
    """
    if df is None or "SKU" not in getattr(df, "columns", []):
        return df
    if KEY_SKU_COL in df.columns:
        return df
    out = df.copy()
    out.insert(out.columns.get_loc("SKU") + 1, KEY_SKU_COL,
               key_sku_mask(df, key_skus).to_numpy())
    return out
