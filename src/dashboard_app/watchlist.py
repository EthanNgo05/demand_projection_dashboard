"""Named SKU/Customer watchlists + the ★ marker used across tables.

Multiple named watchlists live in one shared ``outputs/watchlist.json`` (the app
has no per-user identity, so the lists are shared). Each list holds
``(SKU, Customer Grouping)`` pairs. Which list is *active* is per-session UI state
(session_state), so two viewers can focus different lists without fighting over
the file. Reads are best-effort (missing/corrupt → empty) mirroring
``compute._load_agent_summary``; writes are atomic (temp file + ``os.replace``)
mirroring the snapshot writes in ``data_io``.
"""
import json
import os
import tempfile

import pandas as pd
import streamlit as st

from dashboard_app.config import REPO_ROOT

WATCHLIST_PATH = os.path.join(REPO_ROOT, "outputs", "watchlist.json")

# Prefix stamped onto a starred row's SKU cell (display only — never persisted or
# used in lookups). Replaces the old separate ★ column.
STAR_PREFIX = "★ "

# Export column carrying what the ★ encodes, since the prefix itself can't survive
# into a workbook (see with_watchlist_column).
WATCHLIST_COL = "Watchlist"

# Name used when migrating a legacy single-list file, and the default first list.
DEFAULT_NAME = "My Watchlist"

# Session-state keys: the loaded {name: set(pairs)} cache (invalidated on write)
# and the per-session active-list name.
_CACHE_KEY = "_watchlist"
_ACTIVE_KEY = "_watchlist_active"


# --------------------------------------------------------------------------- #
# Disk I/O + migration                                                        #
# --------------------------------------------------------------------------- #
def _pairs_from_list(items):
    """Coerce a JSON list of ``{"sku","customer"}`` dicts into a pair set."""
    pairs = set()
    for item in items or []:
        try:
            pairs.add((str(item["sku"]), str(item["customer"])))
        except (KeyError, TypeError):
            continue
    return pairs


def _read_disk():
    """Read the file into ``{name: set(pairs)}``; empty dict on any error.

    Migrates two legacy shapes: a bare list (the original single watchlist) and
    an unwrapped ``{name: [...]}`` dict both load into the named-list form.
    """
    try:
        with open(WATCHLIST_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if isinstance(data, list):  # legacy single flat list
        return {DEFAULT_NAME: _pairs_from_list(data)}
    if isinstance(data, dict):
        lists = data.get("watchlists") if "watchlists" in data else data
        if isinstance(lists, dict):
            return {str(name): _pairs_from_list(items)
                    for name, items in lists.items()}
    return {}


def load_all():
    """All watchlists as ``{name: set[(sku, customer)]}``; cached in session."""
    if _CACHE_KEY not in st.session_state:
        st.session_state[_CACHE_KEY] = _read_disk()
    return st.session_state[_CACHE_KEY]


def save_all(mapping):
    """Atomically persist ``{name: set(pairs)}`` and refresh the session cache."""
    mapping = {str(name): set(pairs) for name, pairs in mapping.items()}
    os.makedirs(os.path.dirname(WATCHLIST_PATH), exist_ok=True)
    payload = {
        "watchlists": {
            name: [{"sku": sku, "customer": cust}
                   for sku, cust in sorted(pairs, key=lambda p: (p[1], p[0]))]
            for name, pairs in sorted(mapping.items())
        }
    }
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(WATCHLIST_PATH),
                               prefix=".watchlist_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, WATCHLIST_PATH)
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    st.session_state[_CACHE_KEY] = mapping


# --------------------------------------------------------------------------- #
# List management                                                             #
# --------------------------------------------------------------------------- #
def list_names():
    """Sorted watchlist names."""
    return sorted(load_all().keys())


def get_watchlist(name):
    """The pair set for ``name`` (empty set if it doesn't exist)."""
    return set(load_all().get(name, set()))


def create_watchlist(name):
    """Create an empty watchlist ``name`` (no-op if it already exists), make it
    active, and return it. Blank/whitespace names are ignored."""
    name = (name or "").strip()
    if not name:
        return None
    lists = dict(load_all())
    if name not in lists:
        lists[name] = set()
        save_all(lists)
    set_active(name)
    return name


def rename_watchlist(old, new):
    """Rename ``old`` → ``new`` (preserving its pairs), keeping it active."""
    new = (new or "").strip()
    lists = dict(load_all())
    if not new or old not in lists or new in lists:
        return False
    lists[new] = lists.pop(old)
    save_all(lists)
    if active_name() == old:
        set_active(new)
    return True


def delete_watchlist(name):
    """Delete ``name``. If it was active, the active pointer resets to the first
    remaining list (or None)."""
    lists = dict(load_all())
    if name not in lists:
        return False
    lists.pop(name)
    save_all(lists)
    if active_name() == name:
        st.session_state.pop(_ACTIVE_KEY, None)
    return True


# --------------------------------------------------------------------------- #
# Active-list selection (per session)                                         #
# --------------------------------------------------------------------------- #
def active_name():
    """Currently-active watchlist name: the session choice if it still exists,
    else the first list (sorted), else None."""
    names = list_names()
    cur = st.session_state.get(_ACTIVE_KEY)
    if cur in names:
        return cur
    return names[0] if names else None


def set_active(name):
    """Set the active watchlist for this session."""
    st.session_state[_ACTIVE_KEY] = name


def active_pairs():
    """Pair set of the active watchlist (empty when there is none)."""
    name = active_name()
    return get_watchlist(name) if name is not None else set()


# --------------------------------------------------------------------------- #
# Membership + toggle                                                         #
# --------------------------------------------------------------------------- #
def is_starred(sku, customer, name=None):
    """Whether ``(sku, customer)`` is on ``name`` (default: the active list)."""
    if name is None:
        name = active_name()
    if name is None:
        return False
    return (str(sku), str(customer)) in get_watchlist(name)


def toggle_star(sku, customer, name=None):
    """Add/remove ``(sku, customer)`` on ``name`` (default: the active list).

    Returns the new starred state, or None if there is no active list to act on.
    """
    if name is None:
        name = active_name()
    if name is None:
        return None
    lists = dict(load_all())
    pairs = set(lists.get(name, set()))
    key = (str(sku), str(customer))
    if key in pairs:
        pairs.discard(key)
        now_starred = False
    else:
        pairs.add(key)
        now_starred = True
    lists[name] = pairs
    save_all(lists)
    return now_starred


def remove_star(sku, customer, name=None):
    """Remove ``(sku, customer)`` from ``name`` (default: the active list).

    Idempotent (unlike ``toggle_star``): a no-op returning False if the pair
    isn't on the list, so a stale removal — e.g. another viewer deleting the same
    pair from this shared list between render and click — can never re-add it.
    Returns True when it actually removed the pair.
    """
    if name is None:
        name = active_name()
    if name is None:
        return False
    lists = dict(load_all())
    pairs = set(lists.get(name, set()))
    key = (str(sku), str(customer))
    if key not in pairs:
        return False
    pairs.discard(key)
    lists[name] = pairs
    save_all(lists)
    return True


# --------------------------------------------------------------------------- #
# Table marker helpers                                                        #
# --------------------------------------------------------------------------- #
def customer_col(df):
    """Name of the customer column in ``df`` (grouping preferred), or None.

    Mirrors the detection in ``tables._build_fields`` so the star grain matches
    whatever a given table exposes.
    """
    return next((c for c in ("Customer Grouping", "Customer") if c in df.columns),
                None)


def starred_mask(df, pairs=None):
    """Boolean Series (index-aligned to ``df``) marking rows whose
    ``(SKU, customer)`` is on ``pairs`` (default: the active watchlist).

    All-False when ``df`` lacks SKU or a customer column, or ``pairs`` is empty.
    """
    cust = customer_col(df)
    if df is None or "SKU" not in df.columns or cust is None:
        return pd.Series(False, index=getattr(df, "index", None))
    if pairs is None:
        pairs = active_pairs()
    if not pairs:
        return pd.Series(False, index=df.index)
    return pd.Series(
        [(str(s), str(c)) in pairs
         for s, c in zip(df["SKU"].astype(str), df[cust].astype(str))],
        index=df.index,
    )


def watchlist_names_for(df):
    """Series of comma-joined watchlist names containing each row's (SKU, customer).

    EVERY list, not just the active one: on screen a row carries a single ★ meaning
    "on the list you are looking at", but an export outlives that session choice, so
    naming every list it belongs to is the only self-describing answer.

    Empty string where a row is on no list, and for every row when ``df`` lacks a SKU
    or customer column (watchlist entries are keyed by the pair, so neither alone can
    resolve one).
    """
    cust = customer_col(df)
    if df is None or "SKU" not in getattr(df, "columns", []) or cust is None:
        return pd.Series("", index=getattr(df, "index", None), dtype="object")
    lists = load_all()
    if not lists:
        return pd.Series("", index=df.index, dtype="object")
    # One pair -> names lookup, so a wide watchlist costs one pass rather than one
    # membership test per (row, list).
    names_by_pair = {}
    for name in sorted(lists):
        for pair in lists[name]:
            names_by_pair.setdefault(pair, []).append(name)
    return pd.Series(
        [", ".join(names_by_pair.get((str(s), str(c)), ()))
         for s, c in zip(df["SKU"].astype(str), df[cust].astype(str))],
        index=df.index, dtype="object",
    )


def with_watchlist_column(df):
    """Copy of ``df`` with ``Watchlist`` inserted after the SKU block — for exports.

    The ★ prefix is display-only and never reaches a workbook, so the membership it
    encodes has to travel as its own column. Placed after ``Key SKU`` when that column
    is present, keeping the flags grouped beside the SKU they describe. A no-op
    returning ``df`` unchanged when there is no ``SKU`` column or it already carries
    the column, so export call sites can wrap unconditionally.
    """
    if df is None or "SKU" not in getattr(df, "columns", []):
        return df
    if WATCHLIST_COL in df.columns:
        return df
    out = df.copy()
    # keyskus.KEY_SKU_COL by value, not by import: keyskus imports this module for
    # STAR_PREFIX, so importing it back would cycle. Landing before Key SKU (when a
    # caller adds the columns in the other order) would only reorder them, not break.
    after = "Key SKU" if "Key SKU" in out.columns else "SKU"
    out.insert(out.columns.get_loc(after) + 1, WATCHLIST_COL, watchlist_names_for(df))
    return out


def mark_starred_sku(df, pairs=None):
    """Return a copy of ``df`` whose ``SKU`` cells are prefixed ``"★ "`` for
    starred rows (display only). No-op copy when nothing is starred / no SKU col."""
    if df is None or "SKU" not in getattr(df, "columns", []):
        return df
    mask = starred_mask(df, pairs)
    if not mask.any():
        return df
    out = df.copy()
    out.loc[mask, "SKU"] = STAR_PREFIX + out.loc[mask, "SKU"].astype(str)
    return out
