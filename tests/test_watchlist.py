"""Unit tests for the shared watchlist store (``dashboard_app.watchlist``).

Focus: the new idempotent ``remove_star`` mutator that backs the Watchlist view's
detail-card "Remove" button and the orphan-pair remover. The module has no other
test coverage, so these also lock in the round-trip behaviour of the shared JSON
store (``save_all`` → ``_read_disk``).

``watchlist`` reads/writes both a disk file (``WATCHLIST_PATH``) and the Streamlit
session cache (``st.session_state``). Neither exists under bare pytest, so the
fixture points the path at a temp file and swaps ``session_state`` for a plain
dict — every function here uses only ``in`` / subscript / ``.get`` / ``.pop``,
which a dict satisfies.
"""
import json
import os

import pytest

pytest.importorskip("streamlit")

from dashboard_app import watchlist as wl


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Isolate the watchlist store: temp JSON path + a plain-dict session cache."""
    path = tmp_path / "watchlist.json"
    monkeypatch.setattr(wl, "WATCHLIST_PATH", str(path))
    monkeypatch.setattr(wl.st, "session_state", {})
    return path


def _seed(pairs_by_list):
    """Persist ``{name: [(sku, cust), ...]}`` through the real writer."""
    wl.save_all({name: set(pairs) for name, pairs in pairs_by_list.items()})


def _on_disk(path):
    """Read the store back as ``{name: set[(sku, cust)]}`` (bypassing the cache)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        name: {(i["sku"], i["customer"]) for i in items}
        for name, items in data["watchlists"].items()
    }


def test_remove_star_removes_present_pair(store):
    _seed({"L": [("SKU1", "AMAZON-DC"), ("SKU2", "WEB-US")]})

    assert wl.remove_star("SKU1", "AMAZON-DC", "L") is True

    remaining = _on_disk(store)["L"]
    assert ("SKU1", "AMAZON-DC") not in remaining
    assert ("SKU2", "WEB-US") in remaining
    # Cache stays in sync with disk (no stale membership).
    assert not wl.is_starred("SKU1", "AMAZON-DC", "L")


def test_remove_star_absent_pair_is_noop(store):
    """Idempotent: removing something not on the list returns False and, crucially,
    never *adds* it (a toggle would have)."""
    _seed({"L": [("SKU2", "WEB-US")]})

    assert wl.remove_star("SKU1", "AMAZON-DC", "L") is False

    disk = _on_disk(store)["L"]
    assert ("SKU1", "AMAZON-DC") not in disk
    assert disk == {("SKU2", "WEB-US")}


def test_remove_star_repeat_stays_removed(store):
    _seed({"L": [("SKU1", "AMAZON-DC")]})

    assert wl.remove_star("SKU1", "AMAZON-DC", "L") is True
    # Second call (e.g. another viewer already removed it) is a harmless no-op.
    assert wl.remove_star("SKU1", "AMAZON-DC", "L") is False
    assert _on_disk(store)["L"] == set()


def test_remove_star_scoped_to_named_list(store):
    _seed({
        "A": [("SKU1", "AMAZON-DC")],
        "B": [("SKU1", "AMAZON-DC")],
    })

    assert wl.remove_star("SKU1", "AMAZON-DC", "A") is True

    disk = _on_disk(store)
    assert disk["A"] == set()
    assert disk["B"] == {("SKU1", "AMAZON-DC")}  # untouched


def test_remove_star_defaults_to_active_list(store):
    _seed({"Only": [("SKU1", "AMAZON-DC")]})
    # No explicit active choice → active_name() falls back to the sole list.
    assert wl.active_name() == "Only"

    assert wl.remove_star("SKU1", "AMAZON-DC") is True
    assert _on_disk(store)["Only"] == set()


def test_remove_star_no_lists_returns_false(store):
    # Nothing seeded → no active list to act on.
    assert wl.remove_star("SKU1", "AMAZON-DC") is False
    assert not os.path.exists(store)


# --------------------------------------------------------------------------- #
# The Watchlist export column                                                  #
# --------------------------------------------------------------------------- #
# The ★ prefix is display-only and never reaches a workbook, so exports carry the
# membership it encodes as its own column. It names EVERY list holding the row, not
# just the active one: an export outlives the session's active-list choice.
import pandas as pd  # noqa: E402


def _rows():
    return pd.DataFrame({
        "SKU": ["SKU1", "SKU2", "SKU3"],
        "Customer Grouping": ["AMAZON-DC", "WEB-US", "AMAZON-DC"],
        "Units": [1, 2, 3],
    })


def test_watchlist_names_lists_every_list_holding_the_row(store):
    _seed({"Alpha": [("SKU1", "AMAZON-DC")],
           "Beta": [("SKU1", "AMAZON-DC"), ("SKU2", "WEB-US")]})

    names = wl.watchlist_names_for(_rows())

    # Sorted and comma-joined, so the column is stable across runs.
    assert list(names) == ["Alpha, Beta", "Beta", ""]


def test_watchlist_names_ignore_the_active_choice(store):
    """Membership is per-pair, not per-session: switching the active list must not
    change what an export says."""
    _seed({"Alpha": [("SKU1", "AMAZON-DC")], "Beta": []})
    wl.set_active("Beta")

    assert wl.watchlist_names_for(_rows()).iloc[0] == "Alpha"


def test_watchlist_names_match_on_the_pair_not_the_sku(store):
    """SKU3 shares SKU1's customer and SKU1's list holds a different pair — a
    SKU-only match would wrongly star it."""
    _seed({"Alpha": [("SKU1", "AMAZON-DC")]})
    assert list(wl.watchlist_names_for(_rows())) == ["Alpha", "", ""]


def test_with_watchlist_column_sits_after_the_key_sku_flag(store):
    _seed({"Alpha": [("SKU1", "AMAZON-DC")]})
    df = _rows()
    df.insert(1, "Key SKU", [True, False, False])

    out = wl.with_watchlist_column(df)

    assert list(out.columns) == [
        "SKU", "Key SKU", wl.WATCHLIST_COL, "Customer Grouping", "Units",
    ]
    assert list(out[wl.WATCHLIST_COL]) == ["Alpha", "", ""]
    assert wl.WATCHLIST_COL not in df.columns  # the source frame is untouched


def test_with_watchlist_column_falls_back_to_after_sku(store):
    _seed({"Alpha": [("SKU1", "AMAZON-DC")]})
    out = wl.with_watchlist_column(_rows())
    assert list(out.columns)[:2] == ["SKU", wl.WATCHLIST_COL]


def test_with_watchlist_column_noops_without_a_customer_column(store):
    """Entries are keyed by (SKU, customer), so a SKU-only frame cannot resolve one."""
    _seed({"Alpha": [("SKU1", "AMAZON-DC")]})
    df = pd.DataFrame({"SKU": ["SKU1"], "Units": [1]})

    out = wl.with_watchlist_column(df)

    assert wl.WATCHLIST_COL in out.columns
    assert list(out[wl.WATCHLIST_COL]) == [""]


def test_with_watchlist_column_noops_without_a_sku_column(store):
    df = pd.DataFrame({"Month": ["2026-01"], "Revenue": [10.0]})
    assert wl.with_watchlist_column(df) is df
