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
