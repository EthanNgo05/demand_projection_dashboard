"""Every cached reader's invalidation argument must actually key its cache.

Streamlit excludes underscore-prefixed parameters from a ``@st.cache_data`` key.
The readers in ``dashboard_app/datasources.py`` were originally written as
``load_raw_from_path(path, _mtime, model_path)`` and documented as busting the
cache when the file changed — but the underscore meant they never did. The
observable bugs: the incremental demand pull rewrites the SAME snapshot path, so
the dashboard kept serving the pre-refresh frame; "Refresh from Plytix" bumped a
nonce that was ignored, so it never re-fetched; the allocation-pairs and on-hand
readers were meant to roll over weekly and never did; and two uploads sharing a
filename returned each other's data.

This suite fails if any of those parameters regains its underscore.
"""

import inspect

import pytest

CACHED_READERS = {
    # function name -> the parameters that must participate in the cache key
    "load_key_skus": ["mtime"],
    "read_plytix_from_path": ["mtime"],
    "read_plytix_from_bytes": ["data"],
    "fetch_plytix_from_url": ["nonce"],
    "load_raw_from_path": ["mtime"],
    "load_raw_from_bytes": ["data"],
    "load_warehouse_from_paths": ["mtimes"],
    "load_allocation_pairs_from_path": ["mtime", "week_key"],
    "load_allocation_pairs_from_bytes": ["data", "week_key"],
    "load_onhand_by_sku_from_path": ["mtime", "week_key"],
    "load_onhand_by_sku_from_bytes": ["data", "week_key"],
    "load_prices_from_path": ["mtime"],
    "load_prices_from_bytes": ["data"],
}


def _wrapped(fn):
    """Unwrap Streamlit's cache decorator to reach the real function."""
    return getattr(fn, "__wrapped__", fn)


@pytest.mark.parametrize("name,expected", sorted(CACHED_READERS.items()))
def test_invalidation_args_are_not_underscored(name, expected):
    from dashboard_app import datasources

    fn = getattr(datasources, name, None)
    assert fn is not None, f"datasources.{name} no longer exists"
    params = list(inspect.signature(_wrapped(fn)).parameters)
    for want in expected:
        assert want in params, (
            f"datasources.{name} should take '{want}' (found {params}). An "
            f"underscore-prefixed '_{want}' is EXCLUDED from the @st.cache_data "
            f"key, so the cache would never invalidate."
        )
    underscored = [p for p in params if p.startswith("_")]
    assert not underscored, (
        f"datasources.{name} has underscore-prefixed parameter(s) {underscored}; "
        "Streamlit drops those from the cache key."
    )


def test_streamlit_really_excludes_underscored_params():
    """Pin the upstream behaviour this whole suite rests on.

    If a future Streamlit starts hashing underscore-prefixed arguments, this test
    fails and the comment block in datasources.py can be revisited — rather than
    the convention being cargo-culted forever.
    """
    import streamlit as st

    seen = []

    @st.cache_data(show_spinner=False)
    def underscored(path, _mtime):
        seen.append(("under", _mtime))
        return _mtime

    @st.cache_data(show_spinner=False)
    def plain(path, mtime):
        seen.append(("plain", mtime))
        return mtime

    assert underscored("f", 1) == 1
    assert underscored("f", 2) == 1, "underscored arg unexpectedly keyed the cache"
    assert [s for s in seen if s[0] == "under"] == [("under", 1)]

    assert plain("f", 1) == 1
    assert plain("f", 2) == 2, "plain arg failed to key the cache"
    assert [s for s in seen if s[0] == "plain"] == [("plain", 1), ("plain", 2)]


def test_mtime_change_reloads_the_raw_frame(tmp_path, monkeypatch):
    """End-to-end: rewriting a snapshot in place must produce the new frame.

    This is the user-visible bug — the dashboard's "🔄 Refresh data" button runs
    the INCREMENTAL pull, which merges into and rewrites the newest snapshot
    under its existing filename. With the old underscore key the page kept
    showing pre-refresh numbers until the file name changed.
    """
    import pandas as pd

    from dashboard_app import datasources

    calls = []

    def fake_read_raw_frame(path):
        calls.append(path)
        return pd.DataFrame({
            "'Demand'[DisplaySKU]": ["A-1"],
            "Description": ["Widget"],
            "Custnmbr": ["AMAZON-DC"],
            "WeekDate": pd.to_datetime(["2026-07-19"]),
            "POS": [float(len(calls))],          # differs per read
            "Sum of Quantity": [1.0],
            "Projection": [1.0],
        })

    monkeypatch.setattr(datasources.data_io, "read_raw_frame", fake_read_raw_frame)
    model_path = next(iter(__import__(
        "dashboard_app.config", fromlist=["MODEL_OPTIONS"]).MODEL_OPTIONS.values()))

    snap = str(tmp_path / "all_demand_projections_2026-07-29.xlsx")
    first = datasources.load_raw_from_path(snap, 1000.0, model_path)
    again = datasources.load_raw_from_path(snap, 1000.0, model_path)
    assert len(calls) == 1, "same mtime should be a cache hit"
    assert first["POS"].iloc[0] == again["POS"].iloc[0] == 1.0

    # Same path, newer mtime -> must re-read.
    fresh = datasources.load_raw_from_path(snap, 2000.0, model_path)
    assert len(calls) == 2, "a newer mtime must invalidate the cache"
    assert fresh["POS"].iloc[0] == 2.0
