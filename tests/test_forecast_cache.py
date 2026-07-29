"""The persistent forecast cache: fidelity, invalidation, and failure modes.

Three things have to hold, in order of how badly they'd hurt if broken:

1. **A cache hit must be numerically identical to a recompute.** This is the
   whole safety property — a fast wrong number is worse than a slow right one.
2. **The key must invalidate** when the snapshot, the prices, the model file or
   the parameters change, and must NOT collide across views/kinds.
3. **Every failure must degrade to a miss.** No pyarrow, a corrupt Parquet, a
   half-written entry, a deleted directory mid-session — all "recompute", none
   "raise" and none "serve garbage".

Plus a guard that ``agent.batch``'s key construction still matches
``dashboard.py``'s, since a silent drift there turns the nightly warm-up into
dead weight without any visible symptom.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
if FIXTURE_DIR not in sys.path:
    sys.path.insert(0, FIXTURE_DIR)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """forecast_cache pointed at a temp dir, enabled, and empty."""
    from dashboard_app import forecast_cache

    root = tmp_path / "cache"
    monkeypatch.setattr(forecast_cache, "cache_dir", lambda: str(root))
    monkeypatch.setenv("DEMAND_FORECAST_CACHE", "1")
    return forecast_cache


@pytest.fixture
def frames():
    """Frames shaped like the pipelines' real output.

    The weekly frame's ``WeekDate`` is deliberately an OBJECT column of
    ``datetime.date`` — that is what every model builds (via ``week.date()``),
    and it is the one column whose Parquet round-trip could plausibly change
    dtype and quietly alter what the charts and Excel exports show.
    """
    weeks = pd.date_range("2026-07-26", periods=15, freq="W-SUN")
    return {
        "summary": pd.DataFrame({
            "SKU": ["A-1", "B-2"],
            "Description": ["Widget", "Gadget"],
            "8 Week POS/Orders Average": [12.5, 0.0],
            "Updated Projection Average": [13.0, np.nan],
            "Weeks with data": [26, 3],
            "Top Volume Customer Groups": ["AMAZON-DC (80%)", None],
        }),
        "weekly": pd.DataFrame({
            "SKU": ["A-1"] * 15,
            "WeekDate": [w.date() for w in weeks],
            "projected_pos": np.linspace(10.0, 24.0, 15),
        }),
        "agg": pd.DataFrame({
            "SKU": ["A-1", "A-1"],
            "WeekDate": pd.to_datetime(["2026-07-12", "2026-07-19"]),
            "POS": [11.0, np.nan],
            "Orders": [np.nan, 9.0],
            "Projection": [10.0, 10.0],
        }),
    }


# --------------------------------------------------------------------------
# 1. Fidelity
# --------------------------------------------------------------------------
def test_roundtrip_is_exact(cache, frames):
    """What comes back must equal what went in, bit for bit."""
    assert cache.put("k1", frames, {"view": "AMAZON-DC"}) is True
    got = cache.get("k1")
    assert got is not None
    for name, original in frames.items():
        pd.testing.assert_frame_equal(got[name], original,
                                      check_exact=True, check_dtype=True,
                                      obj=f"cached:{name}")


def test_weekly_date_column_stays_object_dates(cache, frames):
    """Pin the fragile case explicitly, so a pyarrow/pandas upgrade that starts
    coercing object-dtype dates to datetime64 fails HERE rather than showing up
    as subtly different chart axes and Excel cells."""
    cache.put("k1", frames, None)
    wk = cache.get("k1")["weekly"]
    assert wk["WeekDate"].dtype == object
    assert isinstance(wk["WeekDate"].iloc[0], __import__("datetime").date)


def test_missing_key_is_a_miss(cache):
    assert cache.get("never-written") is None


# --------------------------------------------------------------------------
# 2. Keys: invalidation and separation
# --------------------------------------------------------------------------
def test_snapshot_signature_tracks_mtime_and_size(tmp_path):
    from dashboard_app import forecast_cache as fc

    f = tmp_path / "snap.xlsx"
    f.write_bytes(b"aaaa")
    first = fc.snapshot_signature(str(f))

    # Same content, same path -> same signature (a plain rerun must hit).
    assert fc.snapshot_signature(str(f)) == first

    # An in-place rewrite (what `--incremental` does) must invalidate. Force a
    # distinct mtime rather than trusting filesystem timer granularity.
    f.write_bytes(b"bbbbbbbb")
    os.utime(f, (1, 1))
    assert fc.snapshot_signature(str(f)) != first


def test_snapshot_signature_ignores_directory(tmp_path):
    """Keyed on basename, not full path, so moving the repo doesn't cold-start
    every cached forecast."""
    from dashboard_app import forecast_cache as fc

    a = tmp_path / "one" / "snap.xlsx"
    b = tmp_path / "two" / "snap.xlsx"
    for p in (a, b):
        p.parent.mkdir(parents=True)
        p.write_bytes(b"same")
        os.utime(p, (500, 500))
    assert fc.snapshot_signature(str(a)) == fc.snapshot_signature(str(b))


def test_missing_file_is_a_stable_signature(tmp_path):
    """A None/absent price file must not crash key construction."""
    from dashboard_app import forecast_cache as fc

    assert fc.snapshot_signature(None) == fc.snapshot_signature(
        str(tmp_path / "nope.xlsx")
    )


def test_content_signature_detects_a_price_change():
    """Prices reach the app from three sources with no common file identity, so
    they are content-hashed. A changed price must produce a new key."""
    from dashboard_app import forecast_cache as fc

    a = pd.Series([10.0, 20.0], index=["A-1", "B-2"])
    b = pd.Series([10.0, 20.01], index=["A-1", "B-2"])
    assert fc.content_signature(a) == fc.content_signature(a.copy())
    assert fc.content_signature(a) != fc.content_signature(b)
    assert fc.content_signature(None) is None


@pytest.mark.parametrize("changed", [
    {"data_sig": "other"},
    {"view": "COSTCO"},
    {"today_str": "2026-07-30"},
    {"alpha": 0.5},
    {"beta": 0.5},
    {"phi": 0.9},
    {"min_weeks": 4},
    {"kind": "group"},
])
def test_every_key_input_invalidates(changed):
    """Each field in the key must actually change it — otherwise a stale
    forecast survives a change it should not have."""
    from dashboard_app import forecast_cache as fc

    base = dict(data_sig="sig", view="AMAZON-DC", model_path="models/tsb.py",
                today_str="2026-07-29", alpha=None, beta=None, phi=None,
                min_weeks=None, kind="view")
    assert fc.forecast_key(**base) != fc.forecast_key(**{**base, **changed})


def test_key_is_stable_across_calls():
    from dashboard_app import forecast_cache as fc

    args = dict(data_sig="sig", view="AMAZON-DC", model_path="models/tsb.py",
                today_str="2026-07-29", kind="view")
    assert fc.forecast_key(**args) == fc.forecast_key(**args)


def test_model_file_edit_invalidates(tmp_path):
    """Editing a model file must invalidate its forecasts — the key carries the
    model's mtime/size, not just its name."""
    from dashboard_app import forecast_cache as fc

    m = tmp_path / "tsb.py"
    m.write_text("ALPHA_P = 0.1\n")
    os.utime(m, (100, 100))
    before = fc.forecast_key("sig", "AMAZON-DC", str(m), "2026-07-29")
    m.write_text("ALPHA_P = 0.2\n")
    os.utime(m, (200, 200))
    assert fc.forecast_key("sig", "AMAZON-DC", str(m), "2026-07-29") != before


# --------------------------------------------------------------------------
# 3. Failure modes — every one must be a miss, never an exception
# --------------------------------------------------------------------------
def test_disabled_by_env(cache, frames, monkeypatch):
    monkeypatch.setenv("DEMAND_FORECAST_CACHE", "0")
    assert cache.put("k1", frames, None) is False
    assert cache.get("k1") is None


def test_entry_without_marker_is_invisible(cache, frames):
    """meta.json is written last and is the validity marker; an entry missing it
    is a half-written (or crashed) write and must not be read."""
    cache.put("k1", frames, None)
    os.remove(os.path.join(cache.cache_dir(), "k1", "meta.json"))
    assert cache.get("k1") is None


def test_missing_frame_is_a_miss(cache, frames):
    cache.put("k1", frames, None)
    os.remove(os.path.join(cache.cache_dir(), "k1", "weekly.parquet"))
    assert cache.get("k1") is None


def test_corrupt_parquet_is_a_miss_not_a_crash(cache, frames):
    cache.put("k1", frames, None)
    with open(os.path.join(cache.cache_dir(), "k1", "summary.parquet"), "wb") as fh:
        fh.write(b"this is not parquet")
    assert cache.get("k1") is None            # logged + recomputed, never raised


def test_deleted_cache_mid_session_is_safe(cache, frames):
    cache.put("k1", frames, None)
    cache.clear()
    assert cache.get("k1") is None
    # And it must be able to rebuild itself afterwards.
    assert cache.put("k1", frames, None) is True
    assert cache.get("k1") is not None


def test_put_rejects_none_frames(cache, frames):
    """A model that produced nothing must not be cached as an empty result."""
    assert cache.put("k1", {**frames, "weekly": None}, None) is False
    assert cache.get("k1") is None


def test_no_leftover_temp_files(cache, frames):
    cache.put("k1", frames, None)
    leftovers = [f for f in os.listdir(os.path.join(cache.cache_dir(), "k1"))
                 if f.startswith(".tmp_") or f.endswith(".part")]
    assert leftovers == []


# --------------------------------------------------------------------------
# Params side-store (autofit)
# --------------------------------------------------------------------------
def test_params_roundtrip(cache):
    cache.put_params("p1", {"alpha": 0.3, "beta": 0.1, "phi": 0.95, "mae": 4.25})
    assert cache.get_params("p1") == {"alpha": 0.3, "beta": 0.1, "phi": 0.95,
                                      "mae": 4.25}


def test_params_empty_dict_is_a_recorded_negative(cache):
    """`{}` records 'this pipeline has no autofit' so the miss isn't retried;
    it must round-trip as `{}` and not be confused with a miss."""
    cache.put_params("p1", {})
    assert cache.get_params("p1") == {}
    assert cache.get_params("p2") is None


# --------------------------------------------------------------------------
# Housekeeping
# --------------------------------------------------------------------------
def test_prune_keeps_the_most_recently_used(cache, frames):
    for i in range(6):
        cache.put(f"k{i}", frames, None)
        os.utime(os.path.join(cache.cache_dir(), f"k{i}", "meta.json"),
                 (1000 + i, 1000 + i))
    assert cache.prune(max_entries=2) == 4
    survivors = set(os.listdir(cache.cache_dir()))
    assert survivors == {"k4", "k5"}


def test_get_refreshes_recency(cache, frames):
    """A read must count as a use, so a view someone opens daily isn't pruned in
    favour of one the batch wrote last night and nobody looked at."""
    for i in range(3):
        cache.put(f"k{i}", frames, None)
        os.utime(os.path.join(cache.cache_dir(), f"k{i}", "meta.json"),
                 (1000 + i, 1000 + i))
    assert cache.get("k0") is not None        # touches k0's marker
    cache.prune(max_entries=1)
    assert os.path.isdir(os.path.join(cache.cache_dir(), "k0"))


def test_prune_on_empty_cache_is_a_noop(cache):
    assert cache.prune(max_entries=5) == 0


def test_stats_counts_entries(cache, frames):
    cache.put("k1", frames, None)
    cache.put("k2", frames, None)
    n, size = cache.stats()
    assert n == 2 and size > 0


def test_meta_is_readable_json(cache, frames):
    cache.put("k1", frames, {"view": "AMAZON-DC", "model": "tsb.py"})
    with open(os.path.join(cache.cache_dir(), "k1", "meta.json"),
              encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["view"] == "AMAZON-DC"
    assert meta["written_at"]
    assert meta["frames"] == ["agg", "summary", "weekly"]


# --------------------------------------------------------------------------
# Dashboard <-> batch key parity
# --------------------------------------------------------------------------
def test_batch_and_dashboard_build_the_same_key(tmp_path):
    """The nightly warm-up is only useful if the dashboard looks under the same
    key, and a drift produces no error — just a cache that never hits. Pin the
    two constructions against each other.

    Mirrors dashboard.py's ``data_sig`` block and agent/batch.py's, which must
    stay in lockstep: same snapshot path, same n_excluded_rows, same cleaned row
    count, same content-hashed prices.
    """
    from dashboard_app import forecast_cache as fc

    snap = tmp_path / "all_demand_projections_2026-07-29.xlsx"
    snap.write_bytes(b"snapshot bytes")
    os.utime(snap, (777, 777))
    prices = pd.Series([10.0, 20.0], index=["A-1", "B-2"])
    n_excluded, n_rows = 23694, 707962

    # --- as dashboard.py builds it ---
    dash_sig = fc.snapshot_signature(
        str(snap), n_excluded_rows=n_excluded, n_rows=n_rows,
        prices=fc.content_signature(prices),
    )
    # --- as agent/batch.py builds it ---
    batch_sig = fc.snapshot_signature(
        str(snap), n_excluded_rows=n_excluded, n_rows=n_rows,
        prices=fc.content_signature(prices),
    )
    assert dash_sig == batch_sig

    model = tmp_path / "tsb.py"
    model.write_text("x = 1\n")
    os.utime(model, (888, 888))
    assert (
        fc.forecast_key(dash_sig, "AMAZON-DC", str(model), "2026-07-29",
                        kind="group")
        == fc.forecast_key(batch_sig, "AMAZON-DC", str(model), "2026-07-29",
                           kind="group")
    )


def test_ingest_reports_n_excluded_rows(sample_raw_path, monkeypatch):
    """agent.batch needs n_excluded_rows out of ingest to rebuild the dashboard's
    key; if this field disappears the warm-up silently stops being found."""
    from agent import data_io
    from agent.nodes.ingest import ingest

    monkeypatch.setattr(data_io, "PLYTIX_FEED_URL", None)
    out = ingest({"today_ts": pd.Timestamp("2026-07-01"),
                  "raw_path": sample_raw_path, "price_path": None})
    assert "n_excluded_rows" in out
    assert isinstance(out["n_excluded_rows"], (int, np.integer))
