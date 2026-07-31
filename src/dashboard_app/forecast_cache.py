"""Persistent on-disk cache for computed forecast frames.

Why this exists
---------------
Forecasts used to live only in ``st.session_state`` (dashboard.py's ``fc_cache``,
capped at 16 entries). That made a *revisited* view fast within one browser
session, but a page refresh, an app restart, or a second planner opening the
dashboard paid the full cost again — and the full cost is real: ~5s for a single
Holt view, ~40s for the all-customers per-group loop, ~55s for Optimized
Projections. Meanwhile the nightly ``agent.batch`` already fits every model for
every view and threw all of those numbers away, persisting only which model won.

This module is the second cache tier: frames written to ``outputs/.cache`` as
Parquet, keyed on the *inputs* that determine them. Anything computed once —
tonight by the batch, or earlier today by a colleague — comes back as a ~0.1s
Parquet read instead of a recompute.

Contract
--------
* **Streamlit-free.** Carries no ``st.*`` so ``agent/batch.py`` can warm the cache
  in a worker process with no Streamlit runtime (the same constraint
  ``config.py`` and ``summaries.py`` honour).
* **Best-effort, never load-bearing.** Every read and write is wrapped: a missing
  pyarrow, a corrupt file, a locked file or a half-written entry logs at debug/
  warning level and returns as a miss, so the caller recomputes. A broken cache
  must never break the dashboard, and deleting ``outputs/.cache`` at any moment
  must be safe.
* **Numerically exact.** Frames go through Parquet, which round-trips every dtype
  these pipelines produce — including the object-dtype ``datetime.date`` column
  in the weekly frames. ``tests/test_forecast_cache.py`` pins that: a cached
  frame must satisfy ``assert_frame_equal(check_exact=True)`` against a fresh
  compute, so a pandas/pyarrow upgrade that breaks fidelity fails a test rather
  than silently changing displayed numbers.
* **Atomic.** Frames are written via ``mkstemp`` + ``os.replace`` inside the
  entry directory, and ``meta.json`` is written LAST. ``get`` treats an entry
  without ``meta.json`` as absent, so a crash mid-write leaves a miss, never a
  partial read. (Contrast ``agent/nodes/publish.py``, which writes its JSON
  non-atomically — deliberately not copied here.)

Invalidation
------------
Keys are built from file identity, not file contents: ``snapshot_signature``
hashes ``(basename, st_mtime_ns, st_size)`` for the demand snapshot, the price
export and the model file. That is a few ``stat()`` calls rather than a hash of
708k rows, and it invalidates on exactly the events that matter — a new nightly
snapshot, an incremental refresh rewriting the same path, a fresh Plytix export,
or an edit to a model file. ``forecast_key`` folds that signature together with
the view, the smoothing parameters and the run date.

Disable with ``DEMAND_FORECAST_CACHE=0`` (reads and writes both become no-ops),
which is what ``scripts/bench_dashboard.py`` does so a warm cache can't flatter
a timing run.
"""

import hashlib
import json
import logging
import os
import shutil
import tempfile
import time

import pandas as pd

from dashboard_app.config import REPO_ROOT

log = logging.getLogger(__name__)

# Entry directories kept before the oldest are pruned. ~70 views x a handful of
# models/parameter sets, so a few hundred covers a normal working set; each entry
# is a few hundred KB of Parquet.
MAX_ENTRIES = int(os.environ.get("DEMAND_FORECAST_CACHE_MAX", "500"))

# The frames compute_view / _forecast_one_group return, in order.
FRAME_NAMES = ("summary", "weekly", "agg")

_META = "meta.json"


def enabled():
    """False when DEMAND_FORECAST_CACHE=0 — reads and writes become no-ops."""
    return os.environ.get("DEMAND_FORECAST_CACHE", "1") != "0"


def cache_dir():
    """``<repo>/outputs/.cache/forecasts`` (outputs/ is already gitignored)."""
    return os.path.join(REPO_ROOT, "outputs", ".cache", "forecasts")


# --------------------------------------------------------------------------
# Keys
# --------------------------------------------------------------------------
def _stat_part(path):
    """``(basename, mtime_ns, size)`` for ``path``, or None if it isn't there.

    Basename rather than the full path so the cache survives the repo being
    moved or checked out elsewhere; mtime_ns + size to catch an in-place rewrite
    (which the ``--incremental`` demand pull does to the same filename).
    """
    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    return [os.path.basename(path), st.st_mtime_ns, st.st_size]


def content_signature(obj):
    """Content hash of a Series/DataFrame (or None), for small exact-keyed inputs.

    Used for the list-price Series: prices reach the dashboard from three
    different sources — an uploaded workbook, the Plytix HTTP feed, or the newest
    local ``list_prices_*.xlsx`` — so no single file path identifies them, and the
    feed can change without any filename changing. Prices feed the revenue-risk
    columns, so a stale price in a cached forecast would be a wrong number on
    screen. Hashing the ~2k-row Series outright costs about a millisecond, which
    buys exactness rather than a ``len()`` proxy.

    Not used for the demand frame: that is 700k+ rows, and file identity
    (mtime_ns + size) already distinguishes snapshots including in-place rewrites.
    """
    if obj is None:
        return None
    try:
        return hashlib.sha1(
            pd.util.hash_pandas_object(obj, index=True).to_numpy().tobytes()
        ).hexdigest()[:16]
    except Exception as exc:                    # noqa: BLE001 - never break a caller
        log.debug("content_signature failed (%s); falling back to length", exc)
        return f"len:{len(obj)}"


def snapshot_signature(*paths, **extras):
    """Short hash identifying a set of input files plus any extra scalars.

    Pass the demand snapshot, the price export and the model file. ``extras``
    carries values that change the cleaned frame without changing any file — the
    dashboard passes ``n_excluded_rows`` and the row count, because the frame
    handed to the compute functions is the POST-exclusion one and exclusions
    depend on the Plytix status columns.
    """
    payload = {
        "files": [_stat_part(p) for p in paths],
        "extras": {k: extras[k] for k in sorted(extras)},
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def forecast_key(data_sig, view, model_path, today_str,
                 alpha=None, beta=None, phi=None, min_weeks=None, kind="view"):
    """Cache key for one forecast.

    ``kind`` separates otherwise-identical keys whose payload means different
    things (a single view's frames vs. one customer group's frames), so the
    per-view and per-group caches can never collide.
    """
    payload = {
        "kind": kind,
        "data": data_sig,
        "view": str(view),
        "model": os.path.basename(str(model_path)),
        "model_stat": _stat_part(model_path),
        "today": str(today_str),
        "alpha": None if alpha is None else float(alpha),
        "beta": None if beta is None else float(beta),
        "phi": None if phi is None else float(phi),
        "min_weeks": None if min_weeks is None else int(min_weeks),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Read / write
# --------------------------------------------------------------------------
def get(key, names=FRAME_NAMES):
    """Return ``{name: DataFrame}`` for ``key``, or None on any kind of miss.

    A miss is: caching disabled, no entry directory, no ``meta.json`` (so the
    entry is still being written or was left half-finished), a missing frame
    file, or any read error at all. Every one of those degrades to "recompute".
    """
    if not enabled():
        return None
    entry = os.path.join(cache_dir(), key)
    try:
        if not os.path.exists(os.path.join(entry, _META)):
            return None
        frames = {}
        for name in names:
            path = os.path.join(entry, f"{name}.parquet")
            if not os.path.exists(path):
                return None                     # incomplete entry -> treat as miss
            frames[name] = pd.read_parquet(path)
    except Exception as exc:                    # noqa: BLE001 - cache must not raise
        log.debug("Forecast cache read failed for %s: %s", key[:12], exc)
        return None
    # Touch the marker so prune()'s oldest-first ordering is least-recently-USED
    # rather than least-recently-written; a view someone opens daily stays put.
    try:
        os.utime(os.path.join(entry, _META), None)
    except OSError:
        pass
    return frames


def put(key, frames, meta=None):
    """Write ``{name: DataFrame}`` under ``key``. Returns True if it landed.

    ``meta.json`` is written last and is what ``get`` looks for, so an entry is
    either fully readable or invisible. Failures are swallowed (logged once) —
    the caller already has the computed frames and does not care.
    """
    if not enabled():
        return False
    if any(f is None for f in frames.values()):
        return False                            # nothing useful to store
    entry = os.path.join(cache_dir(), key)
    try:
        os.makedirs(entry, exist_ok=True)
        for name, df in frames.items():
            _atomic_write(
                os.path.join(entry, f"{name}.parquet"),
                lambda p, d=df: d.to_parquet(p, index=False),
            )
        payload = dict(meta or {})
        payload["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        payload["frames"] = sorted(frames)
        _atomic_write(
            os.path.join(entry, _META),
            lambda p: _write_json(p, payload),
        )
        return True
    except Exception as exc:                    # noqa: BLE001 - cache must not raise
        log.warning("Could not write forecast cache entry %s: %s", key[:12], exc)
        # Leave no half-written entry behind: without meta.json get() ignores it,
        # but drop the directory anyway so prune() doesn't count dead weight.
        try:
            if not os.path.exists(os.path.join(entry, _META)):
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            pass
        return False


def get_params(key):
    """Return a cached scalar-parameter dict for ``key``, or None on a miss.

    The autofit grid search costs ~5s per view and returns a handful of floats
    rather than frames, so it gets its own tiny JSON entry instead of Parquet.
    Same marker discipline as ``get``: no ``meta.json``, no hit.
    """
    if not enabled():
        return None
    entry = os.path.join(cache_dir(), key)
    try:
        if not os.path.exists(os.path.join(entry, _META)):
            return None
        with open(os.path.join(entry, "params.json"), encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:                    # noqa: BLE001 - cache must not raise
        log.debug("Forecast cache param read failed for %s: %s", key[:12], exc)
        return None


def put_params(key, params, meta=None):
    """Persist a scalar-parameter dict (autofit's alpha/beta/phi + scores)."""
    if not enabled() or params is None:
        return False
    entry = os.path.join(cache_dir(), key)
    try:
        os.makedirs(entry, exist_ok=True)
        _atomic_write(os.path.join(entry, "params.json"),
                      lambda p: _write_json(p, dict(params)))
        payload = dict(meta or {})
        payload["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        payload["frames"] = ["params"]
        _atomic_write(os.path.join(entry, _META), lambda p: _write_json(p, payload))
        return True
    except Exception as exc:                    # noqa: BLE001 - cache must not raise
        log.warning("Could not write forecast cache params %s: %s", key[:12], exc)
        try:
            if not os.path.exists(os.path.join(entry, _META)):
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            pass
        return False


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _atomic_write(path, writer):
    """``writer(tmp)`` into a sibling temp file, then ``os.replace`` into place.

    Same shape as watchlist.py's atomic JSON write. The temp file is created in
    the destination directory so the replace is same-volume (and therefore
    atomic) on Windows as well as POSIX.
    """
    out_dir = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=".tmp_", suffix=".part")
    os.close(fd)
    try:
        writer(tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------
# Housekeeping
# --------------------------------------------------------------------------
def prune(max_entries=None):
    """Drop the least-recently-used entries beyond ``max_entries``.

    Called at the end of ``agent.batch`` so the nightly warm-up can't grow
    ``outputs/.cache`` without bound. Entries whose ``meta.json`` is unreadable
    sort oldest and get collected first. Returns the number removed.
    """
    cap = MAX_ENTRIES if max_entries is None else int(max_entries)
    root = cache_dir()
    if not os.path.isdir(root):
        return 0
    entries = []
    try:
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if not os.path.isdir(path):
                continue
            try:
                stamp = os.path.getmtime(os.path.join(path, _META))
            except OSError:
                stamp = 0.0                     # no/unreadable marker -> evict first
            entries.append((stamp, path))
    except OSError as exc:
        log.debug("Forecast cache prune could not list %s: %s", root, exc)
        return 0

    removed = 0
    for _, path in sorted(entries)[: max(len(entries) - cap, 0)]:
        try:
            shutil.rmtree(path)
            removed += 1
        except OSError as exc:
            log.debug("Forecast cache prune could not remove %s: %s", path, exc)
    if removed:
        log.info("Forecast cache: pruned %d of %d entries (cap %d).",
                 removed, len(entries), cap)
    return removed


def clear():
    """Remove the whole cache. Used by tests and available for manual reset."""
    shutil.rmtree(cache_dir(), ignore_errors=True)


def stats():
    """``(n_entries, total_bytes)`` — for a diagnostics caption or a log line."""
    root = cache_dir()
    n = 0
    size = 0
    if not os.path.isdir(root):
        return 0, 0
    for dirpath, _dirnames, filenames in os.walk(root):
        if _META in filenames:
            n += 1
        for f in filenames:
            try:
                size += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return n, size
