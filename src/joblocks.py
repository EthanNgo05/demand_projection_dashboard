"""Shared, Streamlit-free primitives for the background data jobs.

Both front-ends launch the same four jobs — the demand pull, the warehouse pull,
the key-SKU pull and the all-views ``agent.batch`` — and both coordinate through
the same lock files, failure markers and dated log files:

  * ``dashboard_app/refresh.py`` launches them **detached** from a button click,
    so the page stays usable while a ~10-minute query runs, and polls the lock
    on each rerun.
  * ``scheduler.py`` runs them **sequentially and blocking** at 00:00, because
    the nightly job cares about ordering (``agent.batch`` must see the fresh
    snapshot) and about the aggregate exit code.

Those two shapes are genuinely different, but the *coordination* is identical,
and it has to be: a manual click at 00:05 and the nightly run must see each
other's lock, and a nightly failure must write the marker the dashboard's error
banner already reads. This module is that shared half.

It is deliberately free of any ``streamlit`` import — ``refresh.py`` carries one
(and touches ``st.session_state`` in the agent-batch helpers), which is exactly
why the scheduler daemon cannot import it. Same rule, and same reason, as
``log_config.py`` and ``forecast_cache.py``.

**Path resolution deliberately stays with the callers.** The dashboard resolves
the raw folder through ``dashboard_app.datasources._raw_dir()`` (which honours
``DEMAND_PIPELINE``) and the scheduler through
``agent.data_io._raw_dir(default_pipeline())``. Both end at the same folder —
they read the same ``RAW_INPUTS_FOLDER`` constant and both honour
``DEMAND_RAW_DIR`` — but centralising that here would create a third resolution
path and a way for the two to drift onto different lock files, which is the one
failure this module exists to prevent.
"""

import os
import re
import time
import socket
import logging
import subprocess

from log_config import dated_log_path

logger = logging.getLogger("demand_dashboard")

# The one timestamp spelling shared by every lock, failure marker and run header.
# ``refresh.batch_elapsed_suffix`` parses it back with ``pd.Timestamp`` and
# ``_batch_run_finished`` matches it as a substring of a run header, so the
# format is a contract between writers and readers, not a display choice.
STAMP_FMT = "%Y-%m-%d %H:%M:%S"


def now_stamp():
    """Current local time in the shared lock/log timestamp format."""
    return time.strftime(STAMP_FMT)


# --------------------------------------------------------------------------- #
# Child-process conventions                                                    #
# --------------------------------------------------------------------------- #
def bg_creationflags():
    """Windows flags for a background child that shows NO console window.

    ``CREATE_NO_WINDOW`` gives the child a *hidden* console (rather than none at
    all, as ``DETACHED_PROCESS`` would); any grandchildren it spawns — e.g. the
    agent batch's process-pool workers — inherit that hidden console instead of
    each allocating a fresh visible terminal window. ``CREATE_NEW_PROCESS_GROUP``
    keeps the child off the parent's Ctrl-C group. Non-Windows: no flags.
    """
    if os.name == "nt":
        return subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    return 0


def child_env(**overrides):
    """Child environment for a background subprocess whose log we read live.

    - ``PYTHONIOENCODING=utf-8``: without it the child writes its log in the
      Windows locale code page (cp1252), where characters like '×' become byte
      0xd7 that a UTF-8 reader can't decode.
    - ``PYTHONUNBUFFERED=1``: a redirected-to-file stdout is block-buffered by
      default, so per-view progress lines wouldn't reach the log until the child
      exits (leaving "Check progress" stuck on "getting started"). Unbuffering
      flushes each line as it's printed so the progress reader sees it live.
    """
    return {**os.environ, "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1", **overrides}


# --------------------------------------------------------------------------- #
# Per-pull log files                                                           #
# --------------------------------------------------------------------------- #
# One log file PER PULL. The demand, warehouse and key-SKU children are launched
# within the same second and each inherits its own append handle; pointing all
# three at a single logs_refresh.txt made their output interleave and clobber
# each other mid-line — during the 2026-08 outage the "Connecting:" line was cut
# off mid-token and two of the three children's output vanished entirely, which
# is a large part of why a hard connection failure went undiagnosed for six days.
# The repo also lives on a network path shared by more than one host, so several
# writers are the norm, not the exception.
DEMAND_LOG = "logs_refresh_demand.txt"
WAREHOUSE_LOG = "logs_refresh_warehouse.txt"
KEY_SKUS_LOG = "logs_refresh_key_skus.txt"
BATCH_LOG = "logs_agent_batch.txt"
SCHEDULER_LOG = "logs_scheduler.txt"


def refresh_log_path(filename=DEMAND_LOG):
    """Today's log for one pull: ``logs/<date>/<filename>``. Computed per call
    (not at import) so a long-running dashboard files each refresh under the day
    it ran, and shares the exact file the scheduled task writes."""
    return dated_log_path(filename)


# Lines that mean the pull failed. The extracts log "Database error: (...)" for
# any pyodbc failure and "ERROR" for config problems; a traceback covers the
# crashes neither path catches.
ERROR_LINE_RE = re.compile(r"ERROR|Traceback \(most recent call last\)")
# Header written before each run, used to scope the error scan to the CURRENT
# run rather than an earlier one in the same day's file.
RUN_HEADER_PREFIX = "====="


def run_header(label, when, host=None):
    """The one run-header spelling every launcher writes before a child starts.

    ``last_error_line`` scopes its scan to the newest header, so the demand pull
    launched by the nightly scheduler and the one launched by the button have to
    write the *same* marker into the *same* file or an error from one run gets
    attributed to the other.
    """
    host = host or socket.gethostname()
    return f"\n===== {label} started {when} on {host} =====\n"


def last_error_line(log_path):
    """The last error line from this pull's CURRENT run, or None.

    Takes a resolved *path*, not a log name: the day's folder is decided by the
    caller (``refresh._refresh_log_path``, which tests monkeypatch to redirect a
    whole run into a tmpdir), so resolving it again here would quietly ignore
    that redirection.

    Only lines after the newest run header are considered, so an earlier failure
    in the same day's file can't be misreported as the running pull's outcome.
    Best-effort: any read problem simply yields None.
    """
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return None

    for i in range(len(lines) - 1, -1, -1):
        if lines[i].lstrip().startswith(RUN_HEADER_PREFIX):
            lines = lines[i + 1:]
            break

    hits = [ln.strip() for ln in lines if ERROR_LINE_RE.search(ln)]
    return hits[-1][:500] if hits else None


# --------------------------------------------------------------------------- #
# Locks                                                                        #
# --------------------------------------------------------------------------- #
# Every pull's lock is a ``.refresh.lock`` inside the folder that pull writes to,
# so a manual click and the nightly run coordinate through one file regardless of
# which started it. The name is chosen so no snapshot-discovery glob
# (``all_demand_projections_*.xlsx``, ``*.xlsx``, ``key_skus_*.xlsx``) matches it
# and it can never be mistaken for a data file.
LOCK_NAME = ".refresh.lock"

# The agent batch produces no dated snapshot of its own, so its lock lives under
# ``outputs/`` beside the ``agent_summary_<view>.json`` files it writes, and is
# named for the job rather than for the folder.
BATCH_LOCK_NAME = ".agent_batch.lock"


def lock_path(folder):
    """The lock file for a pull that writes into ``folder``."""
    return os.path.join(folder, LOCK_NAME)


def batch_lock_path(repo_root):
    """The all-views ``agent.batch`` lock: ``<repo_root>/outputs/``."""
    return os.path.join(repo_root, "outputs", BATCH_LOCK_NAME)


# How long a lock may sit with no result before it is treated as a crashed run
# rather than a live one, so a button re-enables (dashboard) or a step is retried
# (scheduler) instead of wedging forever. Shared, because the two front-ends have
# to agree on whether a given lock is live.
#
#   * demand / warehouse: comfortably above the ~10-minute typical runtime and
#     the extract's own 900s SQL_QUERY_TIMEOUT default.
#   * key SKUs: a single DISTINCT query, so a failed run frees up quickly.
#   * batch: 114 views x 5 models has measured 48-58 minutes with the LLM on.
#     The old 90-minute window left under half an hour of headroom, and the lock
#     self-clearing UNDER a live run is worse than a stale lock — the dashboard
#     then reports the batch finished while it is still writing summaries. Three
#     hours is the same "generous multiple of the real runtime" the other two
#     windows use.
REFRESH_STALE_SECONDS = 30 * 60
KEY_SKUS_STALE_SECONDS = 5 * 60
BATCH_STALE_SECONDS = 3 * 60 * 60


def acquire(path, started, pid=None, note=None):
    """Write a lock recording ``started`` (and optionally the child's PID).

    Line 1 is the start timestamp; line 2, when present, is the PID of the
    process doing the work. The caller writes the lock BEFORE launching, so a
    double-click can't spawn two pulls, then rewrites it with the PID once the
    child exists.

    ``pid`` must be the PID of the process that actually *does* the work — for
    the agent batch that is the ``python -m agent.batch`` child, not the
    scheduler that spawned it: ``refresh._pid_running_batch`` confirms the
    recorded PID's command line contains ``agent.batch`` before believing the run
    is alive, and would read a launcher's PID as "batch already finished".

    ``note`` writes a free-text third line (the scheduler records its hostname
    there). Readers only ever look at the first two lines, so an extra line is
    invisible to them — but a SINGLE-line lock must stay exactly one line with no
    trailing newline: ``_refresh_state`` shows ``f.read().strip()`` to the user as
    the start time, so a second line would reach the UI as part of the timestamp.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = started
    if pid:
        body = f"{started}\n{pid}\n"
        if note:
            body = f"{started}\n{pid}\n{note}\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)


def read_lock(path):
    """``(started_str, pid_or_None)`` from a lock; ``("", None)`` on any error.

    Line 1 is the start timestamp; line 2, if present, is the worker PID. Older
    single-line locks have no PID and read back as ``pid=None``.
    """
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return "", None
    started = lines[0].strip() if lines else ""
    pid = None
    if len(lines) > 1 and lines[1].strip().isdigit():
        pid = int(lines[1].strip())
    return started, pid


def read_note(path):
    """The free-text third line of a lock (the scheduler's host), or ""."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return ""
    return lines[2].strip() if len(lines) > 2 else ""


def release(path):
    """Remove a lock (or a failure marker), ignoring an already-gone file."""
    try:
        os.remove(path)
    except OSError:
        pass


def lock_is_live(path, stale_seconds):
    """``(live, started_str)`` — is a run holding ``path`` right now?

    "Live" means the lock exists and is younger than ``stale_seconds``. Older
    than that and the run that wrote it crashed or was killed without cleaning
    up, so the lock is ignored (and the caller is free to take over).

    This is the *cheap* liveness question, the one a launcher asks before
    starting work. ``refresh._refresh_state`` asks a richer one — it also
    consults the child handle, the pull's log and whether the expected output
    landed — because it has to drive a UI that says *why* a pull is not running.
    A launcher only needs to know whether starting a second one would collide.
    """
    if not os.path.exists(path):
        return False, None
    try:
        age = time.time() - os.path.getmtime(path)
    except OSError:
        return False, None
    if age > stale_seconds:
        return False, None
    started, _pid = read_lock(path)
    return True, started


# --------------------------------------------------------------------------- #
# Failure markers                                                              #
# --------------------------------------------------------------------------- #
def failure_path(lock):
    """Sibling of the lock recording the last failed pull.

    Persisted next to the lock (not in session_state) so the error survives a
    browser reload, reaches a second browser session, and is visible from the
    other host that shares this folder. Not matched by any snapshot glob.
    """
    return lock + ".failed"


def record_failure(lock, label, log_path, reason):
    """Log the failure and persist it for the UI banner.

    Before this, a failed pull was indistinguishable from an idle one: the lock
    was cleared, the button re-enabled, and nothing was written anywhere the
    user would look. Six days of failed syncs went unnoticed that way. The
    nightly scheduler writes the same marker for the same reason, and more
    urgently — at 00:00 there is no one watching a progress spinner at all.
    """
    detail = last_error_line(log_path) or reason
    logger.error("%s failed on host %s: %s", label, socket.gethostname(), detail)
    try:
        with open(failure_path(lock), "w", encoding="utf-8") as f:
            f.write("\t".join([now_stamp(), socket.gethostname(), detail]))
    except OSError:  # never let a diagnostics write break the pull's state
        pass
    return detail


def read_failure(lock):
    """``(when, host, detail)`` for a pull's last failure, or None."""
    try:
        with open(failure_path(lock), encoding="utf-8") as f:
            when, host, detail = f.read().split("\t", 2)
    except (OSError, ValueError):
        return None
    return when, host, detail
