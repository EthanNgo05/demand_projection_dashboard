"""Subprocess-backed manual refresh (demand / warehouse / agent batch)."""
import os
import re
import sys
import time
import logging
import subprocess

import pandas as pd
import streamlit as st

from log_config import dated_log_path
from agent import data_io

from dashboard_app.config import HERE, REPO_ROOT
from dashboard_app import datasources

logger = logging.getLogger("demand_dashboard")


def _bg_creationflags():
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


def _child_env(**overrides):
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
# Manual data-warehouse refresh                                               #
# --------------------------------------------------------------------------- #
# The demand snapshot is normally refreshed by a nightly scheduled task that
# runs extract_demand_details.py (the ~10-minute SQL pull) OUT of the request
# path, so the dashboard always serves a recent file instantly. This button lets
# a user force a fresh pull on demand WITHOUT blocking the page: it launches the
# extract as a detached background process and drops a lock file in the snapshot
# folder. While the lock is live the page stays fully usable on the current
# snapshot; when the child writes the new (atomic) workbook, the snapshot
# dropdown auto-selects it. The lock also stops a manual click and the nightly
# task from overlapping into two concurrent 10-minute queries.
EXTRACT_SCRIPT = os.path.join(HERE, "extract_demand_details.py")


def _refresh_log_path():
    """Today's refresh log: ``logs/<date>/logs_refresh.txt``. Computed per call
    (not at import) so a long-running dashboard files each refresh under the day
    it ran, and shares the exact file the scheduled task writes."""
    return dated_log_path("logs_refresh.txt")
# A pull older than this with no new file is treated as crashed, so the button
# re-enables instead of wedging the UI forever. Comfortably above the ~10-minute
# typical runtime and the extract's own 900s SQL_QUERY_TIMEOUT default.
REFRESH_STALE_SECONDS = 30 * 60


def _refresh_lock_path():
    """Lock file marking an in-flight DW pull, kept in the snapshot folder.

    Lives inside the raw folder (not matched by the ``all_demand_projections_*``
    glob, so it never shows up as a snapshot) so a click and the nightly task
    coordinate through one file regardless of which one started the pull.
    """
    return os.path.join(datasources._raw_dir(), ".refresh.lock")


def _clear_lock(lock_path):
    """Remove a refresh lock, ignoring the case where it's already gone."""
    try:
        os.remove(lock_path)
    except OSError:
        pass


def _refresh_state(lock_path, completed_since, label):
    """Shared lock state-machine: (running, started_str) for a background pull.

    Self-healing, so no process has to clean up after itself:
      * ``completed_since(lock_mtime)`` says whether the pull's output has
        landed since the lock appeared — if so, clear the lock and report idle.
      * If the lock is older than REFRESH_STALE_SECONDS with no output, the
        pull crashed/was killed — clear the lock so the button re-enables.

    What "output has landed" means differs per pull (the demand snapshot is one
    atomic workbook; a warehouse snapshot is a five-file set), which is exactly
    the ``completed_since`` seam.
    """
    if not os.path.exists(lock_path):
        return False, None
    lock_mtime = os.path.getmtime(lock_path)

    if completed_since(lock_mtime):
        _clear_lock(lock_path)
        return False, None

    if time.time() - lock_mtime > REFRESH_STALE_SECONDS:
        logger.warning("%s refresh lock is stale (>%ds); clearing it.",
                       label, REFRESH_STALE_SECONDS)
        _clear_lock(lock_path)
        return False, None

    try:
        with open(lock_path, encoding="utf-8") as f:
            started = f.read().strip()
    except OSError:
        started = ""
    return True, started


def refresh_in_progress():
    """(running, started_str): is a background DW pull active, and when it began.

    Completion = any demand snapshot written since the lock appeared (the
    extract writes one atomic workbook, so the first newer file IS the result).
    """
    def _completed(lock_mtime):
        files = datasources.discover_raw_files()
        return bool(files) and max(
            os.path.getmtime(p) for _, p in files
        ) >= lock_mtime

    return _refresh_state(_refresh_lock_path(), _completed, "DW")


def start_refresh(incremental: bool = True):
    """Launch extract_demand_details.py in the background. Returns (ok, message).

    Reuses THIS interpreter (``sys.executable``) so the pull runs in the same
    venv the dashboard was started with, and inherits the environment (the SQL_*
    connection vars). ``DEMAND_RAW_DIR`` is pinned to the exact folder the
    dashboard reads so the child writes where we look, regardless of CWD. The
    child's output is appended to logs/<date>/logs_refresh.txt for diagnosis.

    ``incremental`` (the default) pulls only the last few weeks of actuals plus
    all forward projections and merges them into the newest snapshot — minutes
    instead of the ~20-minute full pull. The nightly scheduled task still runs
    the full pull as the self-healing baseline.
    """
    running, started = refresh_in_progress()
    if running:
        return False, f"A refresh is already running (started {started})."

    raw_dir = datasources._raw_dir()
    os.makedirs(raw_dir, exist_ok=True)
    mode = "incremental" if incremental else "full"
    return _launch_refresh(
        _refresh_lock_path(),
        EXTRACT_SCRIPT,
        ["--incremental"] if incremental else [],
        {"DEMAND_RAW_DIR": raw_dir},
        f"DW refresh ({mode})",
    )


def _launch_refresh(lock_path, script, extra_args, env_overrides, header):
    """Write ``lock_path``, then launch ``script`` detached. Returns (ok, msg).

    The lock is written BEFORE launching so a double-click can't spawn two
    pulls. The child reuses THIS interpreter (``sys.executable``) so it runs in
    the same venv the dashboard was started with and inherits the environment
    (the SQL_* connection vars) plus ``env_overrides`` (the raw-dir pin, so the
    child writes exactly where the dashboard looks, regardless of CWD). Output
    is appended to logs/<date>/logs_refresh.txt for diagnosis.
    """
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(now)

    try:
        env = _child_env(**env_overrides)
        # Run hidden on Windows so the pull outlives this Streamlit run/rerun
        # without flashing a console window (see _bg_creationflags).
        creationflags = _bg_creationflags()
        logf = open(_refresh_log_path(), "a", encoding="utf-8")
        try:
            logf.write(f"\n===== {header} started {now} =====\n")
            logf.flush()
            subprocess.Popen(
                [sys.executable, script] + extra_args,
                cwd=HERE,
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
                close_fds=True,
            )
        finally:
            # The child holds its own duplicated handle; drop ours so the parent
            # doesn't leak a file handle per click.
            logf.close()
    except Exception as exc:
        _clear_lock(lock_path)
        logger.exception("Failed to launch %s", header)
        return False, f"Could not start refresh: {exc}"

    logger.info("%s launched (%s)", header, now)
    return True, now


# --------------------------------------------------------------------------- #
# Manual warehouse-projections refresh                                        #
# --------------------------------------------------------------------------- #
# Same lock-file coordination as the demand refresh above, with one twist: a
# warehouse snapshot is FIVE region files written back-to-back, not one atomic
# workbook. Each file is atomic, but the set is not — so completion means "the
# newest dated group holds every region, all newer than the lock", not "any
# newer file exists" (which would clear the lock after the first region lands
# and briefly serve a partial snapshot).
WAREHOUSE_EXTRACT_SCRIPT = os.path.join(HERE, "extract_warehouse_projections.py")


def _wh_refresh_lock_path():
    """Lock for an in-flight warehouse pull, in the warehouse snapshot folder
    (not matched by the ``*.xlsx`` discovery glob)."""
    return os.path.join(data_io._warehouse_dir(), ".refresh.lock")


def _wh_snapshot_complete_since(lock_mtime):
    """True once a full 5-region snapshot newer than the lock exists."""
    groups = data_io.discover_warehouse_files()
    if not groups:
        return False
    newest = [
        p for p in next(iter(groups.values())) if data_io._warehouse_region(p)
    ]
    regions = {data_io._warehouse_region(p) for p in newest}
    if not set(data_io.REGION_PREFIXES) <= regions:
        return False
    return all(os.path.getmtime(p) >= lock_mtime for p in newest)


def warehouse_refresh_in_progress():
    """(running, started_str): is a background warehouse pull active."""
    return _refresh_state(
        _wh_refresh_lock_path(), _wh_snapshot_complete_since, "Warehouse"
    )


def start_warehouse_refresh():
    """Launch extract_warehouse_projections.py in the background; (ok, msg)."""
    running, started = warehouse_refresh_in_progress()
    if running:
        return False, f"A warehouse refresh is already running (started {started})."

    wh_dir = data_io._warehouse_dir()
    os.makedirs(wh_dir, exist_ok=True)
    return _launch_refresh(
        _wh_refresh_lock_path(),
        WAREHOUSE_EXTRACT_SCRIPT,
        [],
        {"WAREHOUSE_RAW_DIR": wh_dir},
        "Warehouse refresh",
    )


# --------------------------------------------------------------------------- #
# Precompute every view's agent summary (agent.batch)                          #
# --------------------------------------------------------------------------- #
# The "Run Agent Summary" button above runs the agent for ONE view live. This
# section runs it for EVERY view (the same work as `python -m agent.batch`),
# which backtests all models across ~60 views and can take up to an hour. It
# reuses the demand-refresh pattern — a detached background process plus a lock
# file — so the page stays usable while it runs; the batch writes each
# outputs/agent_summary_<view>.json exactly as the nightly job does. We also keep
# the Popen handle in session_state so the current session detects completion
# promptly (the lock's stale timeout is only the cross-restart fallback).
BATCH_STALE_SECONDS = 90 * 60  # generous: a full LLM batch can approach an hour.


def _batch_lock_path():
    """Lock file marking an in-flight all-views batch, kept under outputs/."""
    return os.path.join(REPO_ROOT, "outputs", ".agent_batch.lock")


def _batch_log_path():
    """Today's batch log: ``logs/<date>/logs_agent_batch.txt`` (computed per call
    so a long-running dashboard files each run under the day it ran)."""
    return dated_log_path("logs_agent_batch.txt")


def _batch_result_line():
    """The last 'Done: N ok, M failed …' line from the batch log, or None.

    Lets the completion toast report the outcome without the batch signalling
    back into this process. Best-effort: any read error just yields None.

    ``errors="replace"`` keeps legacy cp1252 logs (written before the child was
    pinned to UTF-8) from raising UnicodeDecodeError while iterating every line;
    the 'Done:' line itself is pure ASCII, so it is never affected.
    """
    try:
        with open(_batch_log_path(), encoding="utf-8", errors="replace") as f:
            done = [ln.strip() for ln in f if ln.strip().startswith("Done:")]
        return done[-1] if done else None
    except OSError:
        return None


def batch_progress():
    """(done, total) from the batch log's latest '[N/M]' line, or None.

    The batch prints one '  [k/total] <view> -> …' line per finished view, so the
    highest k seen is how many views are done. Best-effort and tolerant of legacy
    (non-UTF-8) logs; returns None if no progress line is present yet.
    """
    try:
        done = total = None
        with open(_batch_log_path(), encoding="utf-8", errors="replace") as f:
            for ln in f:
                m = re.search(r"\[(\d+)/(\d+)\]", ln)
                if m:
                    done, total = int(m.group(1)), int(m.group(2))
        if total:
            return done, total
    except OSError:
        pass
    return None


def batch_elapsed_suffix(started):
    """" Started 2:05 PM, running 12 min." for the run banner, or "".

    ``started`` is the "%Y-%m-%d %H:%M:%S" start string batch_in_progress
    returns; yields "" if it's missing or unparseable so the banner degrades
    cleanly to no timing text.
    """
    if not started:
        return ""
    try:
        start = pd.Timestamp(started)
    except (ValueError, TypeError):
        return ""
    hour = start.strftime("%I").lstrip("0") or "12"          # 2, not 02
    clock = start.strftime(f"{hour}:%M %p")
    mins = max(int((pd.Timestamp.now() - start).total_seconds() // 60), 0)
    run = f"{mins} min" if mins < 60 else f"{mins // 60}h {mins % 60}m"
    return f" Started {clock}, running {run}."


def batch_failures():
    """[(view, error), …] for the most recent batch run, or [] if none.

    The batch prints a 'Failures:' block after its final 'Done:' line, one
    '  <view>: <error>' per line. We read the block belonging to the LATEST run
    (after the last 'Done:'), so a retry that clears a view drops it from the
    list next time. View names contain no ': ', so the first ': ' splits view
    from error. Tolerant of legacy (non-UTF-8) logs.
    """
    try:
        with open(_batch_log_path(), encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    done_idx = max(
        (i for i, ln in enumerate(lines) if ln.strip().startswith("Done:")),
        default=None,
    )
    if done_idx is None:
        return []
    out = []
    in_block = False
    for ln in lines[done_idx + 1:]:
        if ln.strip() == "Failures:":
            in_block = True
            continue
        if in_block and ln.startswith("  ") and ": " in ln:
            view, err = ln.strip().split(": ", 1)
            out.append((view, err))
        elif in_block and ln.strip() and not ln.startswith("  "):
            break  # reached the next run's header / unrelated content
    return out


def batch_result_message():
    """A friendly, non-technical completion sentence for the toast, or None.

    Translates the raw 'Done: N ok, M failed …' log line into planner-facing
    wording. Returns None only when there is no result line yet (caller supplies
    its own fallback).
    """
    line = _batch_result_line()
    if not line:
        return None
    m = re.search(r"Done:\s*(\d+)\s*ok,\s*(\d+)\s*failed", line)
    if not m:
        return "Recommendations finished."
    ok, failed = int(m.group(1)), int(m.group(2))
    total = ok + failed
    if failed == 0:
        return f"✅ Finished — recommendations updated for all {total} views."
    return (f"Finished — {ok} of {total} views updated; "
            f"{failed} need another try (listed below).")


def _batch_run_finished(started):
    """True once the run whose header carries ``started`` has logged a 'Done:'.

    A cross-session completion signal that doesn't need this session's Popen
    handle: the child prints 'Done:' as it exits, which flushes to the log. We
    match on the exact ``started`` timestamp so a *later*, still-running batch
    isn't reported finished just because an earlier run's 'Done:' sits above it.
    """
    if not started:
        return False
    try:
        with open(_batch_log_path(), encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return False
    hdr = None
    for i, ln in enumerate(lines):  # last header matching this run
        if ln.startswith("=====") and f"started {started}" in ln:
            hdr = i
    if hdr is None:
        return False
    return any(ln.strip().startswith("Done:") for ln in lines[hdr + 1:])


def _read_lock(lock_path):
    """(started_str, pid_or_None) from the batch lock; ('', None) on any error.

    Line 1 is the start timestamp (written before launch); line 2, if present,
    is the child PID (written just after launch). Older single-line locks have
    no PID and read back as pid=None.
    """
    try:
        with open(lock_path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return "", None
    started = lines[0].strip() if lines else ""
    pid = None
    if len(lines) > 1 and lines[1].strip().isdigit():
        pid = int(lines[1].strip())
    return started, pid


def _pid_running_batch(pid):
    """Whether ``pid`` is a live agent.batch process: True / False / None.

    None means "can't tell" (psutil unavailable) — callers keep the run marked
    running and defer to the log 'Done:' check and the stale timeout. We confirm
    the command line is the batch to guard against the OS reusing the PID for an
    unrelated process after the batch exits.
    """
    if not pid:
        return None
    try:
        import psutil
    except ImportError:
        return None
    try:
        proc = psutil.Process(int(pid))
    except psutil.NoSuchProcess:
        return False            # definitely gone
    except (ValueError, TypeError, psutil.Error):
        return None             # can't tell
    try:
        return "agent.batch" in " ".join(proc.cmdline())
    except psutil.NoSuchProcess:
        return False
    except psutil.Error:
        return True             # exists but cmdline unreadable -> assume alive


def batch_in_progress():
    """(running, started_str): is the all-views batch active, and when it began.

    Self-healing like _refresh_state; completion/crash is detected four ways:
      * the Popen handle we kept this session has exited (prompt, same-session),
      * the log shows this run's 'Done:' line (clean finish, cross-session),
      * the batch PID is no longer a live agent.batch process (crash or finish,
        cross-session), or
      * the lock is older than BATCH_STALE_SECONDS (last-ditch fallback).
    """
    lock_path = _batch_lock_path()
    if not os.path.exists(lock_path):
        return False, None
    lock_mtime = os.path.getmtime(lock_path)

    started, pid = _read_lock(lock_path)

    proc = st.session_state.get("agent_batch_proc")
    if proc is not None and proc.poll() is not None:
        _clear_lock(lock_path)
        return False, None

    # Cross-session: this session may not hold the Popen handle (browser refresh,
    # another session, or the nightly job). Trust the log's 'Done:' so the status
    # flips to finished promptly instead of waiting out the stale timeout.
    if _batch_run_finished(started):
        _clear_lock(lock_path)
        return False, None

    # Cross-session crash/finish: the recorded PID is no longer a running batch.
    # (False = confidently gone; None = unknown, leave it to the other checks.)
    if _pid_running_batch(pid) is False:
        logger.info("Agent-batch process %s is no longer running; clearing lock.",
                    pid)
        _clear_lock(lock_path)
        return False, None

    if time.time() - lock_mtime > BATCH_STALE_SECONDS:
        logger.warning("Agent-batch lock is stale (>%ds); clearing it.",
                       BATCH_STALE_SECONDS)
        _clear_lock(lock_path)
        return False, None

    return True, started


def start_agent_batch(provider, views=None):
    """Launch `python -m agent.batch` in the background. Returns (ok, message).

    Reuses THIS interpreter/venv and inherits the environment; ``provider`` pins
    the reasoning LLM (LLM_PROVIDER) for the run. Pass ``views`` (a list of view
    names) to re-run ONLY those — the "Retry failed views" path — instead of the
    full ~60-view batch. The lock is written BEFORE launching so a double-click
    can't spawn two batches, then rewritten with the child PID so ANY session can
    check whether the batch is still alive. Output is appended to today's batch
    log; the Popen handle is stored in session_state so this session detects
    completion without waiting for the stale timeout.
    """
    running, started = batch_in_progress()
    if running:
        return False, f"An agent batch is already running (started {started})."

    lock_path = _batch_lock_path()
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(now)

    try:
        env = _child_env(LLM_PROVIDER=provider)
        creationflags = _bg_creationflags()
        cmd = [sys.executable, "-m", "agent.batch", "--provider", provider]
        if views:
            cmd += ["--views", *views]
        scope = f"{len(views)} view(s)" if views else "all views"
        logf = open(_batch_log_path(), "a", encoding="utf-8")
        try:
            logf.write(f"\n===== Agent batch ({scope}) started {now} =====\n")
            logf.flush()
            # `-m agent.batch` resolves because cwd=HERE is src/ (agent is a
            # package under src/). Provider is also passed as a flag so a stale
            # env can't override it.
            proc = subprocess.Popen(
                cmd,
                cwd=HERE,
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
                close_fds=True,
            )
        finally:
            logf.close()
    except Exception as exc:
        _clear_lock(lock_path)
        logger.exception("Failed to launch agent batch")
        return False, f"Could not start agent batch: {exc}"

    # Record the PID (line 2) so any session can check liveness / crash without
    # this session's Popen handle. Best-effort — liveness gracefully degrades.
    try:
        with open(lock_path, "w", encoding="utf-8") as f:
            f.write(f"{now}\n{proc.pid}\n")
    except OSError:
        pass

    st.session_state["agent_batch_proc"] = proc
    st.session_state["agent_batch_started"] = now
    logger.info("Agent batch launched (%s, provider=%s, scope=%s, pid=%s)",
                now, provider, scope, proc.pid)
    return True, now
