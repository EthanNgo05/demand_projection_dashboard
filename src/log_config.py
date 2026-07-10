"""Shared, dependency-free helpers for date-organized log files.

Every log the app writes lands under ``logs/<YYYY-MM-DD>/`` at the repo root,
bucketed by the date each entry is written — so a long-running process (the
Streamlit dashboard) still files today's lines under today's folder, not the
folder for the day it started. Kept free of any streamlit/agent imports so both
dashboard.py and the agent package can import it.

Tests redirect all output by monkeypatching ``LOG_ROOT``.
"""

import logging
import os
import time

# <repo root>/logs — this module lives in src/, so climb one level to the repo
# root (the folder holding raw_inputs/, outputs/, logs/) before appending "logs".
LOG_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
)


def dated_log_path(filename, date_str=None):
    """Return ``logs/<YYYY-MM-DD>/<filename>``, creating the day's folder.

    ``date_str`` defaults to today's local date, so each entry lands in the
    folder matching the day it was written. ``LOG_ROOT`` is read at call time,
    so monkeypatching it (in tests) redirects the whole tree.
    """
    if date_str is None:
        date_str = time.strftime("%Y-%m-%d")
    folder = os.path.join(LOG_ROOT, date_str)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, filename)


class DateFolderHandler(logging.Handler):
    """Logging handler that appends each record to ``logs/<date>/<filename>``,
    where ``<date>`` is the record's own timestamp. Reopens the stream when the
    day rolls over, so a process running across midnight splits its lines into
    the correct daily folders.

    File output is best-effort: a read-only filesystem (some hosts) is swallowed
    so the app keeps running — pair this with a StreamHandler for console output,
    exactly how dashboard.py and agent/logging_util.py use it.
    """

    def __init__(self, filename):
        super().__init__()
        self._filename = filename
        self._date = None
        self._stream = None

    def emit(self, record):
        try:
            date_str = time.strftime("%Y-%m-%d", time.localtime(record.created))
            if self._stream is None or date_str != self._date:
                if self._stream is not None:
                    self._stream.close()
                self._stream = open(
                    dated_log_path(self._filename, date_str), "a", encoding="utf-8"
                )
                self._date = date_str
            self._stream.write(self.format(record) + "\n")
            self._stream.flush()
        except OSError:
            # Read-only filesystem: drop file output; the paired StreamHandler
            # still writes to the console.
            pass
        except Exception:  # noqa: BLE001 — logging must never crash the caller
            self.handleError(record)

    def close(self):
        try:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
        finally:
            super().close()
