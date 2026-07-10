"""Shared file logger for the agent — same daily log files as dashboard.py.

Uses a distinct logger name ("demand_agent") but the identical formatter and
DateFolderHandler as dashboard.py, so agent decisions land in the same
``logs/<date>/app.log`` audit trail as the manual Autofit runs already there.
Kept separate from dashboard.py because the agent must never import streamlit.
"""

import logging

from log_config import DateFolderHandler

logger = logging.getLogger("demand_agent")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    # File output is best-effort (read-only hosts): the handler swallows OSError
    # internally, same fallback as dashboard.py.
    _fh = DateFolderHandler("app.log")
    _fh.setFormatter(_fmt)
    logger.addHandler(_fh)
    _sh = logging.StreamHandler()
    _sh.setFormatter(_fmt)
    logger.addHandler(_sh)
    logger.propagate = False
