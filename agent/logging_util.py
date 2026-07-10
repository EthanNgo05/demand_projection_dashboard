"""Shared file logger for the agent — same logs.txt, same format as dashboard.py.

Uses a distinct logger name ("demand_agent") but the identical formatter and
LOG_PATH as dashboard.py (dashboard.py:63-87), so agent decisions land in the
same audit trail as the manual Autofit runs already there. Kept separate from
dashboard.py because the agent must never import streamlit.
"""

import logging
import os

LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs.txt"
)

logger = logging.getLogger("demand_agent")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    try:
        _fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
        _fh.setFormatter(_fmt)
        logger.addHandler(_fh)
    except OSError:
        # Read-only filesystem: console only, same fallback as dashboard.py.
        pass
    _sh = logging.StreamHandler()
    _sh.setFormatter(_fmt)
    logger.addHandler(_sh)
    logger.propagate = False
