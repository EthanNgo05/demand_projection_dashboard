"""Phase 1 foundation tests — see docs/agentic_workflow/01-foundations.md.

Run with: pytest tests/test_phase1_foundations.py -v
"""

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def test_state_and_config_importable():
    from agent.config import MODEL_OPTIONS
    from agent.state import AgentState  # noqa: F401

    assert "Holt's Exponential Smoothing" in MODEL_OPTIONS


def test_model_option_paths_exist():
    from agent.config import MODEL_OPTIONS

    assert MODEL_OPTIONS, "MODEL_OPTIONS is empty — no model files found"
    for label, path in MODEL_OPTIONS.items():
        assert os.path.exists(path), f"{label} -> {path} missing"


def test_all_customers_view_matches_dashboard():
    # The combined-view label must match dashboard.py exactly (dashboard.py:122);
    # a drift here silently compares/filters the wrong view in every later phase.
    from agent.config import ALL_CUSTOMERS_VIEW

    with open(os.path.join(REPO_ROOT, "dashboard.py"), encoding="utf-8") as f:
        dashboard_src = f.read()
    assert f'ALL_CUSTOMERS_VIEW = "{ALL_CUSTOMERS_VIEW}"' in dashboard_src


def test_no_streamlit_import_required():
    # import agent.state/config in a fresh subprocess and confirm streamlit
    # never gets imported as a side effect
    code = (
        "import sys; import agent.state; import agent.config; "
        "assert 'streamlit' not in sys.modules, sorted(sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr


def test_env_key_not_hardcoded():
    # crude guard: no file in agent/ should contain a literal sk-ant- style key
    agent_dir = os.path.join(REPO_ROOT, "agent")
    for root, _dirs, files in os.walk(agent_dir):
        for name in files:
            if not name.endswith(".py"):
                continue
            with open(os.path.join(root, name), encoding="utf-8") as f:
                assert "sk-ant-" not in f.read(), f"possible hardcoded key in {name}"
