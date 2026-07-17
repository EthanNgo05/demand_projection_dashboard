import os
import sys

import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The app's Python lives under src/ (dashboard, log_config, extract, agent, ...);
# put it on the path so tests can import those top-level modules and packages.
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
FIXTURE_RAW = os.path.join(FIXTURE_DIR, "all_demand_projections_2026-07-01.xlsx")


# --------------------------------------------------------------------------
# `slow` marker: the full-matrix parity suites take minutes (they backtest
# every model across many views). Skip them by default so `pytest` stays quick;
# `pytest --runslow` runs everything. Registered in pytest.ini.
# --------------------------------------------------------------------------
def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="run the slow full-matrix parity suites too",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow: pass --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(autouse=True)
def _isolate_agent_outputs(tmp_path_factory, monkeypatch):
    """Keep every test's agent side effects off the real ``outputs/`` and ``logs/``.

    ``build_graph()`` ends in the ``publish`` node, which writes
    ``outputs/agent_summary_<view>.json`` and appends to ``logs/<date>/app.log``.
    The parity/selection suites (test_phase2/3/6) invoke the *whole* graph on the
    tiny synthetic fixture, so without this they overwrite the real agent
    summaries with fixture data (4 SKUs, 9 weeks of history) and pollute the app
    log — which is exactly how the real dashboard once showed a bogus
    "history too short" summary for a healthy view. Redirect both to a temp dir
    for every test. Tests that pin their own ``OUTPUT_DIR`` / ``LOG_ROOT``
    (test_phase5_publish) still win: their monkeypatch runs after this fixture.
    """
    monkeypatch.setattr(
        "agent.nodes.publish.OUTPUT_DIR", str(tmp_path_factory.mktemp("outputs"))
    )
    monkeypatch.setattr(
        "log_config.LOG_ROOT", str(tmp_path_factory.mktemp("logs"))
    )


@pytest.fixture(scope="session")
def sample_raw_path():
    """Path to the small deterministic raw workbook (built on demand, seeded)."""
    if not os.path.exists(FIXTURE_RAW):
        sys.path.insert(0, FIXTURE_DIR)
        from make_fixture import build

        build(FIXTURE_RAW)
    return FIXTURE_RAW


@pytest.fixture(scope="session")
def sample_cleaned_df(sample_raw_path):
    """The fixture workbook read + cleaned exactly as ingest does."""
    from agent import data_io

    return data_io.load_raw(sample_raw_path)


# --------------------------------------------------------------------------
# Phase 4: fake LLM + hand-built states for the reasoning nodes
# --------------------------------------------------------------------------


class RecordingFakeLLM:
    """Minimal stand-in for any LangChain chat model: .invoke(prompt).content.

    Records every prompt so tests can assert on *inputs* (e.g. the anomaly
    table was capped), not exact LLM text. Provider-agnostic by construction —
    it replaces agent.llm.get_llm, which is the single seam both the Claude
    and local providers flow through.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        text = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]

        class _Msg:
            content = text

        return _Msg()


@pytest.fixture
def fake_llm(monkeypatch):
    """Install a fake LLM behind agent.llm.get_llm; returns the model so tests
    can inspect .prompts. Usage: model = fake_llm(["response 1", ...])."""

    def _install(responses):
        model = RecordingFakeLLM(responses)
        monkeypatch.setattr(
            "agent.llm.get_llm", lambda temperature=0, provider=None: model
        )
        return model

    return _install


def _summary_df(n_rows):
    """Summary frame with the columns the reasoning nodes actually read."""
    rows = []
    for i in range(n_rows):
        recent = 100.0 + i
        projected = recent * (3.0 if i == 0 else (0.5 if i == 1 else 1.02))
        rows.append(
            {
                "SKU": f"SKU-{i + 1:03d}",
                "Description": f"Widget {i + 1}",
                "8 Week POS/Orders Average": recent,
                "Updated Projection Average": projected,
                "Weeks with data": 26,
            }
        )
    return pd.DataFrame(rows)


def _state(n_rows, view="TEST GROUP"):
    return {
        "view": view,
        "best_model": "8-Week Moving Average",
        "results": {"8-Week Moving Average": {"summary_df": _summary_df(n_rows), "mase": 1.23}},
        "confidence_flag": False,
        "errors": [],
    }


@pytest.fixture
def sample_state_with_summary():
    """Post-select state with a small summary (SKU-001 jumps, SKU-002 drops)."""
    return _state(10)


@pytest.fixture
def large_summary_state():
    """ALL CUSTOMERS-sized state: 400+ SKU rows to prove the prompt is capped."""
    return _state(420, view="All customers (combined)")
