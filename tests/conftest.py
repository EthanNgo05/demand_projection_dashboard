import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
FIXTURE_RAW = os.path.join(FIXTURE_DIR, "all_demand_projections_2026-07-01.xlsx")


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
