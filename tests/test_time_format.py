"""Display formatting for user-facing timestamps (config.fmt_when).

``fmt_when`` replaced four hand-rolled formatters that had drifted apart (two
24-hour, two 12-hour, one of which forgot to strip the leading zero), and then
absorbed ``fmt_clock``, which rendered the same times without a date. The point
of the tests is to pin the things that drift: the 12/24 boundary, the input
shapes each old call site used, and the never-raise contract.
"""
import datetime

import pandas as pd
import pytest

from dashboard_app.config import fmt_when


# --------------------------------------------------------------------------
# fmt_when — always an absolute ISO date plus a 12-hour clock
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ts, expected", [
    ("2026-08-11 15:15:54", "2026-08-11 3:15 PM"),
    ("2026-08-11 09:41:26", "2026-08-11 9:41 AM"),
    # The leading zero on %I must be stripped: "3:15 AM", never "03:15 AM".
    ("2026-08-11 03:15:00", "2026-08-11 3:15 AM"),
    # Midnight and noon are where the strip would leave an EMPTY hour, since
    # %I renders both as "12" and lstrip("0") is a no-op only by luck.
    ("2026-08-12 00:05:00", "2026-08-12 12:05 AM"),
    ("2026-08-12 12:30:00", "2026-08-12 12:30 PM"),
    # Seconds are always dropped — no rounding, just truncation.
    ("2026-08-12 12:30:59", "2026-08-12 12:30 PM"),
    # The year is ALWAYS shown. This case used to be the "different year"
    # special branch; now it is an ordinary one, which is the whole point.
    ("2025-12-01 09:41:26", "2025-12-01 9:41 AM"),
])
def test_fmt_when_renders_iso_date_and_12_hour_time(ts, expected):
    assert fmt_when(ts) == expected


def test_fmt_when_is_absolute_and_never_relative():
    """No "today"/"yesterday" branch, however recent the timestamp.

    Relative wording made the reader do date arithmetic to answer "which
    snapshot is this?", and it went stale in a tab left open past midnight.
    """
    just_before_midnight = datetime.datetime(2026, 8, 11, 23, 59, 0)
    assert fmt_when(just_before_midnight) == "2026-08-11 11:59 PM"
    assert fmt_when(datetime.datetime.now()).startswith(
        datetime.date.today().strftime("%Y-%m-%d")
    )


def test_fmt_when_accepts_every_shape_the_app_produces():
    """One funnel for the four mechanisms that used to format timestamps."""
    naive = datetime.datetime(2026, 8, 11, 15, 15, 54)
    epoch = naive.timestamp()          # time.time() / os.path.getmtime()
    assert fmt_when(naive) == "2026-08-11 3:15 PM"                   # datetime
    assert fmt_when(pd.Timestamp(naive)) == "2026-08-11 3:15 PM"     # pd.Timestamp
    assert fmt_when(epoch) == "2026-08-11 3:15 PM"                   # epoch float
    assert fmt_when("2026-08-11T15:15:54") == "2026-08-11 3:15 PM"   # ISO string
    assert fmt_when("2026-08-11 15:15:54") == "2026-08-11 3:15 PM"   # lock-file string


# --------------------------------------------------------------------------
# The never-raise contract
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [None, "", "garbage", float("nan"), pd.NaT, [], {}])
def test_unparseable_input_degrades_to_a_dash(bad):
    """A cosmetic timestamp must never be able to take down a render path.

    The lock-file string in particular is read off a network share and can be
    missing or half-written, and it is interpolated into a banner that renders
    before the page's error boundary would help.
    """
    assert fmt_when(bad) == "—"


def test_booleans_are_not_mistaken_for_epoch_seconds():
    """bool is a subclass of int — True must not format as 1970-01-01."""
    assert fmt_when(True) == "—"
    assert fmt_when(False) == "—"
