"""Unit tests for data_io.compute_missing_pos_orders.

Active SKUs (Parts included) that sold in a region they're 'Active in' within the
past 3 months but have since gone silent. Combos that never had data or that last
sold more than 3 months ago are deliberately excluded (no recent demand to be
"missing").
"""
import numpy as np
import pandas as pd
import pytest

from agent import data_io

P = data_io.default_pipeline()  # first configured model (region_for_group, week_anchors, ...)

TODAY = pd.Timestamp.today().normalize()
_, LCW, CWS = P.week_anchors(TODAY)        # last completed week; current week start
# The 3-month floor is measured from the CURRENT week: sold on/after CUTOFF stays,
# a week older drops. (BT1028 regression: last sale exactly LCW-3mo must be OUT.)
CUTOFF = CWS - pd.DateOffset(months=3)     # earliest last-sale week still in scope
JUST_IN = CUTOFF                           # sold at the cutoff -> kept
JUST_OUT = CUTOFF - pd.Timedelta(weeks=1)  # one week older -> dropped
EARLIEST = LCW - pd.Timedelta(weeks=10)    # snapshot floor
GONE_LAST = LCW - pd.Timedelta(weeks=3)    # last week the "gone silent" combo had data
LONG_DEAD = LCW - pd.Timedelta(weeks=20)   # last sale >3mo ago -> excluded by the floor


def _row(sku, cust, week, pos=np.nan, orders=np.nan):
    return {
        "SKU": sku, "Description": "d", "Customer": cust, "WeekDate": week,
        "POS": pos, "Orders": orders, "Projection": np.nan,
        "Customer Grouping": P.COMBINED_GROUPING.get(cust, cust),
    }


def _demand_df():
    rows = [
        # SKUGONE @ TARGET-HQ (US): data at EARLIEST + GONE_LAST, blank row at LCW
        _row("SKUGONE", "TARGET-HQ", EARLIEST, pos=5),
        _row("SKUGONE", "TARGET-HQ", GONE_LAST, pos=2),
        _row("SKUGONE", "TARGET-HQ", LCW),                 # exists but no data
        # SKUGONE @ AMAZON-EU (EU): stale, but SKU is US-only -> must NOT flag
        _row("SKUGONE", "AMAZON-EU", GONE_LAST),
        # SKUORD @ TARGET-HQ (US): last sold via Orders (no POS) -> Orders source
        _row("SKUORD", "TARGET-HQ", GONE_LAST, orders=9),
        # SKUZERO @ TARGET-HQ (US): real sale (POS=6) at GONE_LAST, then a trailing
        # 0 at LCW. The 0 is a zero-sale week -> still flagged, Last Value = 6, and
        # the gap starts after GONE_LAST (not after the 0).
        _row("SKUZERO", "TARGET-HQ", GONE_LAST, pos=6),
        _row("SKUZERO", "TARGET-HQ", LCW, pos=0),
        # SKUZONLY @ TARGET-HQ (US): only ever a 0 -> no real sale -> excluded like
        # a never-sold combo.
        _row("SKUZONLY", "TARGET-HQ", GONE_LAST, pos=0),
        # SKUNEG @ TARGET-HQ (US): real sale (Orders=8) at GONE_LAST, then a return
        # (Orders=-2) at LCW. The negative is not a sale -> still flagged, Last
        # Value = 8, gap starts after GONE_LAST.
        _row("SKUNEG", "TARGET-HQ", GONE_LAST, orders=8),
        _row("SKUNEG", "TARGET-HQ", LCW, orders=-2),
        # SKUNEGONLY @ TARGET-HQ (US): only ever a return (negative) -> no real sale
        # -> excluded.
        _row("SKUNEGONLY", "TARGET-HQ", GONE_LAST, pos=-5),
        # SKULIVE @ TARGET-HQ: has data AT the reference week -> not missing
        _row("SKULIVE", "TARGET-HQ", LCW, orders=7),
        # SKUNEVER @ TARGET-HQ: appears but never any data -> never part of the
        # assortment -> excluded (no recent demand to be "missing")
        _row("SKUNEVER", "TARGET-HQ", EARLIEST),
        # SKUDEAD @ TARGET-HQ: last sold >3mo ago -> long-dead -> excluded
        _row("SKUDEAD", "TARGET-HQ", LONG_DEAD, pos=4),
        # SKUIN @ TARGET-HQ: last sold exactly at the 3-month cutoff -> kept
        _row("SKUIN", "TARGET-HQ", JUST_IN, pos=3),
        # SKUOUT @ TARGET-HQ: last sold one week past the cutoff -> dropped
        # (the BT1028 regression: LCW-3mo is one week too early to qualify)
        _row("SKUOUT", "TARGET-HQ", JUST_OUT, pos=3),
        # SKUPART (a Part) @ TARGET-HQ: no data -> never part of assortment -> excluded
        _row("SKUPART", "TARGET-HQ", GONE_LAST),
        # SKULSPART (LS-prefixed) @ TARGET-HQ: no data -> excluded (never had data)
        _row("LSPART1", "TARGET-HQ", GONE_LAST),
        # SKUDISC @ TARGET-HQ: discontinued in Plytix -> never a candidate
        _row("SKUDISC", "TARGET-HQ", GONE_LAST),
    ]
    df = pd.DataFrame(rows)
    df["WeekDate"] = pd.to_datetime(df["WeekDate"])
    return df


def _plytix():
    return pd.DataFrame([
        {"SKU": "SKUGONE",  "SKU Status": "Active",       "SKU Type": "Product", "Active in": "US"},
        {"SKU": "SKUORD",   "SKU Status": "Active",       "SKU Type": "Product", "Active in": "US"},
        {"SKU": "SKUZERO",  "SKU Status": "Active",       "SKU Type": "Product", "Active in": "US"},
        {"SKU": "SKUZONLY", "SKU Status": "Active",       "SKU Type": "Product", "Active in": "US"},
        {"SKU": "SKUNEG",   "SKU Status": "Active",       "SKU Type": "Product", "Active in": "US"},
        {"SKU": "SKUNEGONLY","SKU Status": "Active",      "SKU Type": "Product", "Active in": "US"},
        {"SKU": "SKULIVE",  "SKU Status": "Active",       "SKU Type": "Product", "Active in": "US"},
        {"SKU": "SKUNEVER", "SKU Status": "Active",       "SKU Type": "Product", "Active in": "US"},
        {"SKU": "SKUDEAD",  "SKU Status": "Active",       "SKU Type": "Product", "Active in": "US"},
        {"SKU": "SKUIN",    "SKU Status": "Active",       "SKU Type": "Product", "Active in": "US"},
        {"SKU": "SKUOUT",   "SKU Status": "Active",       "SKU Type": "Product", "Active in": "US"},
        {"SKU": "SKUPART",  "SKU Status": "Active",       "SKU Type": "Part",    "Active in": "US"},
        {"SKU": "LSPART1",  "SKU Status": "Active",       "SKU Type": "Part",    "Active in": "US"},
        {"SKU": "SKUDISC",  "SKU Status": "Discontinued", "SKU Type": "Product", "Active in": "US"},
    ])


def test_flags_recently_gone_silent_only():
    out = data_io.compute_missing_pos_orders(_demand_df(), _plytix(), P, anchors=P.week_anchors(TODAY))

    assert list(out.columns) == data_io.MISSING_POS_COLS
    assert out["Missing Weeks"].isna().sum() == 0

    flagged = set(out["SKU"])
    assert "SKUGONE" in flagged      # sold within the past 3 months, then went silent
    assert "SKUZERO" in flagged      # real sale then a trailing 0 -> still flagged
    assert "SKUNEG" in flagged       # real sale then a return (negative) -> flagged
    assert "SKUNEVER" not in flagged  # never had data -> never part of assortment
    assert "SKUDEAD" not in flagged   # last sold >3mo ago -> long-dead
    assert "SKUIN" in flagged         # sold exactly at the 3-month cutoff -> kept
    assert "SKUOUT" not in flagged    # one week past the cutoff -> dropped
    assert "SKUZONLY" not in flagged  # only ever a 0 -> no real sale
    assert "SKUNEGONLY" not in flagged  # only ever a return (negative) -> no real sale
    assert "SKUPART" not in flagged   # never had data (Part)
    assert "LSPART1" not in flagged   # never had data
    assert "SKULIVE" not in flagged  # has data at the reference week
    assert "SKUDISC" not in flagged  # not an active SKU


def test_region_restricted_to_active_in():
    out = data_io.compute_missing_pos_orders(_demand_df(), _plytix(), P, anchors=P.week_anchors(TODAY))
    # SKUGONE is US-only: its stale EU (AMAZON-EU) combo must not appear.
    gone = out[out["SKU"] == "SKUGONE"]
    assert list(gone["Region Code"]) == ["US"]
    assert "EU" not in set(out["Region Code"])


def test_missing_weeks_counts_gap_since_last_sale():
    out = data_io.compute_missing_pos_orders(_demand_df(), _plytix(), P, anchors=P.week_anchors(TODAY))

    gone = out[(out["SKU"] == "SKUGONE") & (out["Customer"] == "TARGET-HQ")].iloc[0]
    # Gap = week after GONE_LAST .. LCW inclusive = 3 weeks.
    assert gone["Missing Weeks"] == 3
    assert pd.Timestamp(gone["First Missing Week"]) == GONE_LAST + pd.Timedelta(weeks=1)
    assert pd.Timestamp(gone["Last Missing Week"]) == LCW


def test_reports_last_source_and_value():
    out = data_io.compute_missing_pos_orders(_demand_df(), _plytix(), P, anchors=P.week_anchors(TODAY))

    # SKUGONE's last data week (GONE_LAST) carried POS=2.
    gone = out[out["SKU"] == "SKUGONE"].iloc[0]
    assert gone["Data Source"] == "POS"
    assert gone["Last Value"] == 2

    # SKUORD only ever had Orders -> Orders source, value from its last week.
    ord_row = out[out["SKU"] == "SKUORD"].iloc[0]
    assert ord_row["Data Source"] == "Orders"
    assert ord_row["Last Value"] == 9

    # SKUZERO: trailing 0 is ignored -> Last Value is the real sale (POS=6) and the
    # gap starts after GONE_LAST, not after the 0 at LCW.
    zero = out[out["SKU"] == "SKUZERO"].iloc[0]
    assert zero["Data Source"] == "POS"
    assert zero["Last Value"] == 6
    assert pd.Timestamp(zero["First Missing Week"]) == GONE_LAST + pd.Timedelta(weeks=1)
    assert zero["Missing Weeks"] == 3

    # SKUNEG: trailing return (negative) is ignored -> Last Value is the real sale.
    neg = out[out["SKU"] == "SKUNEG"].iloc[0]
    assert neg["Data Source"] == "Orders"
    assert neg["Last Value"] == 8
    assert pd.Timestamp(neg["First Missing Week"]) == GONE_LAST + pd.Timedelta(weeks=1)


def test_none_when_plytix_lacks_active_in():
    df = _demand_df()
    assert data_io.compute_missing_pos_orders(df, _plytix().drop(columns=["Active in"]), P,
                                              anchors=P.week_anchors(TODAY)) is None
    assert data_io.compute_missing_pos_orders(df, None, P, anchors=P.week_anchors(TODAY)) is None
    assert data_io.compute_missing_pos_orders(df, _plytix(), P, anchors=None) is None
