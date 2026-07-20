"""Unit tests for data_io.compute_missing_pos_orders.

Mirrors the missing_pos.ipynb logic: active SKUs (Parts included) with no recent
POS/Orders data in a region they're 'Active in', measured over full history.
"""
import numpy as np
import pandas as pd
import pytest

from agent import data_io

P = data_io.default_pipeline()  # first configured model (region_for_group, week_anchors, ...)

TODAY = pd.Timestamp.today().normalize()
_, LCW, _ = P.week_anchors(TODAY)          # last completed week (the reference)
EARLIEST = LCW - pd.Timedelta(weeks=10)    # snapshot floor
GONE_LAST = LCW - pd.Timedelta(weeks=3)    # last week the "gone silent" combo had data


def _row(sku, cust, week, pos=np.nan, orders=np.nan):
    return {
        "SKU": sku, "Description": "d", "CUSTNMBR": cust, "WeekDate": week,
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
        # SKULIVE @ TARGET-HQ: has data AT the reference week -> not missing
        _row("SKULIVE", "TARGET-HQ", LCW, orders=7),
        # SKUNEVER @ TARGET-HQ: appears but never any data -> flagged from EARLIEST
        _row("SKUNEVER", "TARGET-HQ", EARLIEST),
        # SKUPART (a Part) @ TARGET-HQ: no data -> Parts are included
        _row("SKUPART", "TARGET-HQ", GONE_LAST),
        # SKULSPART (LS-prefixed) @ TARGET-HQ: no data -> LS not excluded here
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
        {"SKU": "SKULIVE",  "SKU Status": "Active",       "SKU Type": "Product", "Active in": "US"},
        {"SKU": "SKUNEVER", "SKU Status": "Active",       "SKU Type": "Product", "Active in": "US"},
        {"SKU": "SKUPART",  "SKU Status": "Active",       "SKU Type": "Part",    "Active in": "US"},
        {"SKU": "LSPART1",  "SKU Status": "Active",       "SKU Type": "Part",    "Active in": "US"},
        {"SKU": "SKUDISC",  "SKU Status": "Discontinued", "SKU Type": "Product", "Active in": "US"},
    ])


def test_flags_gone_silent_and_never_had_data():
    out = data_io.compute_missing_pos_orders(_demand_df(), _plytix(), P, anchors=P.week_anchors(TODAY))

    assert list(out.columns) == data_io.MISSING_POS_COLS
    assert out["Missing Weeks"].isna().sum() == 0

    flagged = set(out["SKU"])
    assert "SKUGONE" in flagged
    assert "SKUNEVER" in flagged
    assert "SKUPART" in flagged      # Parts included
    assert "LSPART1" in flagged      # LS not excluded
    assert "SKULIVE" not in flagged  # has data at the reference week
    assert "SKUDISC" not in flagged  # not an active SKU


def test_region_restricted_to_active_in():
    out = data_io.compute_missing_pos_orders(_demand_df(), _plytix(), P, anchors=P.week_anchors(TODAY))
    # SKUGONE is US-only: its stale EU (AMAZON-EU) combo must not appear.
    gone = out[out["SKU"] == "SKUGONE"]
    assert list(gone["Location"]) == ["US"]
    assert "EU" not in set(out["Location"])


def test_missing_weeks_counts_full_gap():
    out = data_io.compute_missing_pos_orders(_demand_df(), _plytix(), P, anchors=P.week_anchors(TODAY))

    gone = out[(out["SKU"] == "SKUGONE") & (out["CUSTNMBR"] == "TARGET-HQ")].iloc[0]
    # Gap = week after GONE_LAST .. LCW inclusive = 3 weeks.
    assert gone["Missing Weeks"] == 3
    assert pd.Timestamp(gone["First Missing Week"]) == GONE_LAST + pd.Timedelta(weeks=1)
    assert pd.Timestamp(gone["Last Missing Week"]) == LCW

    never = out[out["SKU"] == "SKUNEVER"].iloc[0]
    # Never had data -> counted from the earliest snapshot week (EARLIEST) = 11.
    assert pd.Timestamp(never["First Missing Week"]) == EARLIEST
    assert never["Missing Weeks"] == 11


def test_none_when_plytix_lacks_active_in():
    df = _demand_df()
    assert data_io.compute_missing_pos_orders(df, _plytix().drop(columns=["Active in"]), P,
                                              anchors=P.week_anchors(TODAY)) is None
    assert data_io.compute_missing_pos_orders(df, None, P, anchors=P.week_anchors(TODAY)) is None
    assert data_io.compute_missing_pos_orders(df, _plytix(), P, anchors=None) is None
