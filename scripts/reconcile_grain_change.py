"""Explain, per SKU, why a view's forecast total changed when it became a sum.

Quick Projections used to show a total fit as ONE series per SKU from the
customer-summed history (top-down). It now shows the sum of the per-(SKU, customer)
forecasts (bottom-up), because a total that disagrees with the rows underneath it
cannot be used to place an order. On the live snapshot that moved the all-customers
total by about +20%, so every SKU that moved needs an auditable reason.

This writes one workbook with three sheets:

* ``totals``   — the headline before/after, and the change split by cause.
* ``by_sku``   — one row per SKU: old, new, delta, and how much of the delta is
                 recovered Orders-only demand vs. fit arithmetic.
* ``orphans``  — (SKU, customer) pairs carrying a forward plan but no recent demand
                 to forecast from. They are outside both totals; the Exceptions view
                 is where they get actioned.

The split is exact, not apportioned. "Recovered Orders-only demand" is the summed
forecast of the (SKU, customer) rows fit on Orders where the top-down fit resolved
POS for that SKU — demand the old total could not see at all, because
``aggregate_to_sku_week`` sums POS and Orders into separate columns and the fit then
picks ONE of them. Whatever is left over is the fit arithmetic: per-series span
denominators, the zero floor, and per-row rounding.

    python scripts/reconcile_grain_change.py
    python scripts/reconcile_grain_change.py --view "All Customers - US (LBC+NJ)"
    python scripts/reconcile_grain_change.py --model exponential_smoothing

Reads the newest snapshot in ``raw_inputs/demand_projections`` and writes to
``outputs/`` (gitignored). The on-disk forecast cache is left ENABLED here — unlike
bench_dashboard.py this is measuring numbers, not time, so a warm cache only makes
it faster.
"""

import argparse
import contextlib
import io
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

from dashboard_app.compute import (  # noqa: E402
    _region_frame, attach_current_projection, compute_by_customer_frames,
    compute_view, roll_up_summary, roll_up_to_sku_week,
)
from dashboard_app.config import ALL_CUSTOMERS_VIEW, MODEL_OPTIONS, region_from_view  # noqa: E402
from dashboard_app.datasources import discover_raw_files, load_raw_from_path  # noqa: E402
from dashboard_app.pipeline import load_pipeline  # noqa: E402

UPD = "Updated Projection Average"
CUR = "Current Projection Average"
OUT_DIR = os.path.join(REPO_ROOT, "outputs")


def _model_path(name):
    """Resolve a model by MODEL_OPTIONS label or by file stem (e.g. 'regression')."""
    if name is None:
        return next(iter(MODEL_OPTIONS.values()))
    if name in MODEL_OPTIONS:
        return MODEL_OPTIONS[name]
    for path in MODEL_OPTIONS.values():
        if os.path.splitext(os.path.basename(path))[0] == name:
            return path
    raise SystemExit(
        f"unknown model {name!r}; try one of "
        + ", ".join(sorted(os.path.splitext(os.path.basename(p))[0]
                           for p in MODEL_OPTIONS.values()))
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--view", default=ALL_CUSTOMERS_VIEW,
                    help="view to reconcile (default: the combined view)")
    ap.add_argument("--model", default=None,
                    help="MODEL_OPTIONS label or model file stem")
    ap.add_argument("--out", default=None, help="output .xlsx path")
    args = ap.parse_args()

    view = args.view
    if view != ALL_CUSTOMERS_VIEW and region_from_view(view) is None:
        raise SystemExit(
            f"{view!r} is a single customer group — its total already IS the part, "
            "so there is nothing to reconcile. Pass the combined view or a region "
            "rollup."
        )

    mp = _model_path(args.model)
    P = load_pipeline(mp)
    snap_date, raw = sorted(discover_raw_files())[-1]
    df = load_raw_from_path(raw, os.path.getmtime(raw), mp)
    today_ts = pd.Timestamp(snap_date)
    anchors = P.week_anchors(today_ts)
    region_all = region_from_view(view)
    src = df if region_all is None else _region_frame(df, P, region_all)

    print(f"snapshot : {os.path.basename(raw)}  ({len(df):,} rows)")
    print(f"view     : {view}")
    print(f"model    : {os.path.basename(mp)}")
    print("computing both grains…", flush=True)

    # Model files print a progress note per fit; keep the console readable.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        old_summary, old_weekly, _ = compute_view(df, view, today_ts, mp)
        by_cust, wbg, abg = compute_by_customer_frames(src, today_ts, mp)

    if old_summary is None or by_cust is None:
        raise SystemExit("nothing forecast for this view on this snapshot")

    weekly_all = roll_up_to_sku_week(wbg, ["projected_pos"], min_count=0)
    agg_all = roll_up_to_sku_week(abg, ["POS", "Orders", "Projection"])
    horizon = weekly_all["WeekDate"]
    by_cust = attach_current_projection(
        by_cust, abg, horizon, ["Customer Grouping", "SKU"])
    new_summary = roll_up_summary(by_cust, agg_all, anchors, view)

    # --- Cause 1: rows fit on Orders that the top-down fit could not see ----
    old_src = dict(zip(old_summary["SKU"].astype(str), old_summary["Data Source"]))
    rows = by_cust.copy()
    rows["SKU"] = rows["SKU"].astype(str)
    invisible = rows[
        (rows["Data Source"] == "Orders")
        & (rows["SKU"].map(old_src) == "POS")
    ]
    recovered = (
        pd.to_numeric(invisible[UPD], errors="coerce")
        .groupby(invisible["SKU"]).sum()
    )

    # --- Per-SKU ledger ----------------------------------------------------
    old_u = pd.to_numeric(old_summary.set_index(
        old_summary["SKU"].astype(str))[UPD], errors="coerce")
    new_u = pd.to_numeric(new_summary.set_index(
        new_summary["SKU"].astype(str))[UPD], errors="coerce")
    old_c = pd.to_numeric(old_summary.set_index(
        old_summary["SKU"].astype(str))[CUR], errors="coerce")
    new_c = pd.to_numeric(new_summary.set_index(
        new_summary["SKU"].astype(str))[CUR], errors="coerce")
    desc = new_summary.set_index(new_summary["SKU"].astype(str))["Description"]

    skus = sorted(set(old_u.index) | set(new_u.index))
    ledger = pd.DataFrame(index=pd.Index(skus, name="SKU"))
    ledger["Description"] = desc.reindex(skus)
    ledger["Customers"] = rows.groupby("SKU")["Customer Grouping"].nunique().reindex(skus)
    ledger["Updated (old, top-down)"] = old_u.reindex(skus)
    ledger["Updated (new, sum of parts)"] = new_u.reindex(skus)
    ledger["Updated delta"] = (
        ledger["Updated (new, sum of parts)"].fillna(0)
        - ledger["Updated (old, top-down)"].fillna(0)
    )
    ledger["...from recovered Orders-only demand"] = recovered.reindex(skus).fillna(0.0)
    ledger["...from fit arithmetic"] = (
        ledger["Updated delta"] - ledger["...from recovered Orders-only demand"]
    )
    ledger["Updated delta %"] = np.where(
        ledger["Updated (old, top-down)"].fillna(0) > 0,
        ledger["Updated delta"] / ledger["Updated (old, top-down)"] * 100.0,
        np.nan,
    ).round(1)
    ledger["Existing plan (old)"] = old_c.reindex(skus)
    ledger["Existing plan (new, fixed denominator)"] = new_c.reindex(skus)
    ledger["Existing plan delta"] = (
        ledger["Existing plan (new, fixed denominator)"].fillna(0)
        - ledger["Existing plan (old)"].fillna(0)
    )
    ledger["Mixed source now"] = (
        new_summary.set_index(new_summary["SKU"].astype(str))["Data Source"]
        .reindex(skus).eq("Mixed (POS + Orders)")
    )
    ledger = ledger.sort_values("Updated delta", key=abs, ascending=False)

    # --- Orphaned plans: a forward plan, but nothing to forecast from -------
    fwd = abg[pd.to_datetime(abg["WeekDate"]).isin(pd.to_datetime(horizon).unique())]
    planned = (
        pd.to_numeric(fwd["Projection"], errors="coerce")
        .groupby([fwd["Customer Grouping"].astype(str), fwd["SKU"].astype(str)])
        .sum(min_count=1)
        / max(pd.to_datetime(horizon).nunique(), 1)
    )
    planned = planned[planned.notna() & (planned != 0)]
    have = set(zip(rows["Customer Grouping"].astype(str), rows["SKU"]))
    orphans = (
        planned[[k not in have for k in planned.index]]
        .rename("Existing plan (units/wk)")
        .reset_index()
        .rename(columns={"level_0": "Customer Grouping", "level_1": "SKU"})
        .sort_values("Existing plan (units/wk)", ascending=False)
    )

    # --- Totals ------------------------------------------------------------
    old_total, new_total = old_u.sum(), new_u.sum()
    rec_total = recovered.sum()
    totals = pd.DataFrame([
        ("Updated forecast — old (top-down fit)", old_total),
        ("Updated forecast — new (sum of parts)", new_total),
        ("Change", new_total - old_total),
        ("  ...recovered Orders-only customer demand", rec_total),
        ("  ...fit arithmetic (span / floor / rounding)",
         (new_total - old_total) - rec_total),
        ("Change %", (new_total - old_total) / old_total * 100 if old_total else np.nan),
        ("", np.nan),
        ("Existing plan — old (ragged denominator)", old_c.sum()),
        ("Existing plan — new (fixed 15-week denominator)", new_c.sum()),
        ("", np.nan),
        ("SKUs in view", float(len(skus))),
        ("SKUs whose total moved", float((ledger["Updated delta"] != 0).sum())),
        ("SKUs now mixed-source", float(ledger["Mixed source now"].sum())),
        ("Orphaned (SKU, customer) plans", float(len(orphans))),
        ("Orphaned plan units/wk", float(orphans["Existing plan (units/wk)"].sum())
         if len(orphans) else 0.0),
    ], columns=["Measure", "Value"])

    out = args.out or os.path.join(
        OUT_DIR,
        f"grain_reconciliation_{view.replace('/', '-').replace(' ', '_')}"
        f"_{snap_date}.xlsx",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        totals.to_excel(w, sheet_name="totals", index=False)
        ledger.reset_index().to_excel(w, sheet_name="by_sku", index=False)
        orphans.to_excel(w, sheet_name="orphans", index=False)

    print()
    print(totals.to_string(index=False, na_rep=""))
    print()
    print("worst 10 SKUs by absolute change:")
    cols = ["Description", "Customers", "Updated (old, top-down)",
            "Updated (new, sum of parts)", "Updated delta",
            "...from recovered Orders-only demand", "Updated delta %"]
    print(ledger[cols].head(10).to_string(na_rep=""))
    print()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
