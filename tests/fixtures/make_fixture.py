"""Generate the small, deterministic raw-workbook fixture used by Phase 2 tests.

Mimics the PowerBI export layout: two banner rows, headers on Excel row 3
(``pd.read_excel(path, header=2)``), one row per SKU-week-customer.

Data window is anchored to the pinned test date TODAY = 2026-07-01 (a
Wednesday): historical Sunday-start weeks 2026-04-26 .. 2026-06-21 carry
POS/Orders/Projection; future weeks 2026-06-28 .. 2026-08-16 carry Projection
only, like a real export. Seeded RNG -> regenerating produces identical data.

Run directly to (re)build:  python tests/fixtures/make_fixture.py
"""

import os

import numpy as np
import pandas as pd

FIXTURE_NAME = "all_demand_projections_2026-07-01.xlsx"

# (custnmbr, group note)  — AMAZON-DC + AMAZON-DS fold into the AMAZON-DC
# group; Others - UK is in CUSTOMERS_TO_IGNORE and must be filtered by _clean.
CUSTOMERS = ["AMAZON-DC", "AMAZON-DS", "SANIKAL-KG", "COSTCO-CAN", "Web Sales - AU", "Others - UK"]

HIST_WEEKS = pd.date_range("2026-04-26", "2026-06-21", freq="7D")   # 9 Sundays
FUT_WEEKS = pd.date_range("2026-06-28", "2026-08-16", freq="7D")    # 8 Sundays


def build(out_path):
    rng = np.random.default_rng(42)
    rows = []
    for i in range(12):
        sku = f"SKU-{i + 1:03d}"
        desc = f"Test Product {i + 1}"
        # each SKU is sold through 2-3 customers, deterministic per SKU
        n_cust = 2 + (i % 2)
        custs = [CUSTOMERS[(i + k) % len(CUSTOMERS)] for k in range(n_cust)]
        orders_only = i % 4 == 0  # every 4th SKU has no POS -> Orders fallback
        for cust in custs:
            base = float(rng.integers(20, 200))
            trend = float(rng.normal(0, 1.5))
            for t, wk in enumerate(HIST_WEEKS):
                level = max(base + trend * t + rng.normal(0, base * 0.08), 0.0)
                pos = np.nan if orders_only else round(level)
                orders = round(level * float(rng.uniform(0.7, 1.1)))
                proj = round(level * float(rng.uniform(0.85, 1.15)))
                rows.append([sku, desc, cust, wk, pos, orders, proj])
            for wk in FUT_WEEKS:  # future: original projection only
                proj = round(max(base + trend * len(HIST_WEEKS), 0.0))
                rows.append([sku, desc, cust, wk, np.nan, np.nan, proj])

    raw = pd.DataFrame(
        rows,
        columns=["'Demand'[DisplaySKU]", "Description", "Custnmbr", "WeekDate",
                 "POS", "Sum of Quantity", "Projection"],
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        # startrow=2 -> two banner rows above the header, matching header=2 reads
        raw.to_excel(w, sheet_name="Sheet1", index=False, startrow=2)
    return out_path


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    print("wrote", build(os.path.join(here, FIXTURE_NAME)))
