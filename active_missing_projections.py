# Find SKUs that are "Active In" each region but do not have forecasts.
#
# All the heavy lifting lives in agent/data_io.py (the single source of truth,
# shared with the dashboard and the agent):
#   - combine_warehouse_projections : clean the wide warehouse exports -> long
#   - compute_missing_projections   : active SKUs missing future projections in
#                                     regions they ARE 'Active in'
#
# This script is just the batch entry point: point it at the raw folders, run
# the shared logic, and write the final missing_projections file to outputs/.
# The cleaned long-format data is kept in memory and never saved to disk.

import os
from datetime import date

from agent import data_io

TODAY = date.today().strftime('%Y-%m-%d')

INPUT_PATH = "raw_inputs/warehouse_projections"
PLYTIX_FILE = "raw_inputs/list_prices/list_prices_06-30.xlsx"
OUTPUT_PATH = "outputs"
os.makedirs(OUTPUT_PATH, exist_ok=True)


if __name__ == '__main__':

    # 1) Clean + combine every warehouse export (kept in memory, not saved)
    xlsx_files = [f for f in os.listdir(INPUT_PATH) if f.endswith(".xlsx")]
    if not xlsx_files:
        print(f"No Excel files found in {INPUT_PATH}")
        raise SystemExit

    sources = [(os.path.join(INPUT_PATH, f), f) for f in xlsx_files]
    projections = data_io.combine_warehouse_projections(sources)
    for loc in sorted(projections["Location"].unique()):
        print(f"Cleaned {loc}: {(projections['Location'] == loc).sum()} rows")

    # 2) Active SKUs missing future projections in regions they ARE 'Active in'.
    #    df/P are only used to add a Region label (for the dashboard); the batch
    #    output doesn't need it, so pass None and drop the column.
    plytix_df = data_io.read_plytix(PLYTIX_FILE)
    missing = data_io.compute_missing_projections(
        projections, plytix_df, df=None, P=None
    ).drop(columns=["Region"])

    output_fname = f"{OUTPUT_PATH}/active_missing_projections_{TODAY}.xlsx"
    missing.to_excel(output_fname, index=False)

    print(missing.head(20))
    print(f"\nTotal rows: {len(missing)}")
    print(f"Saved to {output_fname}")
    print("Done!")
