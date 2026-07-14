"""Pull warehouse projections from SQL Server and write the 5 regional files.

Sibling of ``extract_demand_details.py``: reads a .sql file (default:
``sql/warehouse_projections.sql``), runs it against SQL Server, keeps the final
result set, and reshapes it into the same per-region files the planning team
used to export from PowerBI by hand:

    AU/CA/EU/JP/US_warehouse_projections_<date>.xlsx

Each file is the long "export data" layout the dashboard reads (banner row,
blank row, ``header=2``: ``'Projection_by_Warehouse'[DisplaySKU]``,
``CUSTNMBR``, ``WeekDate``, ``Sum of Proj``), written atomically into the
folder ``agent.data_io._warehouse_dir()`` resolves (``WAREHOUSE_RAW_DIR`` or
``raw_inputs/warehouse_projections``) — so a plain run drops a snapshot
straight into the dashboard's "Warehouse snapshot" dropdown, and
``active_missing_projections.py`` picks it up unchanged.

``warehouse_projections.sql`` is a multi-statement T-SQL batch — it builds
several temp tables and emits its answer as the closing ``SELECT`` (an UNPIVOT
of the promo/base projections, allocated to warehouses, zero rows filtered).
We run the whole batch and keep the *last* result set that has columns,
exactly like the demand-details extract does. That closing select returns:

    CUSTNMBR, ITEMNMBR, WeekDate, Warehouse, ProjType, Proj

The transform sums the two ProjTypes (and a region's warehouses — the US is
LBC+NJ) per (SKU, customer, week) and drops zero totals: the PowerBI exports
omit zero rows, and downstream treats an absent (pair, week) cell as a
*missing* projection, which is exactly what a zero allocation is.

Connection details and auth come from the same environment variables the
demand-details extract uses (loaded from ``.env`` via python-dotenv) — see that
script's docstring or ``.env.example``. Leave SQL_USER blank for Windows
(trusted) authentication.

Run (from the repo root):
    python src/extract_warehouse_projections.py             # pull -> 5 dated regional .xlsx
    python src/extract_warehouse_projections.py --ping      # connectivity smoke test
    python src/extract_warehouse_projections.py --raw-out raw.xlsx   # also dump the raw table
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date

# Reuse the battle-tested plumbing from the demand-details extract instead of
# duplicating it: connection string (secure defaults, redaction), the .sql
# reader (BOM-aware — the SQL file is UTF-16), the multi-result-set runner
# that keeps the closing SELECT, the atomic PowerBI-layout writer, and the
# snapshot pruner. Running this as ``python src/<this>.py`` puts src/ on
# sys.path, so these bare imports resolve (same as the sibling script's
# ``from agent import data_io``).
from extract_demand_details import (  # noqa: E402
    KEEP_SNAPSHOTS,
    REPO_ROOT,
    connect,
    ping,
    prune_old_snapshots,
    read_sql_file,
    redacted_connection_string,
    run_query,
    write_powerbi_xlsx,
)

log = logging.getLogger("extract_warehouse_projections")

DEFAULT_SQL = os.path.join(REPO_ROOT, "sql", "warehouse_projections.sql")

# Columns the closing SELECT is expected to emit. We fail loudly if the shape
# drifts so a silently empty/wrong snapshot never gets written.
REQUIRED_COLUMNS = ("CUSTNMBR", "ITEMNMBR", "WeekDate", "Warehouse", "ProjType", "Proj")

# Which physical warehouses roll up into each fulfillment region. Must cover
# every Warehouse value the allocation emits — transform_to_regions fails
# loudly on an unmapped warehouse so a newly added DC can't silently vanish
# from the missing-projections check.
REGION_WAREHOUSES = {
    "US": ("LBC", "NJ"),
    "EU": ("SH-CTS",),
    "AU": ("ACR",),
    "CA": ("YYZ5",),
    "JP": ("NETDEPOT",),
}

# Exact column headers of the manual PowerBI "export data" files, so the
# dashboard reader treats both sources identically.
PBI_COLUMNS = [
    "'Projection_by_Warehouse'[DisplaySKU]", "CUSTNMBR", "WeekDate", "Sum of Proj",
]

BANNER = (
    "Generated from the data warehouse by extract_warehouse_projections.py\n"
    "Region = filename prefix; zero-projection rows omitted (missing = absent row)"
)


def load_warehouse_projections(path: str = DEFAULT_SQL) -> "pd.DataFrame":  # noqa: F821
    """Read the .sql file, run the batch, and return the final result set."""
    sql = read_sql_file(path)
    log.info("Connecting: %s", redacted_connection_string())
    with connect() as conn:
        df = run_query(sql, conn)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"SQL result is missing expected column(s): {missing}. "
            f"Got columns: {list(df.columns)}"
        )
    return df


def transform_to_regions(df) -> dict:
    """Raw warehouse-allocated rows -> one PBI-layout frame per region.

    Per region: keep that region's warehouses, sum ``Proj`` over ProjType and
    warehouse per (SKU, customer, week), drop zero totals (again — offsetting
    promo/base values can net to zero even though the SQL already filtered
    zero rows), and rename to the exact PowerBI export schema.

    GP CHAR columns are space-padded; keys are stripped here so the SKUs line
    up with the (unpadded) Plytix export downstream. Every region in
    REGION_WAREHOUSES gets an entry, possibly empty.
    """
    import pandas as pd

    work = df.copy()
    for col in ("CUSTNMBR", "ITEMNMBR", "Warehouse"):
        work[col] = work[col].astype(str).str.strip()
    work["Proj"] = pd.to_numeric(work["Proj"], errors="coerce").fillna(0.0)

    mapped = {w for whs in REGION_WAREHOUSES.values() for w in whs}
    unmapped = sorted(set(work["Warehouse"].unique()) - mapped)
    if unmapped:
        raise ValueError(
            f"Unmapped warehouse value(s) in the SQL result: {unmapped}. "
            "Add them to REGION_WAREHOUSES so their region isn't silently "
            "dropped from the missing-projections check."
        )

    frames = {}
    for region, warehouses in REGION_WAREHOUSES.items():
        sub = work[work["Warehouse"].isin(warehouses)]
        agg = (
            sub.groupby(["ITEMNMBR", "CUSTNMBR", "WeekDate"], as_index=False)["Proj"]
            .sum()
        )
        agg = agg[agg["Proj"] != 0]
        agg = agg.rename(columns={
            "ITEMNMBR": PBI_COLUMNS[0],
            "Proj": "Sum of Proj",
        })[PBI_COLUMNS]
        frames[region] = agg.sort_values(PBI_COLUMNS[:3]).reset_index(drop=True)
    return frames


def write_region_files(frames: dict, folder: str, day: date) -> list[str]:
    """Write one dated PBI-layout workbook per region; prune old snapshots.

    Every region is written on every run — a region with nothing to project
    gets a headers-only file — so a snapshot is always a complete 5-file set
    (the dashboard's refresh-done check counts on that). Each file is written
    atomically (temp + ``os.replace``); the set as a whole is not atomic, but
    the window is seconds.
    """
    os.makedirs(folder, exist_ok=True)
    written = []
    for region, frame in frames.items():
        path = os.path.join(
            folder, f"{region}_warehouse_projections_{day:%Y-%m-%d}.xlsx"
        )
        write_powerbi_xlsx(frame, path, banner=BANNER)
        written.append(path)
        log.info("Wrote %s (%d rows)", os.path.basename(path), len(frame))
    prune_old_snapshots(
        folder, keep=KEEP_SNAPSHOTS, pattern="*_warehouse_projections_*.xlsx"
    )
    return written


def default_out_dir() -> str:
    """The folder the dashboard discovers warehouse snapshots in.

    Delegates to the dashboard's own resolver (``agent.data_io._warehouse_dir``)
    so the two never drift: WAREHOUSE_RAW_DIR if set, else
    ``raw_inputs/warehouse_projections`` resolved against the repo root. Falls
    back to that standard location if the agent package can't be imported
    (keeps this script runnable stand-alone).
    """
    try:
        from agent import data_io

        return data_io._warehouse_dir()
    except Exception:  # pragma: no cover - defensive fallback
        folder = os.environ.get("WAREHOUSE_RAW_DIR")
        return folder or os.path.join(REPO_ROOT, "raw_inputs", "warehouse_projections")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ping",
        action="store_true",
        help="Fast connectivity smoke test instead of the full pull.",
    )
    parser.add_argument(
        "--sql",
        default=DEFAULT_SQL,
        help=f"Path to the .sql batch to run (default: {DEFAULT_SQL}).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Folder for the 5 regional .xlsx files "
             "(default: WAREHOUSE_RAW_DIR or raw_inputs/warehouse_projections/).",
    )
    parser.add_argument(
        "--raw-out",
        default=None,
        metavar="PATH",
        help="Also dump the raw (untransformed) result set to this .xlsx for "
             "inspection. If placed inside the warehouse folder it is ignored "
             "by the dashboard reader (no region prefix) but inflates its "
             "file-count caption — prefer a path outside it.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import pyodbc  # local import so --help works without the ODBC driver present

    args = _parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    try:
        if args.ping:
            info, cal = ping()
            print("Connection OK:")
            for k, v in info.items():
                print(f"  {k:12} {v}")
            print(f"  pbi.calendar latest row: {tuple(cal) if cal else None}")
            return 0

        df = load_warehouse_projections(args.sql)
        print(f"Pulled {len(df):,} rows x {len(df.columns)} columns")

        if args.raw_out:
            raw_dir = os.path.dirname(os.path.abspath(args.raw_out))
            os.makedirs(raw_dir, exist_ok=True)
            df.to_excel(args.raw_out, sheet_name="Sheet1", index=False)
            print(f"Raw dump: {args.raw_out}")

        frames = transform_to_regions(df)
        out_dir = args.out_dir or default_out_dir()
        written = write_region_files(frames, out_dir, date.today())
        for path in written:
            region = os.path.basename(path).split("_")[0]
            print(f"  {os.path.basename(path)}: {len(frames[region]):,} rows")
        print(f"Wrote {len(written)} regional file(s) to {out_dir}")
        return 0
    except pyodbc.Error as exc:
        log.error("Database error: %s", exc)
        return 1
    except (ValueError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
