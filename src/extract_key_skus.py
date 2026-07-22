"""Pull the key-SKU list from SQL Server and write it to a dated file.

Sibling of ``extract_demand_details.py`` / ``extract_warehouse_projections.py``:
reads ``sql/key_skus.sql`` (``SELECT DISTINCT SKU FROM dmd.week_of_supply_parameters
WHERE KeyItem = 'Yes'``), runs it against SQL Server, and writes the resulting
single ``SKU`` column to::

    raw_inputs/key_skus/key_skus_<date>.xlsx

The dashboard's Exceptions view discovers the newest such file (see
``agent.data_io.discover_key_skus_file`` / ``read_key_skus``) and uses it to
populate the "Key SKUs" watchlist tab — so a plain run drops a fresh list
straight into the app. DB access stays confined to these extract scripts; the
dashboard never queries SQL Server at render time.

Connection details and auth come from the same environment variables the
demand-details extract uses (loaded from ``.env`` via python-dotenv) — see that
script's docstring or ``.env.example``. Leave SQL_USER blank for Windows
(trusted) authentication.

Run (from the repo root):
    python src/extract_key_skus.py            # pull -> dated key_skus_<date>.xlsx
    python src/extract_key_skus.py --ping     # connectivity smoke test
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from datetime import date

import pandas as pd

# Reuse the battle-tested plumbing from the demand-details extract (connection
# string with secure defaults + redaction, the BOM-aware .sql reader, the
# multi-result-set runner, connectivity ping, snapshot pruner) instead of
# duplicating it. Running as ``python src/<this>.py`` puts src/ on sys.path.
from extract_demand_details import (  # noqa: E402
    KEEP_SNAPSHOTS,
    REPO_ROOT,
    connect,
    ping,
    prune_old_snapshots,
    read_sql_file,
    redacted_connection_string,
    run_query,
)

log = logging.getLogger("extract_key_skus")

DEFAULT_SQL = os.path.join(REPO_ROOT, "sql", "key_skus.sql")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "raw_inputs", "key_skus")


def load_key_skus(path: str = DEFAULT_SQL) -> "pd.DataFrame":  # noqa: F821
    """Read the .sql file, run it, and return a one-column ``SKU`` frame."""
    sql = read_sql_file(path)
    log.info("Connecting: %s", redacted_connection_string())
    with connect() as conn:
        df = run_query(sql, conn)

    if "SKU" not in df.columns:
        raise ValueError(
            f"SQL result is missing the expected 'SKU' column. "
            f"Got columns: {list(df.columns)}"
        )
    # Keep just SKU, stripped + de-duplicated, so the file matches the cleaned
    # demand frame's already-stripped SKU keys.
    out = df[["SKU"]].dropna()
    out["SKU"] = out["SKU"].astype(str).str.strip()
    out = out[out["SKU"] != ""].drop_duplicates().sort_values("SKU")
    return out.reset_index(drop=True)


def default_out_path(day: date | None = None) -> str:
    """Dated output path in the folder the dashboard discovers."""
    day = day or date.today()
    return os.path.join(DEFAULT_OUT_DIR, f"key_skus_{day:%Y-%m-%d}.xlsx")


def write_key_skus(df: "pd.DataFrame", path: str) -> None:  # noqa: F821
    """Write the SKU list atomically (temp file in the same dir + os.replace),
    then prune old snapshots so the folder keeps only the newest few."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".xlsx", dir=folder)
    os.close(fd)
    try:
        df.to_excel(tmp, sheet_name="Sheet1", index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    prune_old_snapshots(folder, keep=KEEP_SNAPSHOTS, pattern="key_skus_*.xlsx")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ping", action="store_true",
        help="Fast connectivity smoke test instead of the full pull.",
    )
    parser.add_argument(
        "--sql", default=DEFAULT_SQL,
        help=f"Path to the .sql query to run (default: {DEFAULT_SQL}).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output .xlsx path (default: raw_inputs/key_skus/key_skus_<date>.xlsx).",
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
            return 0

        df = load_key_skus(args.sql)
        out_path = args.out or default_out_path()
        write_key_skus(df, out_path)
        print(f"Wrote {len(df):,} key SKUs to {out_path}")
        return 0
    except pyodbc.Error as exc:
        log.error("Database error: %s", exc)
        return 1
    except (ValueError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
