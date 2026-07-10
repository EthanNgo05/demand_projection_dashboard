"""Pull the demand-details result set from SQL Server into a PowerBI-format xlsx.

Reads a .sql file (default: ``sql/demand_details.sql``), runs it against SQL
Server, keeps the demand-details result set, and writes it as an .xlsx laid out
exactly like the PowerBI export the rest of the pipeline consumes.

``demand_details.sql`` is a multi-statement T-SQL batch: it declares variables,
builds temp tables, and emits several result sets along the way. We run the
whole batch and keep the *last* result set that has columns — i.e. the closing
``select`` — which is the demand-details table we actually want.

Output layout (must match ``pd.read_excel(path, header=2)`` used by dashboard.py,
agent/data_io.py and models/*.py — see README "headers on row 3"):

    row 1  applied-filters banner (single cell, informational)
    row 2  (blank)
    row 3  column headers  ('Demand'[DisplaySKU], Description, ...)
    row 4+ data

Connection details come from environment variables (loaded from ``.env`` via
python-dotenv). SQL-login auth:

    SQL_SERVER=your_server       # required
    SQL_DATABASE=your_database   # required
    SQL_USER=your_login
    SQL_PASSWORD=your_password
    # optional overrides:
    # SQL_DRIVER=ODBC Driver 18 for SQL Server
    # SQL_ENCRYPT=yes           (secure default; set 'no' only if you must)
    # SQL_TRUST_CERT=no         (secure default; 'yes' disables cert validation)
    # SQL_SERVER_CERT=...       (path to the DW cert to pin & validate securely)
    # SQL_LOGIN_TIMEOUT=30      (seconds to establish the connection)
    # SQL_QUERY_TIMEOUT=900     (seconds for the batch; 0 = unlimited)
    # DEMAND_RAW_DIR=...        (where to write the dated output workbook)

Leave SQL_USER blank to use Windows (trusted) authentication instead.

Run:
    python extract_demand_details.py            # full pull -> dated .xlsx
    python extract_demand_details.py --ping      # fast connectivity smoke test
    python extract_demand_details.py --out x.xlsx
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import sys
import tempfile
from datetime import date

import pandas as pd
import pyodbc
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("extract_demand_details")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SQL = os.path.join(HERE, "sql", "demand_details.sql")

# SQL result column -> PowerBI/Excel column the dashboard expects. Only these
# columns are carried into the output; everything else is dropped (and logged).
SQL_TO_POWERBI_FORMAT = {
    "DisplaySKU": "'Demand'[DisplaySKU]",
    "LongName": "Description",
    "Custnmbr": "Custnmbr",
    "WeekDate": "WeekDate",
    "SalesUnits": "POS",
    "ProjQty": "Projection",
    "PromoProj": "Promo Qty",
    "OnHand": "On Hand",
    "OnOrder": "On Order",
    "Quantity": "Sum of Quantity",
    "InStock": "In Stock",
    "StoreCount": "Store Count",
}

# Columns the downstream pipeline (agent/data_io._clean) genuinely depends on.
# If any are absent from the SQL result we fail loudly rather than write a
# silently-malformed workbook.
REQUIRED_SQL_COLUMNS = (
    "DisplaySKU",
    "LongName",
    "Custnmbr",
    "WeekDate",
    "SalesUnits",
    "ProjQty",
)

# Renamed (output) column names used by the __main__ filtering step.
COL_SKU = SQL_TO_POWERBI_FORMAT["DisplaySKU"]
COL_CUST = SQL_TO_POWERBI_FORMAT["Custnmbr"]

# Written into the top banner cell so the file self-documents its filters,
# mirroring the "Applied filters:" note the real PowerBI export carries.
FILTER_BANNER = (
    "Does not include SKUs starting with 'AS' or rows with a blank customer."
)

# Timeouts (seconds). Login guards a hung/unreachable server on the main path;
# query timeout bounds the multi-minute batch. 0 disables the query timeout.
LOGIN_TIMEOUT = int(os.environ.get("SQL_LOGIN_TIMEOUT", "30"))
QUERY_TIMEOUT = int(os.environ.get("SQL_QUERY_TIMEOUT", "900"))

# How many dated snapshot workbooks to keep in the output folder after a
# successful write. Older ones are pruned so the folder (and the dashboard's
# snapshot dropdown) don't grow without bound. 0 (or less) disables pruning.
KEEP_SNAPSHOTS = int(os.environ.get("DEMAND_KEEP_SNAPSHOTS", "10"))

_TRUTHY = {"yes", "true", "1", "on"}


def read_sql_file(path: str = DEFAULT_SQL) -> str:
    """Read a .sql file, decoding by BOM rather than guessing.

    SSMS writes UTF-16 with a BOM; other editors write UTF-8 (with or without a
    BOM). We sniff the BOM to pick the decoder deterministically, and fall back
    to strict UTF-8 when there is none. We deliberately do NOT fall back to a
    never-fails codec like latin-1: silently mis-decoding a T-SQL batch would
    ship corrupted SQL to the server, which is far worse than a clear error.
    """
    with open(path, "rb") as f:
        raw = f.read()

    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{path} is not valid UTF-8/UTF-16 and has no byte-order mark. "
            "Re-save it as UTF-8 or UTF-16 from your editor."
        ) from exc


def _require_env(name: str) -> str:
    """Return a required env var, or raise a clear error if it's unset/blank.

    Server and database names are not hardcoded: they must come from the
    environment (``.env``) so company infrastructure isn't baked into source.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(
            f"{name} is not set. Add it to your .env (see .env.example)."
        )
    return value


def _odbc_quote(value: str) -> str:
    """Brace-quote an ODBC connection-string value.

    Values containing ``;`` ``{`` ``}`` etc. (common in passwords) would
    otherwise break parsing or leak into the wrong keyword. Per the ODBC rule,
    wrap the value in braces and double any closing brace inside it.
    """
    return "{" + value.replace("}", "}}") + "}"


def connection_string() -> str:
    """Build the pyodbc connection string from environment variables.

    NEVER log the return value: it contains the password. Use
    ``redacted_connection_string`` for diagnostics instead.
    """
    driver = os.environ.get("SQL_DRIVER", "ODBC Driver 18 for SQL Server")
    server = _require_env("SQL_SERVER")
    database = _require_env("SQL_DATABASE")

    parts = [
        f"DRIVER={{{driver}}}",
        f"SERVER={server}",
        f"DATABASE={database}",
    ]

    user = os.environ.get("SQL_USER")
    password = os.environ.get("SQL_PASSWORD")
    if user:
        if not password:
            raise ValueError(
                "SQL_USER is set but SQL_PASSWORD is empty. Refusing to connect "
                "with a blank password. Set SQL_PASSWORD, or clear SQL_USER to "
                "use Windows (trusted) authentication."
            )
        parts += [f"UID={_odbc_quote(user)}", f"PWD={_odbc_quote(password)}"]
    else:
        parts.append("Trusted_Connection=yes")

    # Secure defaults: encrypt in transit and validate the server certificate.
    # These can be relaxed via env for legacy servers, but we warn loudly so an
    # insecure connection to the data warehouse is never a silent surprise.
    encrypt = os.environ.get("SQL_ENCRYPT", "yes")
    trust_cert = os.environ.get("SQL_TRUST_CERT", "no")
    parts.append(f"Encrypt={encrypt}")
    parts.append(f"TrustServerCertificate={trust_cert}")

    # Preferred way to accept a self-signed DW cert without disabling
    # validation: pin the server's cert file. ODBC Driver 18 validates the
    # server certificate against this exact PEM/DER instead of the chain, so
    # Encrypt=yes / TrustServerCertificate=no can stay put.
    server_cert = os.environ.get("SQL_SERVER_CERT", "").strip()
    if server_cert:
        if not os.path.isfile(server_cert):
            raise ValueError(
                f"SQL_SERVER_CERT points to a missing file: {server_cert}"
            )
        parts.append(f"ServerCertificate={_odbc_quote(server_cert)}")

    if encrypt.strip().lower() not in _TRUTHY:
        log.warning(
            "SQL_ENCRYPT=%s: data-warehouse traffic (including query results) "
            "will NOT be encrypted in transit.",
            encrypt,
        )
    if trust_cert.strip().lower() in _TRUTHY:
        log.warning(
            "SQL_TRUST_CERT=yes: the server certificate is NOT validated, which "
            "defeats MITM protection. Prefer installing the server CA and "
            "setting SQL_TRUST_CERT=no.",
        )

    return ";".join(parts) + ";"


def redacted_connection_string() -> str:
    """The connection string with the password masked, safe to log."""
    return ";".join(
        "PWD={***}" if p.startswith("PWD=") else p
        for p in connection_string().rstrip(";").split(";")
    )


def connect(login_timeout: int = LOGIN_TIMEOUT) -> pyodbc.Connection:
    """Open a connection with a bounded login timeout on every path."""
    return pyodbc.connect(connection_string(), timeout=login_timeout)


def run_query(sql: str, conn: pyodbc.Connection) -> pd.DataFrame:
    """Execute a (possibly multi-statement) batch, return the last result set.

    ``SET NOCOUNT ON`` is prepended so the intermediate INSERT/UPDATE steps
    don't emit "rows affected" counts that would otherwise show up as empty
    result sets. We then walk every result set with ``nextset()`` and keep the
    last one that actually has columns.
    """
    conn.timeout = QUERY_TIMEOUT  # bounds the multi-minute batch (0 = unlimited)
    cursor = conn.cursor()
    cursor.execute("SET NOCOUNT ON;\n" + sql)

    columns, rows = None, None
    while True:
        if cursor.description:  # this result set has columns
            columns = [c[0] for c in cursor.description]
            rows = cursor.fetchall()
        if not cursor.nextset():
            break

    if columns is None:
        return pd.DataFrame()
    # pyodbc.Row objects -> plain tuples so the DataFrame builds cleanly.
    return pd.DataFrame.from_records([tuple(r) for r in rows], columns=columns)


def select_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns, then keep+rename to the PowerBI schema.

    Raises if a required column is missing (never ship a silently-malformed
    workbook). Logs any expected-but-absent optional columns and any unmapped
    columns that get dropped, so column changes in the SQL are visible.
    """
    missing = [c for c in REQUIRED_SQL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"SQL result is missing required column(s): {missing}. "
            f"Got columns: {list(df.columns)}"
        )

    absent_optional = [
        c
        for c in SQL_TO_POWERBI_FORMAT
        if c not in df.columns and c not in REQUIRED_SQL_COLUMNS
    ]
    if absent_optional:
        log.warning("Expected columns absent from SQL result: %s", absent_optional)

    dropped = [c for c in df.columns if c not in SQL_TO_POWERBI_FORMAT]
    if dropped:
        log.info("Dropping %d unmapped SQL column(s): %s", len(dropped), dropped)

    keep = [c for c in SQL_TO_POWERBI_FORMAT if c in df.columns]
    return df[keep].rename(columns=SQL_TO_POWERBI_FORMAT)


def load_demand_details(path: str = DEFAULT_SQL) -> pd.DataFrame:
    """Read the .sql file, run it, and return the validated result DataFrame."""
    sql = read_sql_file(path)
    log.info("Connecting: %s", redacted_connection_string())
    with connect() as conn:
        df = run_query(sql, conn)
    return select_and_rename(df)


def write_powerbi_xlsx(df: pd.DataFrame, out_path: str, banner: str = FILTER_BANNER) -> None:
    """Write ``df`` in the PowerBI export layout the pipeline reads (header=2).

    Header lands on the 3rd row (``startrow=2``); the banner goes in cell A1 and
    the 2nd row is left blank — matching ``pd.read_excel(path, header=2)``.

    Written atomically: the workbook is built in a temp file in the same folder
    and then ``os.replace``-d onto ``out_path``. A reader (the dashboard) polling
    this folder therefore never sees a half-written workbook mid-refresh, and its
    mtime-keyed cache flips to the complete file in a single step. The temp file
    sits in the destination directory so the replace is a same-filesystem atomic
    rename rather than a cross-device copy.
    """
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".xlsx", dir=out_dir)
    os.close(fd)
    try:
        with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Sheet1", index=False, startrow=2)
            writer.sheets["Sheet1"].cell(row=1, column=1, value=banner)
        os.replace(tmp, out_path)
    except BaseException:
        # Never leave a partial temp workbook behind on failure/interrupt.
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _default_raw_dir() -> str:
    """The folder the dashboard discovers snapshots in (single source of truth).

    Delegates to the dashboard's own resolver (``agent.data_io._raw_dir``) so the
    two never drift: DEMAND_RAW_DIR if set, else the pipeline's
    ``RAW_INPUTS_FOLDER`` (``raw_inputs/demand_projections``) resolved against the
    repo root. Falls back to that standard location if the agent package can't be
    imported (keeps this script runnable stand-alone).
    """
    try:
        from agent import data_io

        return data_io._raw_dir()
    except Exception:  # pragma: no cover - defensive fallback
        folder = os.environ.get("DEMAND_RAW_DIR")
        return folder or os.path.join(HERE, "raw_inputs", "demand_projections")


def default_out_path() -> str:
    """Dated output workbook, in the folder the dashboard reads snapshots from.

    Uses the ``all_demand_projections_YYYY-MM-DD.xlsx`` name the pipeline globs
    for, written into ``_default_raw_dir()`` — so a plain run of this script
    drops the file straight into the dashboard's "Snapshot (raw file)" dropdown
    with no copy step and no rename.
    """
    return os.path.join(
        _default_raw_dir(), f"all_demand_projections_{date.today():%Y-%m-%d}.xlsx"
    )


def prune_old_snapshots(folder: str, keep: int = KEEP_SNAPSHOTS) -> list[str]:
    """Delete all but the newest ``keep`` dated snapshot workbooks in ``folder``.

    Keeps the output folder — and the dashboard's snapshot dropdown — from
    growing without bound as the nightly pull adds a file per day. "Newest" is by
    the ``YYYY-MM-DD`` date embedded in the filename (the same ordering the
    dashboard uses), NOT mtime, so re-running today's pull never evicts an older
    day. Files whose name carries no date are left untouched — never
    auto-deleted. ``keep <= 0`` disables pruning entirely. Returns the list of
    removed paths.
    """
    if keep <= 0:
        return []
    dated = []
    for path in glob.glob(os.path.join(folder, "all_demand_projections_*.xlsx")):
        m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(path))
        if m:
            dated.append((m.group(1), path))
    dated.sort(reverse=True)  # newest date first
    removed = []
    for _, path in dated[keep:]:
        try:
            os.remove(path)
            removed.append(path)
        except OSError as exc:
            log.warning("Could not prune old snapshot %s: %s", path, exc)
    if removed:
        log.info(
            "Pruned %d old snapshot(s), keeping newest %d: %s",
            len(removed), keep, [os.path.basename(p) for p in removed],
        )
    return removed


def ping() -> tuple[dict, object]:
    """Instant smoke test: confirm we can connect and reach the right DB.

    Returns the server version, active database, and clock, plus one row from
    pbi.calendar to prove a real table in the schema is readable. Comes back
    immediately, unlike the multi-minute batch.
    """
    sql = """
        SELECT
            login       = SUSER_SNAME(),
            [database]  = DB_NAME(),
            server_time = SYSDATETIME(),
            version     = LEFT(@@VERSION, 40);
        SELECT TOP 1 TheDate, TheStartingSunday FROM pbi.calendar ORDER BY TheDate DESC;
    """
    with connect(login_timeout=10) as conn:
        conn.timeout = 10
        cursor = conn.cursor()
        cursor.execute(sql)
        # First result set: connection/identity info.
        info = dict(zip([c[0] for c in cursor.description], cursor.fetchone()))
        # Second result set: proof a real table is reachable.
        cursor.nextset()
        cal = cursor.fetchone()
    return info, cal


def _apply_output_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Drop blank-customer rows and SKUs starting with 'AS' (see FILTER_BANNER)."""
    df = df[df[COL_CUST].notna()]
    df = df[~df[COL_SKU].astype(str).str.startswith("AS", na=False)]
    return df


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
        "--out",
        default=None,
        help="Output .xlsx path (default: dated file in DEMAND_RAW_DIR or here).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
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

        df = load_demand_details(args.sql)
        df = _apply_output_filters(df)

        out_path = args.out or default_out_path()
        out_dir = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(out_dir, exist_ok=True)
        write_powerbi_xlsx(df, out_path)
        # Prune only after the new file is safely written, so a failed/partial
        # pull never deletes good history.
        prune_old_snapshots(out_dir)

        print(f"Pulled {len(df):,} rows x {len(df.columns)} columns")
        print("Columns:", list(df.columns))
        print(df.head(10).to_string())
        print(f"Wrote {out_path}")
        return 0
    except pyodbc.Error as exc:
        log.error("Database error: %s", exc)
        return 1
    except (ValueError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
