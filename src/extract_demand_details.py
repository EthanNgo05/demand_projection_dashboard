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
    python extract_demand_details.py               # full pull -> dated .xlsx
    python extract_demand_details.py --incremental  # last few weeks + projections,
                                                    # merged into newest snapshot
    python extract_demand_details.py --ping         # fast connectivity smoke test
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
import time
from datetime import date, timedelta
from typing import Callable

import pandas as pd
import pyodbc
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("extract_demand_details")

HERE = os.path.dirname(os.path.abspath(__file__))
# Repo root (parent of src/) — sql/, raw_inputs/ etc. live there, not under src/.
REPO_ROOT = os.path.dirname(HERE)
DEFAULT_SQL = os.path.join(REPO_ROOT, "sql", "demand_details_optimized.sql")

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
KEEP_SNAPSHOTS = int(os.environ.get("DEMAND_KEEP_SNAPSHOTS", "3"))

# --incremental: how many weeks of recent actuals to re-pull (plus all forward
# projections, which the SQL window always covers). A few weeks of buffer, not
# just one, because recent POS/order rows can be restated after the fact; the
# nightly full pull self-heals anything older.
INCREMENTAL_WEEKS_BACK = int(os.environ.get("DEMAND_INCREMENTAL_WEEKS_BACK", "2"))

# Marker line in the .sql batch that --incremental replaces with a re-assignment
# of @StartSunday. On a full pull it's a plain comment, so the batch is
# unchanged. Matched as a line prefix.
INCREMENTAL_MARKER = "-- INCREMENTAL_START_OVERRIDE"

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


def load_demand_details(
    path: str = DEFAULT_SQL,
    sql_transform: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    """Read the .sql file, run it, and return the validated result DataFrame.

    ``sql_transform``, when given, rewrites the batch text before execution —
    used by ``--incremental`` to narrow the date window. The default (None)
    runs the file as-is.
    """
    sql = read_sql_file(path)
    if sql_transform is not None:
        sql = sql_transform(sql)
    log.info("Connecting: %s", redacted_connection_string())
    with connect() as conn:
        df = run_query(sql, conn)
    return select_and_rename(df)


def incremental_start_sunday(weeks_back: int, today: date | None = None) -> date:
    """The Sunday that starts the incremental window, ``weeks_back`` weeks ago.

    Snaps backward to a Sunday (the pipeline's week anchor) so the cutoff always
    falls on a week boundary — every week is either fully re-pulled or fully
    kept from the previous snapshot, never split.
    """
    d = (today or date.today()) - timedelta(weeks=weeks_back)
    return d - timedelta(days=(d.weekday() + 1) % 7)


def build_incremental_sql(sql_text: str, start_sunday: date) -> str:
    """Rewrite the batch so @StartSunday is ``start_sunday`` instead of 36mo ago.

    Replaces the ``INCREMENTAL_MARKER`` comment line with a re-assignment of
    @StartSunday via the same pbi.calendar lookup the batch already uses (an
    identity lookup, since we pass a Sunday). Raises if the marker is missing —
    e.g. ``--sql`` points at a batch without it — rather than silently running
    the full 36-month pull.
    """
    iso = f"{start_sunday:%Y-%m-%d}"
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", iso):  # belt-and-braces: date only
        raise ValueError(f"Bad incremental start date: {iso!r}")
    override = (
        "select @StartSunday = TheStartingSunday "
        f"from pbi.calendar where TheDate = '{iso}';"
    )
    lines = sql_text.splitlines(keepends=True)
    replaced = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith(INCREMENTAL_MARKER):
            newline = "\n" if line.endswith("\n") else ""
            lines[i] = override + newline
            replaced = True
            break
    if not replaced:
        raise ValueError(
            f"SQL batch has no '{INCREMENTAL_MARKER}' marker line; cannot run "
            "an incremental pull against it. Use the default "
            "demand_details_optimized.sql or drop --incremental."
        )
    return "".join(lines)


def find_previous_snapshot(folder: str) -> str | None:
    """Newest dated snapshot workbook in ``folder``, or None.

    Same filename-date ordering as ``prune_old_snapshots`` and the dashboard's
    snapshot dropdown.
    """
    dated = []
    for path in glob.glob(os.path.join(folder, "all_demand_projections_*.xlsx")):
        m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(path))
        if m:
            dated.append((m.group(1), path))
    if not dated:
        return None
    return max(dated)[1]


def load_previous_snapshot(path: str) -> pd.DataFrame | None:
    """Read a snapshot workbook for merging, or None if it's unusable.

    Any failure — unreadable file, missing required columns, unparseable
    WeekDate — returns None (logged) so the caller falls back to a full pull
    instead of writing a snapshot with a corrupted history half.
    """
    required = [SQL_TO_POWERBI_FORMAT[c] for c in REQUIRED_SQL_COLUMNS]
    try:
        df = pd.read_excel(path, header=2)
        missing = [c for c in required if c not in df.columns]
        if missing:
            log.warning(
                "Previous snapshot %s is missing column(s) %s; ignoring it.",
                path, missing,
            )
            return None
        df["WeekDate"] = pd.to_datetime(df["WeekDate"])
    except Exception as exc:
        log.warning("Could not read previous snapshot %s: %s", path, exc)
        return None
    return df


def merge_snapshots(
    previous: pd.DataFrame, fresh: pd.DataFrame, cutoff: date
) -> pd.DataFrame:
    """History before ``cutoff`` from ``previous`` + everything from ``fresh``.

    The incremental SQL only returns weeks >= cutoff, so this is a clean
    partition at a week boundary, not a key-level upsert. ``fresh``'s column
    set wins: a column added to the SQL appears (NaN for old rows), a column
    removed disappears.
    """
    cutoff_ts = pd.Timestamp(cutoff)
    previous = previous.copy()
    fresh = fresh.copy()
    previous["WeekDate"] = pd.to_datetime(previous["WeekDate"])
    fresh["WeekDate"] = pd.to_datetime(fresh["WeekDate"])

    below = int((fresh["WeekDate"] < cutoff_ts).sum())
    if below:
        log.warning(
            "Incremental pull returned %d row(s) before the %s cutoff; "
            "dropping them (the previous snapshot covers those weeks).",
            below, cutoff_ts.date(),
        )
        fresh = fresh[fresh["WeekDate"] >= cutoff_ts]

    stale_cols = [c for c in previous.columns if c not in fresh.columns]
    if stale_cols:
        log.info(
            "Previous snapshot column(s) not in the fresh pull, dropped: %s",
            stale_cols,
        )
    old = previous[previous["WeekDate"] < cutoff_ts].reindex(
        columns=fresh.columns
    )

    # NB: no duplicate-key check here. (SKU, Custnmbr, WeekDate) is NOT unique
    # in the snapshot — the warehouse grain includes the Customer column, which
    # select_and_rename drops — and the WeekDate partition above already makes
    # old/fresh overlap impossible.
    merged = pd.concat([old, fresh], ignore_index=True)
    return merged.sort_values(
        ["WeekDate", COL_SKU, COL_CUST], kind="stable"
    ).reset_index(drop=True)


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
        _replace_with_retry(tmp, out_path)
    except BaseException:
        # Never leave a partial temp workbook behind on failure/interrupt.
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def parquet_sidecar_path(xlsx_path: str) -> str:
    """The ``.parquet`` sidecar path for a snapshot ``.xlsx`` (same basename).

    Read back by the app's fast-load path (``agent.data_io.read_raw_frame``):
    Parquet preserves dtypes and carries no banner rows, so it loads far faster
    than re-parsing the workbook and feeds ``_clean`` directly.
    """
    root, _ = os.path.splitext(xlsx_path)
    return root + ".parquet"


def write_parquet_sidecar(df: pd.DataFrame, xlsx_path: str) -> str | None:
    """Write ``df`` as a ``.parquet`` sidecar next to ``xlsx_path``, atomically.

    Best-effort: a missing Parquet engine (pyarrow) or a write failure is logged
    and swallowed — the ``.xlsx`` is the source of truth and the app falls back
    to it. Same temp-file + atomic-replace dance as ``write_powerbi_xlsx`` so a
    reader never sees a half-written sidecar. Returns the path written, or None.
    """
    out_path = parquet_sidecar_path(xlsx_path)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".parquet", dir=out_dir)
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        _replace_with_retry(tmp, out_path)
        return out_path
    except Exception as exc:  # engine missing / write error — xlsx still stands
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        log.warning("Could not write Parquet sidecar %s: %s", out_path, exc)
        return None


def _replace_with_retry(
    src: str, dst: str, attempts: int = 6, delay: float = 3.0
) -> None:
    """``os.replace`` that rides out a briefly-locked destination.

    On Windows the replace fails with PermissionError while ANOTHER process has
    ``dst`` open — e.g. the dashboard mid-``read_excel``, or Excel. Same-day
    re-pulls overwrite an existing snapshot, so this collision genuinely
    happens (it killed the 2026-07-13 14:36 refresh). A reader's lock lasts
    seconds; a workbook left open in Excel doesn't — so retry briefly, then
    surface a clear error instead of a bare traceback.
    """
    for attempt in range(1, attempts + 1):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts:
                raise ValueError(
                    f"Could not overwrite {dst}: the file stayed locked by "
                    "another program (Excel?) through "
                    f"{attempts} attempts over ~{int((attempts - 1) * delay)}s. "
                    "Close it and re-run."
                ) from None
            log.warning(
                "%s is locked (attempt %d/%d); retrying in %.0fs…",
                os.path.basename(dst), attempt, attempts, delay,
            )
            time.sleep(delay)


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
        return folder or os.path.join(REPO_ROOT, "raw_inputs", "demand_projections")


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


def prune_old_snapshots(
    folder: str,
    keep: int = KEEP_SNAPSHOTS,
    pattern: str = "all_demand_projections_*.xlsx",
) -> list[str]:
    """Delete all but the newest ``keep`` dated snapshots in ``folder``.

    Keeps the output folder — and the dashboard's snapshot dropdown — from
    growing without bound as the nightly pull adds a snapshot per day. "Newest"
    is by the ``YYYY-MM-DD`` date embedded in the filename (the same ordering
    the dashboard uses), NOT mtime, so re-running today's pull never evicts an
    older day. ``keep`` counts distinct *dates*, not files — a warehouse
    snapshot is five region files sharing one date and lives or dies as a set.
    Files matching ``pattern`` but carrying no date are left untouched — never
    auto-deleted. ``keep <= 0`` disables pruning entirely. Returns the list of
    removed paths.
    """
    if keep <= 0:
        return []
    by_date: dict[str, list[str]] = {}
    for path in glob.glob(os.path.join(folder, pattern)):
        m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(path))
        if m:
            by_date.setdefault(m.group(1), []).append(path)
    removed = []
    for d in sorted(by_date, reverse=True)[keep:]:  # newest dates first
        for path in by_date[d]:
            try:
                os.remove(path)
                removed.append(path)
            except OSError as exc:
                log.warning("Could not prune old snapshot %s: %s", path, exc)
            # Drop the Parquet sidecar for the same snapshot, if one exists.
            sidecar = parquet_sidecar_path(path)
            if os.path.exists(sidecar):
                try:
                    os.remove(sidecar)
                    removed.append(sidecar)
                except OSError as exc:
                    log.warning("Could not prune sidecar %s: %s", sidecar, exc)
    if removed:
        log.info(
            "Pruned %d old snapshot file(s), keeping the newest %d date(s): %s",
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
        "--incremental",
        action="store_true",
        help=(
            "Only pull the last --weeks-back weeks of actuals plus all forward "
            "projections, and merge into the newest existing snapshot. Falls "
            "back to a full pull if no usable snapshot exists."
        ),
    )
    parser.add_argument(
        "--weeks-back",
        type=int,
        default=INCREMENTAL_WEEKS_BACK,
        help=(
            "Incremental window in weeks (default: %(default)s, or "
            "DEMAND_INCREMENTAL_WEEKS_BACK)."
        ),
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

        previous = None
        if args.incremental:
            if args.weeks_back < 0:
                raise ValueError(
                    f"--weeks-back must be >= 0, got {args.weeks_back}."
                )
            prev_path = find_previous_snapshot(_default_raw_dir())
            previous = load_previous_snapshot(prev_path) if prev_path else None
            if previous is None:
                log.warning(
                    "No usable previous snapshot in %s; running a FULL pull.",
                    _default_raw_dir(),
                )

        if previous is not None:
            cutoff = incremental_start_sunday(args.weeks_back)
            log.info(
                "Incremental pull: weeks >= %s (last %d week(s) + projections), "
                "merging into %s", cutoff, args.weeks_back,
                os.path.basename(prev_path),
            )
            fresh = load_demand_details(
                args.sql, sql_transform=lambda s: build_incremental_sql(s, cutoff)
            )
            fresh = _apply_output_filters(fresh)
            if fresh.empty:
                log.error(
                    "Incremental pull returned 0 rows; refusing to write a "
                    "truncated snapshot. Previous file left untouched."
                )
                return 1
            df = merge_snapshots(previous, fresh, cutoff)
            print(
                f"Incremental: {len(fresh):,} fresh rows (weeks >= {cutoff}) + "
                f"{len(df) - len(fresh):,} rows kept from previous snapshot"
            )
        else:
            df = load_demand_details(args.sql)
            df = _apply_output_filters(df)

        out_path = args.out or default_out_path()
        out_dir = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(out_dir, exist_ok=True)
        write_powerbi_xlsx(df, out_path)
        # Fast-load sidecar for the dashboard/agent (best-effort; xlsx is source
        # of truth). Written after the xlsx so the two share the snapshot's date.
        sidecar = write_parquet_sidecar(df, out_path)
        # Prune only after the new file is safely written, so a failed/partial
        # pull never deletes good history.
        prune_old_snapshots(out_dir)

        print(f"Pulled {len(df):,} rows x {len(df.columns)} columns")
        if sidecar:
            print(f"Wrote Parquet sidecar {sidecar}")
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
