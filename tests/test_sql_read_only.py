"""The demand extract must NEVER write to the data warehouse.

Runs entirely offline — no database, no credentials, no network. The guard under
test lives in ``read_sql_file``, which every code path funnels through (full pull,
``--incremental``, a custom ``--sql``) and which runs BEFORE any connection is
opened, so a batch that would write can never reach the server.

Why this file exists rather than a one-off audit: the shipped batch is a rewrite of
an original that ended with ``UPDATE pbi.als_demand_tab`` (a permanent warehouse
table), and ``sql/`` is not tracked by git — so a future edit could reintroduce a
write with no diff to review. These tests are the standing check.
"""
import glob
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from extract_demand_details import (  # noqa: E402
    assert_sql_is_read_only, read_sql_file,
)

SQL_DIR = os.path.join(REPO_ROOT, "sql")
SHIPPED_SQL = sorted(glob.glob(os.path.join(SQL_DIR, "*.sql")))


# --------------------------------------------------------------------------- #
# The guard must not be a false alarm: the real SQL has to pass               #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not SHIPPED_SQL, reason="sql/ not present in this checkout")
@pytest.mark.parametrize("path", SHIPPED_SQL, ids=os.path.basename)
def test_shipped_sql_is_read_only(path):
    """Every .sql file the pipeline can run must pass the guard.

    If this fails, either a write was introduced into production SQL, or the guard
    has become over-strict. Both need a human — do not silence it.
    """
    read_sql_file(path)


@pytest.mark.skipif(not SHIPPED_SQL, reason="sql/ not present in this checkout")
def test_shipped_sql_stages_into_temp_tables():
    """Sanity-check the guard is actually exercised, not vacuously passing.

    The demand batch stages through #temp tables, so if this assertion ever fails
    the file has changed shape enough that the read-only tests above may no longer
    be testing what they claim.
    """
    demand = os.path.join(SQL_DIR, "demand_details_optimized.sql")
    if not os.path.exists(demand):
        pytest.skip("demand_details_optimized.sql not present")
    sql = read_sql_file(demand)
    assert "#gp_pos" in sql, "expected the batch to stage into #temp tables"
    assert "into #" in sql.lower()


# --------------------------------------------------------------------------- #
# The guard must reject every shape of warehouse write                        #
# --------------------------------------------------------------------------- #
# The first entry is the exact statement that was removed from the original batch.
WRITES = {
    "the original's warehouse UPDATE":
        "UPDATE pbi.als_demand_tab SET DisplaySKU = Itemnmbr + '*' "
        "WHERE Discontinued = 1;",
    "insert into a permanent table":
        "insert into pbi.als_demand_tab (SKU) select SKU from #gp_pos;",
    "delete from a permanent table":
        "delete from als.open_order_master where WeekDate < '2023-01-01';",
    "merge into a permanent table":
        "merge into pbi.retailer_master as t using #rep as s on t.id = s.id "
        "when matched then update set t.name = s.name;",
    "truncate a permanent table":
        "truncate table pbi.calendar;",
    "select into a permanent table":
        "select Custnmbr, SKU into RealTable from #gp_pos;",
    "drop a permanent table":
        "drop table pbi.als_demand_tab;",
    "create a permanent table":
        "create table dbo.scratch (id int);",
    "unqualified update":
        "update retailer_master set Country = 'US';",
}


@pytest.mark.parametrize("sql", list(WRITES.values()), ids=list(WRITES))
def test_guard_rejects_warehouse_writes(sql):
    with pytest.raises(PermissionError, match="read-only|WRITE"):
        assert_sql_is_read_only(sql, "test.sql")


def test_rejection_names_the_offending_target():
    """The error has to say WHAT it refused, or nobody can act on it."""
    with pytest.raises(PermissionError) as exc:
        assert_sql_is_read_only(
            "UPDATE pbi.als_demand_tab SET x = 1;", "demand.sql")
    message = str(exc.value)
    assert "pbi.als_demand_tab" in message
    assert "demand.sql" in message
    assert "Refusing to connect" in message


def test_guard_runs_before_any_connection(tmp_path, monkeypatch):
    """A bad batch must be refused at READ time, never at execute time."""
    import extract_demand_details as ed

    def _explode(*a, **k):                      # pragma: no cover - must not run
        raise AssertionError("connect() was called despite a write in the SQL")

    monkeypatch.setattr(ed, "connect", _explode)
    bad = tmp_path / "bad.sql"
    bad.write_text("UPDATE pbi.als_demand_tab SET x = 1;", encoding="utf-8")
    with pytest.raises(PermissionError):
        ed.read_sql_file(str(bad))


# --------------------------------------------------------------------------- #
# ...while still allowing everything the real pipeline legitimately does      #
# --------------------------------------------------------------------------- #
SAFE = {
    "aliased temp-table update (the form the live SQL uses)":
        "UPDATE gp SET Custnmbr = 'x' FROM #gp_pos gp "
        "LEFT JOIN pbi.retailer_master cr ON gp.Customer = cr.Custnmbr;",
    "aliased temp-table update with AS":
        "UPDATE g SET a = 1 FROM #gp_pos AS g;",
    "direct temp-table update":
        "UPDATE #gp_pos SET Custnmbr = NULL WHERE Custnmbr = 'XXX';",
    "insert into temp":
        "insert into #rep (Custnmbr) select distinct customer from #projection;",
    "select into temp":
        "select Custnmbr, SKU into #gp_pos from pbi.als_demand_tab;",
    "drop temp table if exists":
        "drop table if exists #projection;",
    "global temp table":
        "select 1 into ##shared;",
    "table variable":
        "declare @t table (id int); insert into @t values (1);",
    "plain select from warehouse":
        "select * from pbi.calendar where TheDate = '2023-01-01';",
    "cte then temp insert":
        "with c as (select 1 as x) select x into #tmp from c;",
}


@pytest.mark.parametrize("sql", list(SAFE.values()), ids=list(SAFE))
def test_guard_allows_legitimate_read_only_sql(sql):
    assert_sql_is_read_only(sql, "test.sql")


def test_a_comment_mentioning_update_is_not_a_write():
    """The shipped file's own header discusses the removed UPDATE in prose."""
    assert_sql_is_read_only(
        "-- READ-ONLY: the original ended with an UPDATE of pbi.als_demand_tab\n"
        "/* insert into pbi.foo -- also just prose */\n"
        "select 1 from pbi.calendar;",
        "test.sql",
    )


def test_a_string_literal_mentioning_update_is_not_a_write():
    assert_sql_is_read_only(
        "select 'update pbi.als_demand_tab set x=1' as note from pbi.calendar;",
        "test.sql",
    )
