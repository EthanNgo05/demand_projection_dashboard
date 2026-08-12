"""Every name a module references at global scope must be bound or imported.

Streamlit renders views lazily, so a name that was never imported is a NameError
that fires only when the one view using it is opened — and nothing else in this
suite opens a view. That is how ``kpis.py`` shipped a call to
``with_export_flags`` without importing it and crashed the Optimal Projections
page two days later.

The check is a symtable walk rather than a lint dependency: for each module,
collect the names referenced as globals in any scope and subtract the ones bound
at module level, plus builtins. It costs about a second for all of src/.
"""
import builtins
import pathlib
import symtable

import pytest


SRC_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src"
SOURCE_FILES = sorted(SRC_ROOT.rglob("*.py"))
_BUILTINS = frozenset(dir(builtins))


def _undefined_globals(path):
    """Names ``path`` reads at global scope but never binds. Empty set if clean.

    Lets SyntaxError propagate — a module that won't parse is a worse version of
    the failure this test is here to catch, not something to skip past.
    """
    table = symtable.symtable(path.read_text(encoding="utf-8"), str(path), "exec")
    bound = {
        sym.get_name()
        for sym in table.get_symbols()
        if sym.is_assigned() or sym.is_imported() or sym.is_namespace()
    }
    found = set()

    def walk(scope):
        for sym in scope.get_symbols():
            name = sym.get_name()
            if (
                sym.is_global()
                and not sym.is_assigned()
                and name not in bound
                and name not in _BUILTINS
                # Dunders are module machinery the compiler supplies, never
                # imports: __file__, and 3.14's __conditional_annotations__.
                and not (name.startswith("__") and name.endswith("__"))
            ):
                found.add(name)
        for child in scope.get_children():
            walk(child)

    walk(table)
    return found


@pytest.mark.parametrize(
    "path", SOURCE_FILES, ids=[str(p.relative_to(SRC_ROOT)) for p in SOURCE_FILES]
)
def test_module_binds_every_global_name_it_references(path):
    undefined = _undefined_globals(path)
    assert not undefined, (
        f"{path.relative_to(SRC_ROOT)} references {sorted(undefined)} without "
        "importing or defining it — this raises NameError at runtime, whenever "
        "the code path happens to run."
    )


def test_the_scan_actually_finds_a_missing_import(tmp_path):
    """Guard the guard: a suite of vacuous passes would look identical."""
    clean = tmp_path / "clean.py"
    clean.write_text("import os\n\n\ndef f():\n    return os.getcwd()\n", encoding="utf-8")
    assert _undefined_globals(clean) == set()

    broken = tmp_path / "broken.py"
    broken.write_text("def f(df):\n    return with_export_flags(df)\n", encoding="utf-8")
    assert _undefined_globals(broken) == {"with_export_flags"}
