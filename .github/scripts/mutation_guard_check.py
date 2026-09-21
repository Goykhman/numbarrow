#!/usr/bin/env python3
"""Require the suite to fail when any guard loses one of its terms.

A guard is usually a conjunction, and a test that exercises the guard as a
whole can pass while one conjunct is dead weight. That is not hypothetical
here: `trailing_nul = non_empty & live & (raw[last_byte] == 0)` had a test
whose comment named the `live` term, and deleting `live` left all 97 tests
green, because the array the test built had no bytes under its null slot.

So each entry below deletes exactly one term and asserts the suite notices.
A mutation whose `old` text is no longer present is an ERROR rather than a
skip: the guard was rewritten and this catalogue has to be updated with it,
which is what stops the catalogue silently ageing into uselessness.

Run:  python .github/scripts/mutation_guard_check.py [--repo DIR]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# (label, relative file, text to replace, replacement)
MUTATIONS = [
    (
        "trailing-NUL guard drops the `live` term",
        "numbarrow/utils/arrow_array_utils.py",
        "trailing_nul = non_empty & live & (raw[last_byte] == 0)",
        "trailing_nul = non_empty & (raw[last_byte] == 0)",
    ),
    (
        "trailing-NUL guard drops the `non_empty` term",
        "numbarrow/utils/arrow_array_utils.py",
        "trailing_nul = non_empty & live & (raw[last_byte] == 0)",
        "trailing_nul = live & (raw[last_byte] == 0)",
    ),
    (
        "`non_empty` loosened from > to >=",
        "numbarrow/utils/arrow_array_utils.py",
        "non_empty = bounds[1:] > bounds[:n]",
        "non_empty = bounds[1:] >= bounds[:n]",
    ),
    (
        "list guard stops checking null rows",
        "numbarrow/utils/arrow_array_utils.py",
        "    if list_array.null_count:\n        raise NotImplementedError(",
        "    if False:\n        raise NotImplementedError(",
    ),
    (
        "list guard stops checking row widths",
        "numbarrow/utils/arrow_array_utils.py",
        "        if low != high:",
        "        if False:",
    ),
    (
        "struct children stop seeing struct-level validity",
        "numbarrow/utils/arrow_array_utils.py",
        "    masked = list(struct_array.flatten()) if struct_array.null_count else raw_children",
        "    masked = raw_children",
    ),
    (
        "output path stops routing unicode via tolist()",
        "numbarrow/core/mapinarrow_factory.py",
        '    if kind == "U":',
        "    if False:",
    ),
    (
        "output path stops routing bytes via tolist()",
        "numbarrow/core/mapinarrow_factory.py",
        '    if kind == "S":',
        "    if False:",
    ),
    (
        "unicode output stops naming its type",
        "numbarrow/core/mapinarrow_factory.py",
        "        return pa.array(value.tolist(), type=arrow_type or pa.string())",
        "        return pa.array(value.tolist(), type=arrow_type)",
    ),
    (
        "a day-unit datetime64 stops being widened for its declared timestamp",
        "numbarrow/core/mapinarrow_factory.py",
        '    if value.dtype == np.dtype("datetime64[D]") and arrow_type is not None '
        "and pa.types.is_timestamp(arrow_type):",
        "    if False:",
    ),
    (
        "a dict under one output key stops being refused",
        "numbarrow/core/mapinarrow_factory.py",
        "    if isinstance(value, Mapping):",
        "    if False:",
    ),
    (
        "a struct dict key no field has stops being refused",
        "numbarrow/core/mapinarrow_factory.py",
        "        if unexpected_keys:\n            raise ValueError(\n"
        "                f\"declared {type_repr(arrow_type)} but the dicts",
        "        if False:\n            raise ValueError(\n"
        "                f\"declared {type_repr(arrow_type)} but the dicts",
    ),
    (
        "a struct key inside a struct field stops being checked",
        "numbarrow/core/mapinarrow_factory.py",
        "            if _carries_keys(child_type):",
        "            if False:",
    ),
    (
        "a struct key inside a list stops being checked",
        "numbarrow/core/mapinarrow_factory.py",
        "    elif _is_list_like(arrow_type):\n        _check_keys(",
        "    elif False:\n        _check_keys(",
    ),
    (
        "the key check stops passing over a row it cannot look inside",
        "numbarrow/core/mapinarrow_factory.py",
        '        if hasattr(row, "__iter__")\n',
        "        if True\n",
    ),
    (
        "the key check spreads a str or bytes row again",
        "numbarrow/core/mapinarrow_factory.py",
        "        and not isinstance(row, (str, bytes))\n",
        "",
    ),
    (
        "the key check spreads a numeric ndarray row again",
        "numbarrow/core/mapinarrow_factory.py",
        '        and not (isinstance(row, np.ndarray) and row.dtype.kind != "O")\n',
        "",
    ),
    (
        "a struct key inside a map's keys stops being checked",
        "numbarrow/core/mapinarrow_factory.py",
        "        if _carries_keys(arrow_type.key_type):",
        "        if False:",
    ),
    (
        "a ready-built array whose fields differ stops being refused",
        "numbarrow/core/mapinarrow_factory.py",
        "        if unexpected:",
        "        if False:",
    ),
    (
        "output columns of different lengths stop being refused",
        "numbarrow/core/mapinarrow_factory.py",
        "    if len(set(lengths.values())) > 1:",
        "    if False:",
    ),
    (
        "a ChunkedArray stops being named as such",
        "numbarrow/core/adapters.py",
        "    if isinstance(pa_array, pa.ChunkedArray):",
        "    if False:",
    ),
    (
        "a struct child's failure stops naming the field",
        "numbarrow/utils/arrow_array_utils.py",
        '            raise renamed(exc, f"struct field {field_name!r}") from exc',
        "            raise",
    ),
    (
        "invalid UTF-8 stops naming the element",
        "numbarrow/utils/arrow_array_utils.py",
        "        except UnicodeDecodeError as exc:",
        "        except ():",
    ),
    (
        "a string input_columns stops being refused",
        "numbarrow/core/mapinarrow_factory.py",
        "    if isinstance(input_columns, str):",
        "    if False:",
    ),
    (
        "an output_schema that is not a pyarrow Schema stops being refused",
        "numbarrow/core/mapinarrow_factory.py",
        "    if output_schema is not None and not isinstance(output_schema, pa.Schema):",
        "    if False:",
    ),
    (
        "a missing input column stops naming the batch's columns",
        "numbarrow/core/mapinarrow_factory.py",
        "                if col not in names:",
        "                if False:",
    ),
    (
        "a column the batch carries twice stops being refused",
        "numbarrow/core/mapinarrow_factory.py",
        "                if names.count(col) > 1:",
        "                if False:",
    ),
    (
        "an adapter failure stops naming the column",
        "numbarrow/core/mapinarrow_factory.py",
        '                    raise renamed(exc, f"column {col!r}") from exc',
        "                    raise",
    ),
    (
        "a record array stops becoming a struct",
        "numbarrow/core/mapinarrow_factory.py",
        "    if value.dtype.names is not None:",
        "    if False:",
    ),
    (
        "a non-dict UDF result stops being refused",
        "numbarrow/core/mapinarrow_factory.py",
        "    if not isinstance(outputs, Mapping):",
        "    if False:",
    ),
    (
        "an output key the schema does not name stops being refused",
        "numbarrow/core/mapinarrow_factory.py",
        "        if extra:",
        "        if False:",
    ),
    (
        "a non-boolean cache option stops being refused",
        "numbarrow/core/configurations.py",
        '    if "cache" in as_json and not isinstance(as_json["cache"], bool):',
        "    if False:",
    ),
    (
        "is_null_struct stops being compiled with one signature",
        "numbarrow/core/is_null.py",
        '@njit(boolean(int64, Optional(Array(uint8, 1, "C", readonly=True)),\n'
        '              Optional(Array(uint8, 1, "C", readonly=True))), **jit_options)',
        "@njit(**jit_options)",
    ),
    (
        "viewers stop getting a cache name of their own",
        "numbarrow/utils/utils.py",
        '    viewer.__qualname__ = f"{numpy_array_from_ptr_factory.__qualname__}.<locals>.{name}"',
        "    pass",
    ),
    (
        "uniform view stops being read-only at the buffer",
        "numbarrow/utils/arrow_array_utils.py",
        "        memoryview(data_buf).toreadonly(),",
        "        memoryview(data_buf),",
    ),
    (
        "empty string result stops being read-only",
        "numbarrow/utils/arrow_array_utils.py",
        '        empty = np.empty((0,), dtype="|U1")\n        empty.flags.writeable = False',
        '        empty = np.empty((0,), dtype="|U1")',
    ),
    (
        "empty uniform result stops being read-only",
        "numbarrow/utils/arrow_array_utils.py",
        "        empty = np.empty((0,), dtype=data_np_ty)\n"
        "        # Read-only like the non-empty path below, so that the contract does\n"
        "        # not depend on whether the batch happened to be empty.\n"
        "        empty.flags.writeable = False",
        "        empty = np.empty((0,), dtype=data_np_ty)",
    ),
    (
        "non-empty string result stops being read-only",
        "numbarrow/utils/arrow_array_utils.py",
        "    str_array.flags.writeable = False",
        "    pass",
    ),
    (
        "a Nullable stops being split into data and bitmap",
        "numbarrow/core/mapinarrow_factory.py",
        "    if isinstance(value, Nullable):",
        "    if False:",
    ),
    (
        "a Nullable's bitmap stops being folded in",
        "numbarrow/core/mapinarrow_factory.py",
        "        return _with_validity(array, bitmap)",
        "        return array",
    ),
    (
        "a handed-out bitmap stops being checked against the rows it covers",
        "numbarrow/core/mapinarrow_factory.py",
        "        if covers is not None and len(array) != covers:",
        "        if False:",
    ),
    (
        "a handed-out bitmap's count stops coming from the data beside it",
        "numbarrow/core/mapinarrow_factory.py",
        "                handed[id(bitmap)] = len(datas if field is None else datas[field])",
        "                handed[id(bitmap)] = len(datas)",
    ),
    (
        "a Nullable's bitmap of the wrong length stops being refused",
        "numbarrow/core/mapinarrow_factory.py",
        "    if len(bitmap) != (rows + 7) // 8:",
        "    if False:",
    ),
    (
        "a Nullable's bitmap that is not packed uint8 stops being refused",
        "numbarrow/core/mapinarrow_factory.py",
        "    if bitmap.dtype != np.uint8 or bitmap.ndim != 1:",
        "    if False:",
    ),
    (
        "a Nullable's bitmap that is not an ndarray stops being refused",
        "numbarrow/core/mapinarrow_factory.py",
        "    if not isinstance(bitmap, np.ndarray):",
        "    if False:",
    ),
    (
        "a dictionary column stops being kept off the flat path",
        "numbarrow/core/mapinarrow_factory.py",
        "and not pa.types.is_dictionary(array.type)",
        "and True",
    ),
]

COPY = ["numbarrow", "test", "README.md", "docs"]


def build_tree(repo: Path, dest: Path):
    for name in COPY:
        src = repo / name
        if not src.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dest / name,
                            ignore=shutil.ignore_patterns("__pycache__", "_build", "*.pyc"))
        else:
            shutil.copy2(src, dest / name)


def run_suite(tree: Path, neutral_cwd: Path, cache_dir: Path) -> tuple[bool, str]:
    """Whether the suite passes, and the tail of what it printed.

    Run from a cwd outside the tree, or the real installed package lands on
    sys.path[0] and shadows this copy, which is how a mutation can appear to
    survive when it was never even loaded. The tail is what a red job has to
    show: without it a failing baseline named nothing."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree)
    # Own cache dir, so this never disturbs a numba cache shared with other work.
    env["NUMBA_CACHE_DIR"] = str(cache_dir)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(tree / "test"), "-x", "-q",
         "-p", "no:cacheprovider"],
        cwd=str(neutral_cwd), env=env, capture_output=True, text=True)
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
    return proc.returncode == 0, tail


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    args = ap.parse_args(argv)
    repo = Path(args.repo).resolve()

    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        neutral = tmpdir / "cwd"
        neutral.mkdir()
        cache = tmpdir / "numba-cache"
        cache.mkdir()
        baseline = tmpdir / "baseline"
        baseline.mkdir()
        build_tree(repo, baseline)
        print("baseline: ", end="", flush=True)
        passes, tail = run_suite(baseline, neutral, cache)
        if not passes:
            print("FAILS")
            print("The unmutated suite does not pass, so mutation results would be "
                  "meaningless. Fix the suite first. The suite's last lines:")
            print(tail)
            return 1
        print("passes")

        for i, (label, rel, old, new) in enumerate(MUTATIONS):
            tree = tmpdir / f"m{i}"
            tree.mkdir()
            build_tree(repo, tree)
            target = tree / rel
            text = target.read_text()
            if text.count(old) != 1:
                failures.append(
                    f"STALE  {label}\n"
                    f"       expected exactly one occurrence in {rel}, found {text.count(old)}.\n"
                    f"       The guard was rewritten; update this catalogue to match it.")
                print(f"  [{i + 1}/{len(MUTATIONS)}] STALE   {label}")
                continue
            target.write_text(text.replace(old, new))
            survived, _tail = run_suite(tree, neutral, cache)
            if survived:
                failures.append(
                    f"SURVIVED  {label}\n"
                    f"          in {rel}, the suite still passes without this term, so "
                    f"nothing tests it.")
                print(f"  [{i + 1}/{len(MUTATIONS)}] SURVIVED {label}")
            else:
                print(f"  [{i + 1}/{len(MUTATIONS)}] killed   {label}")

    if failures:
        print("\n" + "=" * 70)
        for f in failures:
            print(f)
        print("=" * 70)
        print(f"{len(failures)} of {len(MUTATIONS)} mutations were not killed.")
        return 1
    print(f"\nAll {len(MUTATIONS)} mutations killed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
