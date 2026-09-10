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
        'if getattr(output, "dtype", None) is not None and output.dtype.kind == "U":',
        "if False:",
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
        "key collision stops raising",
        "numbarrow/core/mapinarrow_factory.py",
        "        if owner is not None:",
        "        if False:",
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


def run_suite(tree: Path, neutral_cwd: Path, cache_dir: Path) -> bool:
    """True when the suite passes. Run from a cwd outside the tree, or the
    real installed package lands on sys.path[0] and shadows this copy, which
    is how a mutation can appear to survive when it was never even loaded."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree)
    # Own cache dir, so this never disturbs a numba cache shared with other work.
    env["NUMBA_CACHE_DIR"] = str(cache_dir)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(tree / "test"), "-x", "-q",
         "-p", "no:cacheprovider"],
        cwd=str(neutral_cwd), env=env, capture_output=True, text=True)
    return proc.returncode == 0


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
        if not run_suite(baseline, neutral, cache):
            print("FAILS")
            print("The unmutated suite does not pass, so mutation results would be "
                  "meaningless. Fix the suite first.")
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
            survived = run_suite(tree, neutral, cache)
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
