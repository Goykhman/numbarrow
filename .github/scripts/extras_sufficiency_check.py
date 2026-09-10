#!/usr/bin/env python3
"""Install each optional-dependency extra alone and exercise what it promises.

An extra is a promise: install this and the named feature works. CI installs
the `test` extra, which happens to pin setuptools, so for a long time nothing
noticed that the `mapinarrow` extra did not. A user installing
`numbarrow[mapinarrow]` on the declared minimum Python got
`ModuleNotFoundError: No module named 'distutils'` the first time they called
mapInArrow, because PEP 632 removed distutils from the 3.12 standard library
and setuptools is its only supplier.

A green `test` job cannot see that, by construction. Only installing each extra
on its own can, which is what this does.

Run:  python .github/scripts/extras_sufficiency_check.py [--repo DIR] [--extra NAME]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

# What each extra has to be able to do, once installed on its own. Keep the
# probe to imports and cheap calls: this runs per extra in a fresh environment.
PROBES = {
    "mapinarrow": (
        "import numbarrow\n"
        "from numbarrow.core.mapinarrow_factory import make_mapinarrow_func\n"
        "import pandas\n"
        # pyspark is deliberately not in this extra, the user brings their own
        # cluster's version. But pyspark's mapInArrow path imports distutils,
        # and on the declared minimum Python only setuptools supplies it, so
        # the extra has to carry setuptools whether or not it carries pyspark.
        "import distutils.version\n"
        "print('mapinarrow probe ok')\n"
    ),
    "docs": (
        "import sphinx, sphinx_sitemap, sphinx_rtd_theme\n"
        "print('docs probe ok')\n"
    ),
}

# The test extra is what CI already installs and exercises with the full suite,
# so re-running it here would only duplicate that job.
SKIP = {"test"}


def make_env(root: Path, python: str):
    """A fresh environment with only this extra installed. uv when available,
    since a venv plus install is seconds rather than a minute."""
    venv = root / "venv"
    uv = shutil.which("uv")
    if uv:
        subprocess.run([uv, "venv", "--python", python, str(venv)],
                       check=True, capture_output=True)
        return venv, [uv, "pip", "install", "--python", str(venv / "bin" / "python")]
    subprocess.run([python, "-m", "venv", str(venv)], check=True, capture_output=True)
    return venv, [str(venv / "bin" / "pip"), "install"]


def check(repo: Path, extra: str, probe: str, python: str) -> str | None:
    """None when the extra is sufficient, else the reason it is not."""
    with tempfile.TemporaryDirectory() as tmp:
        venv, installer = make_env(Path(tmp), python)
        install = subprocess.run(installer + [f"{repo}[{extra}]"],
                                 capture_output=True, text=True)
        if install.returncode != 0:
            return f"install of .[{extra}] failed:\n{install.stderr[-1200:]}"
        script = Path(tmp) / "probe.py"
        script.write_text(probe)
        # Run from a neutral cwd, or the repo lands on sys.path[0] and the
        # source tree satisfies imports the installed extra does not.
        run = subprocess.run([str(venv / "bin" / "python"), str(script)],
                             cwd=tmp, capture_output=True, text=True,
                             env={"PATH": os.environ.get("PATH", ""), "HOME": tmp})
        if run.returncode != 0:
            return (f"installed .[{extra}] cannot do what it promises:\n"
                    f"{run.stderr.strip()[-1200:]}")
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--extra", action="append",
                    help="only these extras (default: every one with a probe)")
    ap.add_argument("--python", default=None,
                    help="interpreter for the fresh environments; default is "
                         "the floor from requires-python")
    args = ap.parse_args(argv)
    repo = Path(args.repo).resolve()

    meta = tomllib.loads((repo / "pyproject.toml").read_text())["project"]
    declared = set(meta.get("optional-dependencies", {}))
    # The floor is what matters: distutils is present below 3.12 and absent at
    # and above it, so testing a newer interpreter would hide the failure and
    # testing the floor exposes it.
    python = args.python or "python" + meta["requires-python"].lstrip(">=~^ ")

    unprobed = declared - set(PROBES) - SKIP
    if unprobed:
        print(f"FAIL: extras with no probe here: {sorted(unprobed)}")
        print("      Every extra is a promise; add what it must be able to do.")
        return 1
    stale = set(PROBES) - declared
    if stale:
        print(f"FAIL: probes for extras that no longer exist: {sorted(stale)}")
        return 1

    wanted = args.extra or sorted(PROBES)
    failures = []
    for extra in wanted:
        print(f"  {extra}: ", end="", flush=True)
        reason = check(repo, extra, PROBES[extra], python)
        print("ok" if reason is None else "FAIL")
        if reason:
            failures.append(f"[{extra}] {reason}")

    if failures:
        print("\n" + "=" * 70)
        for f in failures:
            print(f)
        return 1
    print(f"\nAll {len(wanted)} extras are sufficient on {python}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
