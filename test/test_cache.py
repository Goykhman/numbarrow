"""Tests of how numbarrow caches its compiled functions, of the options its decorators take, and of the cache fallback.

numba keeps an index file (``.nbi``) for each cached function, named after its module, its qualified name and its
first source line. The index records which data file (``.nbc``) holds the machine code of each compiled signature, and
numba updates both kinds of file without a lock. If two processes compile different entries into one index at the
same time, they can give both entries one data-file name, and a later process can then load the wrong machine code. A
Spark executor that starts several Python workers on a node with an empty cache can meet this case. numbarrow avoids
it in two ways. Each viewer, the compiled function that reads raw memory as a numpy array of one dtype, has a
qualified name of its own and therefore its own index, and ``is_null_struct`` has one explicit signature and therefore
one entry. Some tests check both layouts, and two of them run eight processes at once on an empty cache. Other tests
check that ``NUMBARROW_JIT_OPTIONS`` reaches the decorators. The rest check the fallback for an import that numba
cannot cache, from an archive or from a read-only install. numbarrow then compiles without a cache, and its warning
says how to get a cache or how to turn caching off. numba 0.61 and later cache an import from a ``.zip``. A test that
needs a directory it cannot write to is skipped on Windows and when run as root. Each test runs numbarrow in a child
process with cache directories of its own, because this process imported numbarrow and read numba's settings before
the test began. ``test_configurations.py`` tests how the options are parsed, and which errors the fallback catches,
with numba's ``njit`` replaced by a stub.
"""
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# On Windows, chmod does not stop a process from writing in a directory, and root writes in a directory whatever its
# mode. A test that needs a directory that it cannot write in is skipped in both cases.
needs_a_directory_it_cannot_write = pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0, reason="needs a directory this user cannot write to")

# The child imports the adapters, which import is_null.py, and numba compiles that module's three functions at
# import because their signatures are explicit. The child prints the file it imported the adapters from, so that a
# test can check that it imported numbarrow from the place that PYTHONPATH names.
IMPORT_AND_SHOW_FILE = "import numbarrow.core.adapters as a; print(a.__file__)"

IMPORT_AND_VIEW = (
    "import numpy as np\n"
    "from numbarrow.utils.utils import arrays_viewers\n"
    "src = np.array([-1, 0, 2147483647], dtype=np.int32)\n"
    "assert arrays_viewers[np.int32](src.ctypes.data, 3).tolist() == src.tolist()\n"
)

# The three dtypes are the ones the adapters ask for: uint8 for validity bitmaps and packed booleans, and int32 and
# int64 for string offsets.
CHECK_EVERY_VIEWER = (
    "import numpy as np\n"
    "from numbarrow.utils.utils import arrays_viewers\n"
    "values = {np.int32: [-1, 0, 2147483647], np.int64: [-1, 0, 2 ** 63 - 1],\n"
    "          np.uint8: [0, 127, 255]}\n"
    "for dtype, vals in values.items():\n"
    "    src = np.array(vals, dtype=dtype)\n"
    "    view = arrays_viewers[dtype](src.ctypes.data, src.size)\n"
    "    assert view.dtype == np.dtype(dtype) and view.tolist() == vals, (dtype, view.dtype, view.tolist())\n"
)


# The child calls is_null_struct with each of eight index types, and with the bitmap or None in each of the two bitmap
# arguments. Row 0 is null in the bitmap, so every call returns True unless both arguments are None.
CHECK_EVERY_STRUCT_SHAPE = (
    "import numpy as np\n"
    "from numbarrow.core.is_null import is_null_struct\n"
    "bitmap = np.array([0b00000010], dtype=np.uint8)\n"
    "index_types = [np.int64, np.int32, np.int16, np.int8, np.uint8, np.uint16, np.uint32, np.uint64]\n"
    "pairs = [(bitmap, bitmap), (None, bitmap), (bitmap, None), (None, None)]\n"
    "for index_type in index_types:\n"
    "    for struct_bitmap, field_bitmap in pairs:\n"
    "        expected = not (struct_bitmap is None and field_bitmap is None)\n"
    "        got = is_null_struct(index_type(0), struct_bitmap, field_bitmap)\n"
    "        assert got == expected, (index_type, struct_bitmap is None, field_bitmap is None, got)\n"
)


def _one_struct_shape(variant):
    """Return a child script that calls is_null_struct with the index type and bitmap pair that ``variant`` picks."""
    return (
        "import numpy as np\n"
        "from numbarrow.core.is_null import is_null_struct\n"
        "bitmap = np.array([0b00000010], dtype=np.uint8)\n"
        "index_types = [np.int64, np.int32, np.int16, np.int8, np.uint8, np.uint16, np.uint32, np.uint64]\n"
        "pairs = [(bitmap, bitmap), (None, bitmap), (bitmap, None), (None, None)]\n"
        f"struct_bitmap, field_bitmap = pairs[{variant} % 4]\n"
        f"index = index_types[{variant}](0)\n"
        "expected = not (struct_bitmap is None and field_bitmap is None)\n"
        "assert is_null_struct(index, struct_bitmap, field_bitmap) == expected\n"
    )


def _env(cache_dir, options):
    # The child runs in a directory outside the checkout, so without PYTHONPATH it would import whatever numbarrow its
    # interpreter has installed. PYTHONPATH points it at the tree under test.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    env["NUMBA_CACHE_DIR"] = str(cache_dir)
    env["NUMBARROW_JIT_OPTIONS"] = json.dumps(options)
    return env


def _run(src, env, cwd):
    return subprocess.run([sys.executable, "-c", src], capture_output=True, text=True,
                          env=env, cwd=str(cwd))


def _index_files(cache_dir):
    return sorted(path.name for path in Path(cache_dir).rglob("*.nbi"))


def _data_files(cache_dir, function):
    """Return the sorted names of the data files that numba wrote for ``function``, one per entry of its index."""
    return sorted(path.name for path in Path(cache_dir).rglob("*.nbc") if function in path.name)


def _viewer_index_files(cache_dir):
    return [name for name in _index_files(cache_dir) if "numpy_array_from_ptr_factory" in name]


def test_each_viewer_has_its_own_cache_index(tmp_path):
    """Check that a viewer is compiled when its dtype is first asked for, into an index file named after that dtype.

    One request leaves one viewer index, and the three dtypes the adapters use leave three. An index shared by several
    viewers can let processes that warm a cold cache together load the wrong viewer's machine code.
    """
    env = _env(tmp_path / "cache", {"cache": True})
    out = _run(IMPORT_AND_VIEW, env, tmp_path)
    assert out.returncode == 0, out.stderr
    viewers = _viewer_index_files(tmp_path / "cache")
    assert len(viewers) == 1 and "view_int32" in viewers[0], viewers
    out = _run(CHECK_EVERY_VIEWER, env, tmp_path)
    assert out.returncode == 0, out.stderr
    viewers = _viewer_index_files(tmp_path / "cache")
    assert len(viewers) == 3, viewers
    for dtype in ("int32", "int64", "uint8"):
        assert any(f"view_{dtype}" in name for name in viewers), (dtype, viewers)


def test_jit_options_reach_the_decorators(tmp_path):
    """Check that ``{"cache": false}`` in ``NUMBARROW_JIT_OPTIONS`` leaves the cache directory empty after a viewer
    compiles. A viewer decorator that ignored the variable would still pass every other test."""
    out = _run(IMPORT_AND_VIEW, _env(tmp_path / "cache", {"cache": False}), tmp_path)
    assert out.returncode == 0, out.stderr
    assert _index_files(tmp_path / "cache") == []


def test_a_cold_cache_survives_a_concurrent_first_import(tmp_path):
    """Check that eight processes asking for the int32 viewer at once, on one empty cache directory, all succeed, and
    that a ninth then reads int32, int64 and uint8 arrays correctly with what they left. This is the first import on a
    Spark executor that starts several Python workers on a node with an empty cache."""
    env = _env(tmp_path / "cache", {"cache": True})
    procs = [
        subprocess.Popen([sys.executable, "-c", IMPORT_AND_VIEW], env=env, cwd=str(tmp_path),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for _ in range(8)
    ]
    errors = [proc.communicate(timeout=600)[1] for proc in procs]
    failed = [err for proc, err in zip(procs, errors) if proc.returncode != 0]
    assert not failed, failed[0]
    out = _run(CHECK_EVERY_VIEWER, env, tmp_path)
    assert out.returncode == 0, out.stderr


def test_is_null_struct_has_one_cache_entry_for_every_shape(tmp_path):
    """Check that every index type and bitmap pair resolves to the one signature that is_null_struct compiles at
    import, so that its cache holds one index file and one data file.

    numba names a new data file with the lowest number that no entry in the index it just read uses. With an entry per
    signature, two processes compiling different signatures into one cold index could therefore pick the same name.
    """
    out = _run(CHECK_EVERY_STRUCT_SHAPE, _env(tmp_path / "cache", {"cache": True}), tmp_path)
    assert out.returncode == 0, out.stderr
    indexes = [name for name in _index_files(tmp_path / "cache") if "is_null_struct" in name]
    assert len(indexes) == 1, indexes
    data = _data_files(tmp_path / "cache", "is_null_struct")
    assert len(data) == 1, data


def test_a_cold_cache_survives_a_concurrent_first_import_of_is_null_struct(tmp_path):
    """Check that eight processes on one empty cache directory, each calling is_null_struct with an index type of its
    own and one of the four bitmap pairs, all succeed, and that a ninth then gets the right answer for every
    combination. If is_null_struct were typed lazily, each process would compile a signature of its own into the
    shared index."""
    env = _env(tmp_path / "cache", {"cache": True})
    procs = [
        subprocess.Popen([sys.executable, "-c", _one_struct_shape(variant)], env=env,
                         cwd=str(tmp_path), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for variant in range(8)
    ]
    errors = [proc.communicate(timeout=600)[1] for proc in procs]
    failed = [err for proc, err in zip(procs, errors) if proc.returncode != 0]
    assert not failed, failed[0]
    out = _run(CHECK_EVERY_STRUCT_SHAPE, env, tmp_path)
    assert out.returncode == 0, out.stderr


def _archive(path):
    """Write numbarrow's modules into a zip archive at ``path`` and return ``path``, for the caller's PYTHONPATH.

    Python imports from the archive whatever its suffix, but numba goes by the name: numba 0.61 and later cache a
    module from an archive only when the archive's name ends in ``.zip``.
    """
    with zipfile.ZipFile(path, "w") as zipped:
        for source in sorted((REPO / "numbarrow").rglob("*.py")):
            zipped.write(source, str(source.relative_to(REPO)))
    return path


@pytest.mark.parametrize("name", ["numbarrow-0.0.0-py3.12.egg", "numbarrow-0.0.0-py3-none-any.whl"])
def test_an_import_from_an_archive_compiles_uncached_with_a_warning_naming_the_remedy(tmp_path, name):
    """Check that an import from an ``.egg`` or a ``.whl`` compiles without a cache and warns that ``NUMBA_CACHE_DIR``
    has no effect, and that with caching off it gives no warning.

    numba finds no cache location for a module inside such an archive. Without the fallback, numba would raise
    ``RuntimeError`` at the first decorated function, and its message names neither a way to a cache nor the option
    that turns caching off. Spark's ``--py-files`` puts ``.egg`` archives on its workers' Python path.
    """
    # NUMBA_CACHE_DIR names a writable directory, and numba writes nothing there: numba reads it only for a source
    # file on disk, so the warning does not offer it.
    archive = _archive(tmp_path / name)
    env = dict(os.environ, PYTHONPATH=str(archive), NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBARROW_JIT_OPTIONS", None)
    run = subprocess.run([sys.executable, "-W", "always", "-c", IMPORT_AND_SHOW_FILE],
                         capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
    assert "compiles without a cache" in run.stderr
    assert "NUMBA_CACHE_DIR has no effect here" in run.stderr and "Set NUMBA_CACHE_DIR" not in run.stderr
    assert _index_files(tmp_path / "cache") == []
    # With caching off, numbarrow does not try to cache, and -W error turns any warning into a failure of the child.
    quiet = subprocess.run([sys.executable, "-W", "error", "-c", IMPORT_AND_SHOW_FILE],
                           capture_output=True, text=True,
                           env=dict(env, NUMBARROW_JIT_OPTIONS='{"cache": false}'), cwd=str(tmp_path))
    assert quiet.returncode == 0, quiet.stderr


def test_a_zip_import_is_cached_by_numba_from_0_61(tmp_path):
    """Check that numba 0.61 and later cache numbarrow imported from a ``.zip``, without a warning, in the user's cache
    directory and not under ``NUMBA_CACHE_DIR``. With numba 0.60, numbarrow compiles a ``.zip`` import without a cache
    and warns, as for any other archive. The warning for an archive offers a ``.zip`` as the way to a cache, and this
    test checks that advice."""
    import numba
    archive = _archive(tmp_path / "numbarrow.zip")
    home = tmp_path / "home"
    env = dict(os.environ, PYTHONPATH=str(archive), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"),
               NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBARROW_JIT_OPTIONS", None)
    run = subprocess.run([sys.executable, "-W", "always", "-c", IMPORT_AND_SHOW_FILE],
                         capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
    cached = tuple(int(part) for part in numba.__version__.split(".")[:2]) >= (0, 61)
    assert ("compiles without a cache" not in run.stderr) == cached, run.stderr
    assert _index_files(tmp_path / "cache") == []
    if os.name != "nt":
        # The check is skipped on Windows, where numba asks the system for the user's cache directory, and HOME and
        # XDG_CACHE_HOME do not move it.
        assert bool(_index_files(home)) == cached


@needs_a_directory_it_cannot_write
def test_a_zip_import_with_no_writable_user_cache_directory_compiles_uncached_with_a_warning(tmp_path):
    """Check that an import from a ``.zip`` under a read-only home compiles without a cache and warns that
    ``NUMBA_CACHE_DIR`` has no effect.

    From version 0.61, numba takes the user's cache directory for a ``.zip`` without checking that it can write there.
    Without the fallback, the first save would raise ``PermissionError`` and fail the import, as on an executor whose
    home is read-only. Before 0.61, numba finds no cache location for a ``.zip`` at all, and the warning is the same.
    """
    archive = _archive(tmp_path / "numbarrow.zip")
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o555)
    try:
        env = dict(os.environ, PYTHONPATH=str(archive), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"),
                   NUMBA_CACHE_DIR=str(tmp_path / "cache"))
        env.pop("NUMBARROW_JIT_OPTIONS", None)
        run = subprocess.run([sys.executable, "-W", "always", "-c", IMPORT_AND_SHOW_FILE],
                             capture_output=True, text=True, env=env, cwd=str(tmp_path))
        assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
        assert "compiles without a cache" in run.stderr and "NUMBA_CACHE_DIR has no effect here" in run.stderr
    finally:
        home.chmod(0o755)


@needs_a_directory_it_cannot_write
def test_a_read_only_install_warns_naming_numba_cache_dir_and_setting_it_caches(tmp_path):
    """Check that a read-only install compiles without a cache and warns with ``Set NUMBA_CACHE_DIR``, and that with
    the variable set to a writable directory the import caches there without a warning. A read-only install has its
    source on disk, but numba can write neither beside it nor in the user's cache directory, so ``NUMBA_CACHE_DIR`` is
    the way to a cache there."""
    site = tmp_path / "site"
    shutil.copytree(REPO / "numbarrow", site / "numbarrow", ignore=shutil.ignore_patterns("__pycache__"))
    home = tmp_path / "home"
    home.mkdir()
    read_only = [home, *(path for path in site.rglob("*") if path.is_dir())]
    for path in read_only:
        path.chmod(0o555)
    try:
        env = dict(os.environ, PYTHONPATH=str(site), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"))
        env.pop("NUMBARROW_JIT_OPTIONS", None)
        env.pop("NUMBA_CACHE_DIR", None)
        run = subprocess.run([sys.executable, "-W", "always", "-c", IMPORT_AND_SHOW_FILE],
                             capture_output=True, text=True, env=env, cwd=str(tmp_path))
        assert run.returncode == 0 and str(site) in run.stdout, run.stderr
        assert "compiles without a cache" in run.stderr and "Set NUMBA_CACHE_DIR" in run.stderr
        cured = subprocess.run([sys.executable, "-W", "error", "-c", IMPORT_AND_SHOW_FILE],
                               capture_output=True, text=True,
                               env=dict(env, NUMBA_CACHE_DIR=str(tmp_path / "cache")), cwd=str(tmp_path))
        assert cured.returncode == 0, cured.stderr
        assert _index_files(tmp_path / "cache")
    finally:
        for path in read_only:
            path.chmod(0o755)


# The child makes is_null and unpack_booleans read past the end of the bitmap, and prints for each call whether it
# returned or raised IndexError. The bitmap is a one-byte slice of an array long enough to hold every byte the calls
# read. Without bounds checking, the calls read past the slice but stay inside memory that the array owns. If the
# bitmap were a one-byte array of its own, the calls would read whatever follows it on the heap, and a read that
# reached an unmapped page would crash the child with an access violation.
CHECK_BOUNDS = (
    "import numpy as np\n"
    "from numbarrow.core.is_null import is_null, unpack_booleans\n"
    "bitmap = np.zeros(100000 // 8 + 1, dtype=np.uint8)[:1]\n"
    "outcomes = []\n"
    "for call in (lambda: is_null(100000, bitmap), lambda: unpack_booleans(0, 100000, bitmap)):\n"
    "    try:\n"
    "        call()\n"
    "        outcomes.append('returned')\n"
    "    except IndexError:\n"
    "        outcomes.append('IndexError')\n"
    "print(' '.join(outcomes))\n"
)

# The child prints the nogil option of each of is_null.py's three dispatchers.
SHOW_NOGIL = (
    "from numbarrow.core.is_null import is_null, unpack_booleans, is_null_struct\n"
    "print(' '.join(str(f.targetoptions.get('nogil')) for f in (is_null, unpack_booleans, is_null_struct)))\n"
)


def test_jit_options_reach_the_is_null_decorators(tmp_path):
    """Check that ``NUMBARROW_JIT_OPTIONS`` reaches each of the three decorators in ``is_null.py``. A decorator that
    hard-coded its options would pass every other test, and only this test checks that ``boundscheck`` turns an index
    past the bitmap into ``IndexError``."""
    # Bounds checking shows on is_null and unpack_booleans, the two functions that index a bitmap. is_null_struct
    # indexes nothing itself: it calls is_null, whose own options decide.
    checked = _run(CHECK_BOUNDS, _env(tmp_path / "checked", {"cache": False, "boundscheck": True}), tmp_path)
    assert checked.returncode == 0 and checked.stdout.split() == ["IndexError", "IndexError"], checked.stderr
    unchecked = _run(CHECK_BOUNDS, _env(tmp_path / "unchecked", {"cache": False}), tmp_path)
    assert unchecked.returncode == 0 and unchecked.stdout.split() == ["returned", "returned"], unchecked.stderr
    # The cache option shows on all three: with caching off, none of them writes to the cache directory, and with
    # caching on, each leaves one index file there.
    assert _index_files(tmp_path / "checked") == [] and _index_files(tmp_path / "unchecked") == []
    cached = _run(CHECK_BOUNDS, _env(tmp_path / "cached", {"cache": True}), tmp_path)
    assert cached.returncode == 0 and cached.stdout.split() == ["returned", "returned"], cached.stderr
    indexes = _index_files(tmp_path / "cached")
    assert len(indexes) == 3, indexes
    # The dash after is_null keeps the name from matching is_null_struct's index file.
    for function in ("is_null-", "unpack_booleans", "is_null_struct"):
        assert any(function in name for name in indexes), (function, indexes)
    # nogil changes nothing that a call can show, so the child reads it back from each dispatcher. This catches a
    # decorator that forwards only the options that show.
    forwarded = _run(SHOW_NOGIL, _env(tmp_path / "nogil", {"cache": False, "nogil": True}), tmp_path)
    assert forwarded.returncode == 0 and forwarded.stdout.split() == ["True", "True", "True"], forwarded.stderr
