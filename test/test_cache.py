"""The on-disk numba cache, as the three viewers and is_null_struct use it.

numba names a cache entry after the function's qualname and source line, so
the viewers one factory makes shared one index file and one set of data
files, and numba writes those without a lock. Processes importing together on
a cold cache read one index and picked the same data-file name for different
viewers, so every process afterwards loaded the wrong machine code: an int32
column read as float64, or a crash in NRT_adapt_ndarray_to_python. A Spark
executor starting its Python workers on a node with an empty cache is exactly
that shape.

``is_null_struct`` shared an index file the other way: lazily typed, it took
one entry per signature, and numba names the next data file by counting the
entries in the index it just read. Its signature is explicit, so there is one
entry there too, whatever a caller hands it.

Every check here runs in a subprocess with its own NUMBA_CACHE_DIR, so the
cache under test is the one the subprocess wrote and nothing else.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

IMPORT_AND_VIEW = (
    "import numpy as np\n"
    "from numbarrow.utils.utils import arrays_viewers\n"
    "src = np.array([-1, 0, 2147483647], dtype=np.int32)\n"
    "assert arrays_viewers[np.int32](src.ctypes.data, 3).tolist() == src.tolist()\n"
)

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
    """A child that calls is_null_struct with one index type and one bitmap pair."""
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
    # PYTHONPATH names the tree under test: from a neutral cwd the child would
    # otherwise import whichever numbarrow its interpreter finds installed.
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
    """The compiled-code files numba wrote for one function, one per index entry."""
    return sorted(path.name for path in Path(cache_dir).rglob("*.nbc") if function in path.name)


def test_each_viewer_has_its_own_cache_index(tmp_path):
    out = _run(IMPORT_AND_VIEW, _env(tmp_path / "cache", {"cache": True}), tmp_path)
    assert out.returncode == 0, out.stderr
    names = _index_files(tmp_path / "cache")
    viewers = [name for name in names if "numpy_array_from_ptr_factory" in name]
    assert len(viewers) == 3, names
    for dtype in ("int32", "int64", "uint8"):
        assert any(f"view_{dtype}" in name for name in viewers), (dtype, viewers)


def test_jit_options_reach_the_decorators(tmp_path):
    # NUMBARROW_JIT_OPTIONS had no end-to-end test: hard-coding the options in
    # the decorators kept every other test green.
    out = _run(IMPORT_AND_VIEW, _env(tmp_path / "cache", {"cache": False}), tmp_path)
    assert out.returncode == 0, out.stderr
    assert _index_files(tmp_path / "cache") == []


def test_a_cold_cache_survives_a_concurrent_first_import(tmp_path):
    # Eight processes importing together on an empty cache directory, then a
    # ninth reading every viewer back from what they wrote.
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
    # Lazily typed it took one entry per signature, and numba picks a data
    # file name by counting the entries in the index it just read, so two
    # processes compiling different signatures could pick the same name. Its
    # signature is explicit instead: every index type and bitmap combination
    # resolves to the one entry compiled at import.
    out = _run(CHECK_EVERY_STRUCT_SHAPE, _env(tmp_path / "cache", {"cache": True}), tmp_path)
    assert out.returncode == 0, out.stderr
    indexes = [name for name in _index_files(tmp_path / "cache") if "is_null_struct" in name]
    assert len(indexes) == 1, indexes
    data = _data_files(tmp_path / "cache", "is_null_struct")
    assert len(data) == 1, data


def test_a_cold_cache_survives_a_concurrent_first_import_of_is_null_struct(tmp_path):
    # Eight processes compiling into one empty cache directory, each calling
    # is_null_struct with an index type and a bitmap pair of its own, then a
    # ninth reading every combination back from what they wrote.
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
