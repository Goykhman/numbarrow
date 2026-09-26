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
        '@jit_with_options(boolean(int64, Optional(Array(uint8, 1, "C", readonly=True)),\n'
        '                          Optional(Array(uint8, 1, "C", readonly=True))))',
        "@jit_with_options()",
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
        "                handed[id(bitmap)] = (bitmap, len(datas if field is None else datas[field]))",
        "                handed[id(bitmap)] = (bitmap, len(datas))",
    ),
    (
        "a handed-out bitmap stops being kept alive for the batch",
        "numbarrow/core/mapinarrow_factory.py",
        "                handed[id(bitmap)] = (bitmap, len(datas if field is None else datas[field]))",
        "                handed[id(bitmap)] = (None, len(datas if field is None else datas[field]))",
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
    (
        "a string for a JIT option other than cache stops being refused",
        "numbarrow/core/configurations.py",
        '        if isinstance(value, str) and name not in ("error_model", "inline"):',
        "        if False:",
    ),
    (
        "a generator output stops being read into a list before the key check",
        "numbarrow/core/mapinarrow_factory.py",
        '        if not hasattr(value, "__len__"):',
        "        if False:",
    ),
    (
        "a record array with no fields stops keeping its rows",
        "numbarrow/core/mapinarrow_factory.py",
        "    if not children:",
        "    if False:",
    ),
    (
        "a viewer built on request stops being kept",
        "numbarrow/utils/utils.py",
        "        viewer = self[dtype_] = numpy_array_from_ptr_factory(dtype_)",
        "        viewer = numpy_array_from_ptr_factory(dtype_)",
    ),
    (
        'a struct field inside a struct field stops being checked in a ready-built array',
        'numbarrow/core/mapinarrow_factory.py',
        '                found.extend(_unexpected_fields(field.type, declared[field.name]))',
        '                pass',
    ),
    (
        'a struct field inside a list stops being checked in a ready-built array',
        'numbarrow/core/mapinarrow_factory.py',
        "    if _is_list_like(source_type) and _is_list_like(declared_type):\n"
        "        return _unexpected_fields(source_type.value_type, declared_type.value_type)",
        "    if _is_list_like(source_type) and _is_list_like(declared_type):\n"
        "        return []",
    ),
    (
        'a struct field inside a map stops being checked in a ready-built array',
        'numbarrow/core/mapinarrow_factory.py',
        "        return (_unexpected_fields(source_type.key_type, declared_type.key_type)\n"
        "                + _unexpected_fields(source_type.item_type, declared_type.item_type))",
        '        return []',
    ),
    (
        'a day-unit datetime64 under a declared type stops being inferred first',
        'numbarrow/core/mapinarrow_factory.py',
        '    if value.dtype == np.dtype("datetime64[D]") and arrow_type is not None:',
        '    if False:',
    ),
    (
        'an output key that is not a str stops being refused by name',
        'numbarrow/core/mapinarrow_factory.py',
        '            if not isinstance(name, str):',
        '            if False:',
    ),
    (
        'a struct field inside a dictionary stops being checked in a ready-built array',
        'numbarrow/core/mapinarrow_factory.py',
        '    if pa.types.is_dictionary(source_type) and pa.types.is_dictionary(declared_type):',
        '    if False:',
    ),
    (
        'a record array under a non-struct declared type stops being refused',
        'numbarrow/core/mapinarrow_factory.py',
        "    if not pa.types.is_struct(arrow_type):\n"
        "        raise TypeError",
        "    if False:\n"
        "        raise TypeError",
    ),
    (
        'a ChunkedArray output stops being combined',
        'numbarrow/core/mapinarrow_factory.py',
        "    if isinstance(value, pa.ChunkedArray):\n"
        "        value = value.combine_chunks()",
        "    if False:\n"
        "        value = value.combine_chunks()",
    ),
    (
        'renamed() stops falling back to ValueError',
        'numbarrow/utils/arrow_array_utils.py',
        "    if not (isinstance(exc, pa.ArrowException) or cls in kept):\n"
        "        cls = ValueError",
        "    if False:\n"
        "        cls = ValueError",
    ),
    (
        "map entries stop checking a pair's shape",
        'numbarrow/core/mapinarrow_factory.py',
        '                if isinstance(pair, (tuple, list)) and len(pair) == 2:',
        '                if True:',
    ),
    (
        'a record array field failure stops naming the field',
        'numbarrow/core/mapinarrow_factory.py',
        '        raise renamed(exc, f"field {name!r}") from exc',
        '        raise',
    ),
    (
        "a record array converted as inferred stops naming a failing field",
        "numbarrow/core/mapinarrow_factory.py",
        "        children = [_record_field(value, name, None) for name in names]",
        "        children = [_convert(value[name], None) for name in names]",
    ),
    (
        "input_columns is read again on every batch",
        "numbarrow/core/mapinarrow_factory.py",
        "            input_columns_ = named if named is not None else list(dict.fromkeys(names))",
        "            input_columns_ = list(dict.fromkeys(input_columns if input_columns is not None else names))",
    ),
    (
        'a time unit multiplier stops being folded in',
        'numbarrow/core/mapinarrow_factory.py',
        '    if target == unit and count == 1:\n        return value',
        '    if target == unit:\n        return value',
    ),
    (
        'a coarse time unit stops being taken to seconds',
        'numbarrow/core/mapinarrow_factory.py',
        '    if unit in ("h", "m") or (family == "timedelta64" and unit in ("W", "D")):\n        target = "s"',
        '    if False:\n        target = "s"',
    ),
    (
        'a str output stops being refused',
        'numbarrow/core/mapinarrow_factory.py',
        '    if isinstance(value, (str, bytes)):\n        raise TypeError(',
        '    if False:\n        raise TypeError(',
    ),
    (
        'a 0-d or 2-d array output stops being refused',
        'numbarrow/core/mapinarrow_factory.py',
        '    if kind in ("U", "S") and value.ndim != 1:',
        '    if False:',
    ),
    (
        'a Nullable inside a Nullable stops being refused',
        'numbarrow/core/mapinarrow_factory.py',
        "    if isinstance(value, Nullable):\n        # A helper's Nullable",
        "    if False:\n        # A helper's Nullable",
    ),
    (
        'tuple rows stop being checked by position',
        'numbarrow/core/mapinarrow_factory.py',
        '                children.extend(row[index] for row in tuples if index < len(row))',
        '                pass',
    ),
    (
        'a namedtuple naming the fields in another order stops being refused',
        'numbarrow/core/mapinarrow_factory.py',
        '        if given is not None and set(given) == set(names) and list(given) != names:',
        '        if False:',
    ),
    (
        'iterability stops being tested with iter',
        'numbarrow/core/mapinarrow_factory.py',
        '        try:\n            iter(row)\n        except TypeError:\n            continue\n        kept.append(row)',
        '        if not hasattr(row, "__iter__"):\n            continue\n        kept.append(row)',
    ),
    (
        'a key/value entry dict stops being read as a pair',
        'numbarrow/core/mapinarrow_factory.py',
        '                if isinstance(pair, Mapping) and set(pair) == {"key", "value"}:',
        '                if False:',
    ),
    (
        'pyarrow scalar rows stop being passed over by the key check',
        'numbarrow/core/mapinarrow_factory.py',
        '        if isinstance(row, (str, bytes, pa.Scalar)) or',
        '        if isinstance(row, (str, bytes)) or',
    ),
    (
        'a pandas Series row stops being refused',
        'numbarrow/core/mapinarrow_factory.py',
        '        if _is_pandas(row, "Series", "DataFrame"):',
        '        if False:',
    ),
    (
        'a KeyError from pa.array stops naming the column',
        'numbarrow/core/mapinarrow_factory.py',
        '    except (pa.ArrowException, TypeError, ValueError, OverflowError, KeyError) as exc:\n'
        '        raise renamed(exc, f"output column {name!r}") from exc',
        '    except (pa.ArrowException, TypeError, ValueError, OverflowError) as exc:\n'
        '        raise renamed(exc, f"output column {name!r}") from exc',
    ),
    (
        'a KeyError from a record field stops naming the field',
        'numbarrow/core/mapinarrow_factory.py',
        '    except (pa.ArrowException, TypeError, ValueError, OverflowError, KeyError) as exc:\n'
        '        raise renamed(exc, f"field {name!r}") from exc',
        '    except (pa.ArrowException, TypeError, ValueError, OverflowError) as exc:\n'
        '        raise renamed(exc, f"field {name!r}") from exc',
    ),
    (
        'a chunked array from pa.array stops being combined',
        'numbarrow/core/mapinarrow_factory.py',
        '    if isinstance(array, pa.ChunkedArray):\n        # A pandas Series over a multi-chunk',
        '    if False:\n        # A pandas Series over a multi-chunk',
    ),
    (
        'the field guard stops seeing through an extension type',
        'numbarrow/core/mapinarrow_factory.py',
        '    source_type = _storage(source_type)\n'
        '    declared_type = _storage(declared_type)',
        '    declared_type = _storage(declared_type)',
    ),
    (
        'a map source stops being paired with a declared list of entries',
        'numbarrow/core/mapinarrow_factory.py',
        '    if pa.types.is_map(source_type) and _is_list_like(declared_type):',
        '    if False:',
    ),
    (
        'view layouts stop counting as list-like',
        'numbarrow/core/mapinarrow_factory.py',
        '            or pa.types.is_fixed_size_list(arrow_type) or _is_list_view(arrow_type))',
        '            or pa.types.is_fixed_size_list(arrow_type))',
    ),
    (
        "a later batch's inferred schema stops being compared with the first's",
        'numbarrow/core/mapinarrow_factory.py',
        '    if built.schema != inferred:',
        '    if False:',
    ),
    (
        'an extension column stops being masked through its storage',
        'numbarrow/core/mapinarrow_factory.py',
        '    if isinstance(array, pa.ExtensionArray):\n'
        '        # The flat test below',
        '    if False:\n'
        '        # The flat test below',
    ),
    (
        'a union child stops being refused before flatten',
        'numbarrow/utils/arrow_array_utils.py',
        '        if _is_union_layout(raw_child.type):',
        '        if False:',
    ),
    (
        "a view's base stops being a buffer that cannot be released",
        'numbarrow/utils/arrow_array_utils.py',
        '        pa.py_buffer(memoryview(data_buf).toreadonly()),',
        '        memoryview(data_buf).toreadonly(),',
    ),
    (
        "a batch's arrays stay bound across the yield",
        'numbarrow/core/mapinarrow_factory.py',
        '            data_dict = bitmap_dict = handed = col_pa = adapted = None',
        '            pass',
    ),
    (
        'structured dtypes stop getting viewers of their own',
        'numbarrow/utils/utils.py',
        '    if dtype_.fields is not None:',
        '    if False:',
    ),
    (
        'a zero-length temporal column stops keeping its bitmap presence',
        'numbarrow/core/adapters.py',
        '    if not len(pa_array):\n'
        '        # The cast of a zero-length array drops its validity buffer, and a',
        '    if False:\n'
        '        # The cast of a zero-length array drops its validity buffer, and a',
    ),
    (
        'the unexpected-keys listing stops being cut',
        'numbarrow/core/mapinarrow_factory.py',
        '            shown = unexpected_keys[:KEYS_SHOWN]',
        '            shown = unexpected_keys',
    ),
    (
        'a scalar stops being described as one at the dispatcher',
        'numbarrow/core/adapters.py',
        '    if isinstance(pa_array, pa.Scalar):',
        '    if False:',
    ),
    (
        'a type attribute that is not a DataType stops being screened',
        'numbarrow/core/adapters.py',
        '    if not isinstance(arrow_type, pa.DataType):',
        '    if arrow_type is None:',
    ),
    (
        'a repeated name in output_schema stops being refused',
        'numbarrow/core/mapinarrow_factory.py',
        '    repeated = _repeated_names(list(output_schema))\n'
        '    if repeated:',
        '    repeated = _repeated_names(list(output_schema))\n'
        '    if False:',
    ),
    (
        'an item that is not a RecordBatch stops being refused by name',
        'numbarrow/core/mapinarrow_factory.py',
        '            if not isinstance(batch, pa.RecordBatch):',
        '            if False:',
    ),
    (
        'the options refusal stops showing the value',
        'numbarrow/core/configurations.py',
        '            f"{invalid_jit_options_err}; {as_str!r} is valid JSON but a {type(as_json).__name__}, not an object"',
        '            invalid_jit_options_err',
    ),
    (
        'the 64-bit date view stops refusing another unit',
        'numbarrow/core/adapters.py',
        '    if np_dtype != np.dtype(f"datetime64[{unit}]"):',
        '    if False:',
    ),
    (
        'a fixed-width binary column stops going to pa.array directly',
        'numbarrow/core/mapinarrow_factory.py',
        '        if arrow_type is not None and pa.types.is_fixed_size_binary(arrow_type):',
        '        if False:',
    ),
    (
        'a nested key refusal stops naming the field path',
        'numbarrow/core/mapinarrow_factory.py',
        '                _check_keys(children, child_type, f"{where}field {name!r}: ")',
        '                _check_keys(children, child_type, where)',
    ),
    (
        'a function that numba cannot cache stops compiling uncached',
        'numbarrow/core/configurations.py',
        '            if "no locator available" not in str(error) or not jit_options.get("cache"):\n'
        '                raise',
        '            raise',
    ),
    (
        'a bytes output stops naming its type',
        'numbarrow/core/mapinarrow_factory.py',
        '        return pa.array(value.tolist(), type=arrow_type or pa.binary())',
        '        return pa.array(value.tolist(), type=arrow_type)',
    ),
    (
        "the dispatcher stops cutting a chunked array's type",
        'numbarrow/core/adapters.py',
        '            f"Not implemented for a ChunkedArray of {pa_array.num_chunks} chunks of type "\n'
        '            f"{type_repr(pa_array.type)}: pass one chunk, or combine_chunks() first"',
        '            f"Not implemented for a ChunkedArray of {pa_array.num_chunks} chunks of type "\n'
        '            f"{pa_array.type}: pass one chunk, or combine_chunks() first"',
    ),
    (
        'a timestamp stops being read at its own unit',
        'numbarrow/core/adapters.py',
        '    return cast_64bit_date_arrow_to_numpy_array(pa_array, np.dtype(f"datetime64[{timestamp_unit}]"))',
        '    return cast_64bit_date_arrow_to_numpy_array(pa_array, np.dtype("datetime64[us]"))',
    ),
]

COPY = ["numbarrow", "test", "README.md", "pyproject.toml", "docs"]


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
