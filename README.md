# numbarrow

Numba adapters for [PyArrow](https://arrow.apache.org/docs/python/) and [PySpark](https://spark.apache.org/docs/latest/api/python/).

numbarrow lets you work with Apache Arrow arrays directly inside Numba `@njit` compiled functions. It converts PyArrow arrays into NumPy views (zero-copy where possible) and extracts validity bitmaps for null handling — bridging PySpark's Arrow-based batch processing with high-performance JIT-compiled code.

## Installation

```bash
pip install numbarrow
```

Optional dependencies for PySpark and pandas support:

```bash
pip install numbarrow[test]       # adds pyspark and everything the tests need
pip install numbarrow[mapinarrow] # adds pandas, which pyspark's mapInArrow requires
```

The adapters themselves need only numba, numpy and pyarrow.

## Quick Start

```python
import pyarrow as pa
from numba import njit
from numbarrow.core.adapters import arrow_array_adapter
from numbarrow.core.is_null import is_null

# Convert a PyArrow array to NumPy for use in @njit
arrow_array = pa.array([10, None, 30, 40], type=pa.int32())
bitmap, data = arrow_array_adapter(arrow_array)

@njit
def sum_non_null(data, bitmap):
    total = 0
    for i in range(len(data)):
        if bitmap is None or not is_null(i, bitmap):
            total += data[i]
    return total

result = sum_non_null(data, bitmap)  # 80
```

## Supported Types

| PyArrow Type | NumPy Result | Copy? |
|---|---|---|
| `Int32Array`, `Int64Array`, `DoubleArray` | Matching dtype | No (view) |
| `BooleanArray` | `bool_` | Yes (bit-unpacking) |
| `Date32Array` | `datetime64[D]` | Yes (int32 → int64) |
| `Date64Array` | `datetime64[ms]` | No (view) |
| `TimestampArray` | `datetime64[unit]` | No (view) |
| `UInt8Array` | `uint8` | No (view) |
| `StringArray`, `LargeStringArray` | Fixed-width Unicode, width in characters | Yes (repacking) |
| `StructArray` | 3-tuple: struct bitmap, bitmaps by field, data by field | Per-field |
| `ListArray` (of structs) | 3-tuple: struct bitmap, bitmaps by field, data by field | Per-field |

Every other array type raises `NotImplementedError` naming the type, including a
`ListArray` whose elements are not structs and a struct with repeated field
names. A `MapArray` is a `ListArray` whose values are a key/value struct, so it
adapts to those two fields, `key` and `value`, rather than raising, but only
when every row holds the same number of entries and both fields are of types
the table names; a map whose values are structs, lists, or any other type the
table does not name raises like any other unsupported struct field. A map
whose rows differ in length, which is the usual shape, raises for the same
reason a ragged list does: the result is the flattened entries with no
offsets, so nothing can say which row an entry belongs to.

A `TimestampArray` adapts by its unit alone: a zoned `timestamp[us, tz=...]`
and a naive `timestamp[us]` holding the same int64 adapt to the same
`datetime64[us]`, exactly as pyarrow's `to_numpy` does, so a UDF's calendar
arithmetic runs on UTC instants and can disagree with Spark's own `to_date` by
the session offset. On the way back out of `make_mapinarrow_func` a
`datetime64` output becomes a naive timestamp of its unit, and a `date64` input
passed through comes back `timestamp[ms]`; pass `output_schema` to restore a
zone or a date type.

A `ListArray` of structs flattens its elements and returns no offsets, so a null
outer row can be neither reported nor accounted for in the element-to-row
mapping. A list column whose `null_count` is non-zero raises
`NotImplementedError` rather than returning a result that silently misaligns.

A string value whose last character is NUL raises `ValueError`: numpy's
fixed-width `|U` dtype pads with NUL, so a trailing NUL is indistinguishable
from padding and cannot be represented. Leading and interior NULs are preserved.

Returned data arrays are read-only and cannot be made writable. The views are
over Arrow buffers the caller does not own, which is also why pyarrow's own
`to_numpy(zero_copy_only=True)` refuses to hand out a writable one; the copies,
booleans, `date32` and strings, are marked read-only as well, so the contract
does not depend on the type. Declare numba signatures that receive them with
`readonly=True`, which accepts writable arrays as well, or leave the function
lazily typed and numba will infer it. Returned bitmaps own their memory and are
writable.

Two exceptions to declaring a signature. A string column adapts to a
fixed-width `|U` dtype whose width is the longest live value **in that batch**,
so the numba type of a string argument varies from batch to batch. Spark splits
a partition at `spark.sql.execution.arrow.maxRecordsPerBatch`, so a signature
that names one width compiles on the first batch and raises `TypeError: No
matching definition` on the next one whose width differs, wider or narrower.
Leave string arguments lazily typed, at the cost of a fresh compilation
whenever a new width appears. The `|U` result is the widest live value times
the row count times four bytes, whatever the other values are: one
100,000-character value in a 4,000-row batch allocates 1.6 GB from 120 KB of
Arrow data, so keep such a column out of the projection that feeds
`mapInArrow`.

A column's bitmap is `None` when the batch carries no validity buffer and a
uint8 array otherwise, which is not the same as having no nulls: Spark's Arrow
transport drops the buffer when a batch has no nulls, while `slice`, `take`,
`filter` and `fill_null` keep an all-valid one. A signature that names a bitmap
array type compiles on the batch that has a null and raises `TypeError: No
matching definition for argument type(s) ..., none` on the next one. Declare
bitmap parameters `Optional(Array(uint8, 1, "C", readonly=True))`, as
[test/test_mapinarrow_spark.py](test/test_mapinarrow_spark.py) does.

A uniform array adapts to a 2-tuple, `(bitmap, data)`, where `bitmap` is `None`
when the array has no validity buffer. A struct or list-of-struct array adapts
to a 3-tuple: the struct-level bitmap, then two dicts keyed by field name. The
struct-level bitmap is the only record of a row that is null as a whole, since
the fields of such a row carry no validity bits of their own; pass both layers
to `is_null_struct`.

## PySpark Integration

Use `make_mapinarrow_func` to create functions compatible with PySpark's `mapInArrow`:

```python
from numbarrow.core.mapinarrow_factory import make_mapinarrow_func

def compute(data_dict, bitmap_dict, broadcasts):
    # data_dict:   {name: np.ndarray} for a uniform column, or
    #              {name: {field: np.ndarray}} for a struct column
    # bitmap_dict: the same shape, each leaf a uint8 bitmap or None where the
    #              column carries no validity buffer; for a struct column the
    #              struct-level validity is folded into each field's bitmap.
    #              For a list of structs the fold covers the flattened
    #              elements, not the outer list rows: a list column holding a
    #              null row is refused
    result = data_dict["value"] * broadcasts["scale"]
    return {"output": result}

udf = make_mapinarrow_func(compute, broadcasts={"scale": 2.0})
df_in = ...           # caller-provided PySpark DataFrame
output_schema = ...   # caller-provided PySpark StructType
df_out = df_in.mapInArrow(udf, output_schema)
```

See [test/test_mapinarrow_spark.py](test/test_mapinarrow_spark.py) for a complete runnable example.

## Compatibility

| Dependency | Versions |
|---|---|
| Python | 3.12+ |
| numba | 0.60.0 – 0.67.0 |
| pyarrow | 14.0 – 24.0 |
| pyspark | 3.4 – 3.x (optional) |
| pandas | 2.2.2+ (optional, required by pyspark's `mapInArrow`) |

`pyproject.toml` is authoritative. CI runs the newest numba the cap admits,
with pandas 2.3.2 and pyspark 3.5.7, on Linux, Linux ARM and Windows, and both
ends of the pyarrow row in a job of their own; the numba floor and the pandas
and pyspark rows are not swept. The pyspark floor is 3.4.0 because pyspark 3.3
bundles cloudpickle 2.0.0, which predates the `co_qualname` argument Python
3.11 added to `code()`, so on the declared Python every UDF dies in the worker
with `TypeError: code() argument 13 must be str, not int`. The pandas row is
2.2.2 rather than the 1.5.0 declared in the `mapinarrow` extra: no pandas
below 2.1.1 publishes a Python 3.12 wheel, and 2.1.1 installs next to numpy 2
but fails to import with `numpy.dtype size changed`; 2.2.2 is the first
release built against numpy 2. The package also builds and passes its suite
on Python 3.10 and 3.11 when installed with `--ignore-requires-python`; treat
that as regression signal rather than a supported configuration, since pip
refuses the install below the declared floor. The pyarrow range is measured
rather than declared, and the
real constraint is numpy rather than pyarrow: 14.0.0 through 24.0.0 all pass,
but pyarrow below 16 is built against numpy 1 and dies with
`numpy.core.multiarray failed to import` if numpy 2 is installed alongside it.
pyarrow 15 caps numpy itself, so it resolves correctly on its own; pyarrow 14
does not, so it needs an explicit `numpy<2`. `pyproject.toml` declares no
pyarrow floor, so the broken combination is reachable.

`NUMBA_DISABLE_JIT=1` is not supported: the viewers are built on a numba
intrinsic that has no pure-Python form, so every adapter raises
`NotImplementedError` under it, while `is_null` and `unpack_booleans` still run
as plain Python.

## Documentation

Full API documentation: [numbarrow docs](https://goykhman.github.io/numbarrow)

## License

See [LICENSE](LICENSE).
