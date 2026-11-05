# numbarrow

numbarrow lets numba `@njit` functions read [Apache Arrow][arrow] data, such as the record batches that Spark sends to
Python in `mapInArrow`. numba cannot compile a function that takes a `pyarrow.Array`. numbarrow converts each array
into numpy arrays that numba accepts: the values, and a validity bitmap that marks the null rows. The values are a view
of the Arrow memory where the Arrow layout matches a numpy dtype, and a copy where it does not.

For [PySpark][pyspark], `make_mapinarrow_func` turns a function of yours into one that [`DataFrame.mapInArrow`][mia]
can run. Your function receives the columns of each batch as numpy arrays, in dicts keyed by column name, and returns a
dict of output columns. numbarrow converts each batch, calls your function, and turns the columns it returns into the
batch that goes back to Spark.

[arrow]: https://arrow.apache.org/docs/python/
[pyspark]: https://spark.apache.org/docs/3.5.7/api/python/index.html
[mia]: https://spark.apache.org/docs/3.5.7/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.mapInArrow.html

## Installation

```bash
pip install numbarrow
```

This installs numba, numpy and pyarrow, which is all that the conversions need.

PySpark's `mapInArrow` also needs pandas and setuptools. pyspark checks that pandas is installed, and it imports
`distutils`, a module that Python 3.12 removed from the standard library ([PEP 632][pep632]). setuptools still provides
`distutils`. The `mapinarrow` extra installs both:

```bash
pip install "numbarrow[mapinarrow]"
```

The extra does not install pyspark. Install the pyspark version that matches your Spark, within the range that
[Compatibility](#compatibility) gives.

The `test` extra installs pyspark and the other packages that numbarrow's own test suite needs. The tests are not part
of the package, so run them from a clone of the repository, installed with `pip install -e ".[test]"`.

[pep632]: https://peps.python.org/pep-0632/

## Quick Start

`arrow_array_adapter` converts one Arrow array, and `is_null` reads its bitmap inside jitted code:

```python
import pyarrow as pa
from numba import njit

from numbarrow.core.adapters import arrow_array_adapter
from numbarrow.core.is_null import is_null


@njit
def sum_valid(bitmap, data):
    total = 0
    for i in range(len(data)):
        if bitmap is None or not is_null(i, bitmap):
            total += data[i]
    return total


bitmap, data = arrow_array_adapter(pa.array([10, None, 30, 40], type=pa.int32()))
print(sum_valid(bitmap, data))  # 80
```

`arrow_array_adapter` returns a pair, `(bitmap, data)`, for every type except a struct or a list of structs. In this
example, `data` is a read-only `int32` array that views the Arrow buffer and holds `[10, 0, 30, 40]`. The null row
still holds a value, here 0, so the function checks the bitmap before it adds. `bitmap` is a `uint8` array with one bit
per row, and `is_null(i, bitmap)` returns `True` when row `i` is null.

`bitmap` is `None` when the Arrow array has no validity buffer, the Arrow memory that marks its nulls. That is common
for an array without nulls, but not guaranteed: [Nulls and validity bitmaps](#nulls-and-validity-bitmaps) explains when.
`is_null` does not accept `None`, so `sum_valid` checks for `None` first. numba compiles a second version of
`sum_valid` for a call with `None`.

## Supported Types

`arrow_array_adapter` accepts the array classes in the table. The Copy? column says whether the data views the Arrow
memory or is a copy. numpy can view a fixed-width layout such as `int64` directly, but numbarrow must convert booleans,
`date32` values and strings.

| pyarrow array | numpy result | Copy? |
|---|---|---|
| `Int32Array`, `Int64Array`, `DoubleArray` | `int32`, `int64` or `float64` | No (view) |
| `BooleanArray` | `bool_` | Yes (bits unpacked to bytes) |
| `Date32Array` | `datetime64[D]` | Yes (32-bit days widened to 64 bits) |
| `Date64Array` | `datetime64[ms]` | No (view) |
| `TimestampArray` | `datetime64` of the same unit, without the time zone | No (view) |
| `UInt8Array` | `uint8` | No (view) |
| `StringArray`, `LargeStringArray` | Fixed-width Unicode, as wide as the longest non-null value | Yes (repacked) |
| `StructArray` | 3-tuple: struct bitmap, then dicts of field bitmaps and field data | Per-field |
| `ListArray` of structs, or a map | The same 3-tuple, over the elements of all the lists | Per-field |

A map column, a `MapArray` in pyarrow, is a `ListArray` whose elements are structs with the two fields `key` and
`value`, and it converts as one. The section [Structs, lists of structs and maps](#structs-lists-of-structs-and-maps)
shows the result.

Any other type raises `NotImplementedError` with the type in the message: for example `int8`, `int16`, `float32`,
`binary`, a dictionary-encoded array, or a list whose elements are not structs. Cast such a column to a supported type
before numbarrow converts it. In Spark, a `ByteType` or `ShortType` column arrives as `int8` or `int16`, and a
`FloatType` column as `float32`. Cast them in the DataFrame that feeds `mapInArrow`, as in `col("s").cast("int")` or
`col("f").cast("double")`. Outside Spark, `array.cast(pa.int32())` converts a pyarrow array.

A struct with two fields of the same name raises `NotImplementedError` too, because numbarrow keys the fields by name.
So does a [`ChunkedArray`][chunked], such as a column of a `pyarrow.Table`. Convert each chunk, such as
`column.chunk(0)`, or call [`combine_chunks()`][combine] first.

### Nulls and validity bitmaps

Arrow marks the null rows of an array in a validity buffer, and the bitmap that `arrow_array_adapter` returns is a copy
of that buffer. It follows Arrow's [validity bitmap][validity] layout: a `uint8` array with one bit per row, least
significant bit first, and a set bit for a valid row. It has `(rows + 7) // 8` bytes. Because it is a copy, it stays
valid after the Arrow array is freed, and your code may change it, for example to mark more rows null in an output
column ([Returning nulls](#returning-nulls)).

Code that reads a bitmap must accept both `None` and an array, as `sum_valid` in the Quick Start does. The bitmap is
`None` when the array has no validity buffer. An array that holds nulls always has one, but an array without nulls may
or may not:

- pyarrow leaves the buffer out when it builds an array without nulls, as in `pa.array([1, 2])`.
- pyarrow's IPC reader drops the buffer of each column that holds no nulls in the batch it reads. Spark sends its
  batches to Python through that reader, so one batch of a column can come with a bitmap and the next with `None`.
- A slice of an array that has a buffer keeps it, even when the slice holds no nulls.

A null row's value carries no meaning. In a batch from Spark, a null row holds `0`, `0.0` or an empty string.

### Strings

A `StringArray` or `LargeStringArray` becomes a numpy array of fixed-width Unicode strings, such as `<U5`. The width is
the length of the longest non-null value, counted in characters, and at least one. A null row holds an empty string.

A value that ends with a NUL character (`"\x00"`) raises `ValueError`. numpy pads every shorter value with NUL
characters, so it cannot tell a trailing NUL from the padding. A NUL at the start or in the middle of a value is kept.

Because the width comes from the longest value of each array, two batches of one column usually convert to different
widths. numba gives a `<U3` array and a `<U5` array different types, so a jitted function without a signature compiles
one more version each time a new width appears. Spark cuts each partition into batches of at most
[`spark.sql.execution.arrow.maxRecordsPerBatch`][batchsize] rows, 10,000 by default, so one job can meet many widths. A
function compiled for an explicit signature accepts one width only, as [String widths](#string-widths) shows.

### Memory for long strings

The copy takes four bytes per character of the width in every row, however short the other values are. One
100,000-character value in a 4,000-row batch of four-character strings makes a 1.6 GB array from about 130 KB of
Arrow data. [`make_mapinarrow_func`](#pyspark-mapinarrow) converts every column of a batch unless its
`input_columns` parameter names the ones to convert. If your function does not need such a column, leave it out of
`input_columns`, so that numbarrow does not convert it. If you also leave the column out of the DataFrame that feeds
`mapInArrow`, Spark does not send it to Python at all.

### Timestamps and time zones

A `TimestampArray` becomes `datetime64` of the same unit, and numbarrow drops the time zone, as pyarrow's own
`to_numpy` does. The values are instants in UTC. A `timestamp[us, tz=America/New_York]` array and a `timestamp[us]`
array with the same stored values convert to the same result.

Date arithmetic in your function therefore works in UTC. Spark's [`to_date`][to-date] works in the session time zone,
`spark.sql.session.timeZone`, so the two can give different dates. For example, with the session time zone set to
`America/Los_Angeles`, Spark's `to_date` of the timestamp 2020-01-01 20:00 is 2020-01-01. The same instant is
2020-01-02 04:00 in UTC, so `datetime64[D]` gives 2020-01-02 in your function.

[Output column types](#output-column-types) describes what a `datetime64` result becomes on its way back to Spark.

### Structs, lists of structs and maps

A `StructArray` becomes a 3-tuple, `(struct_bitmap, field_bitmaps, field_data)`. `struct_bitmap` marks the rows that
are null as a whole, or is `None`. `field_bitmaps` and `field_data` are dicts keyed by field name: `field_bitmaps["x"]`
is the bitmap of the field `x`, and `field_data["x"]` is its data, as for an array of the field's type. The table shows
whether each field is a view or a copy.

A row that is null as a whole is not always null in its fields: pyarrow, for example, fills the fields of such a row
with valid zeros and empty strings. `is_null_struct(i, struct_bitmap, field_bitmap)` therefore checks both bitmaps. It
returns `True` when either one marks row `i` null, and it accepts `None` for either:

```python
import pyarrow as pa
from numba import njit

from numbarrow.core.adapters import arrow_array_adapter
from numbarrow.core.is_null import is_null_struct


@njit
def count_valid(struct_bitmap, field_bitmap, field):
    count = 0
    for i in range(len(field)):
        if not is_null_struct(i, struct_bitmap, field_bitmap):
            count += 1
    return count


points = pa.array([{"x": 1, "y": 2.0}, None, {"x": None, "y": 4.0}])
struct_bitmap, field_bitmaps, field_data = arrow_array_adapter(points)
print(count_valid(struct_bitmap, field_bitmaps["x"], field_data["x"]))  # 1
```

Row 1 is null as a whole, and row 2 has a null `x`. `field_bitmaps["x"]` marks only row 2, so a check of that bitmap
alone would count row 1 as valid.

`make_mapinarrow_func` combines the two bitmaps for you. The `bitmap_dict` that it passes to your function holds no
bitmap for the struct itself, and the bitmap of each field also marks the rows where the whole struct is null. One
`is_null` call per field is therefore enough there, and `is_null_struct` is not needed.

A `ListArray` whose elements are structs becomes the same 3-tuple, over the elements of all its lists in order. Its
`struct_bitmap` marks null elements, not null rows. The result has no offsets, so it cannot show where one row's
elements end. numbarrow therefore requires every row to hold the same number of elements, `n`, and element `j` belongs
to row `j // n`. To find `n`, divide the length of a field's data by the number of rows. That number is
`len(list_array)` when you call the adapter, and the length of another column, such as `len(data_dict["id"])`, in
`make_mapinarrow_func`. A list column with rows of different lengths, or with a null row, raises
`NotImplementedError`.

A map converts as a list of structs with the two fields `key` and `value`. `field_data["key"]` holds the keys of all
the rows in order, and `field_data["value"]` holds the values. Every row must hold the same number of entries, so a map
with rows of different lengths raises `NotImplementedError`. So does a map whose keys or values are of a type the table
does not list, such as a struct or a list.

### Read-only arrays

Every data array is read-only. A view shares memory with the Arrow array, whose buffers are immutable, and numpy refuses
to make it writable: `ValueError: cannot set WRITEABLE flag to True of this array`. A view from pyarrow's own
`to_numpy(zero_copy_only=True)` refuses in the same way. A copy, from a boolean, `date32` or string array, is read-only
too, so that the result does not depend on the type. You may set a copy's `writeable` flag, because its memory is yours.

An `@njit` function that returns a slice or a reshape of a view returns a new array, also read-only. numba makes an
object of its own the base of that array, so numpy cannot see that the memory belongs to a read-only Arrow buffer, and
it lets you set the array's `writeable` flag. A write through the array then changes the Arrow array. Call `.copy()` on
such an array before you change it.

### Explicit signatures

An `@njit` function without a signature needs no declaration for numbarrow's arrays, because numba compiles it for the
argument types of each call. A function compiled for an [explicit signature][eager] accepts only the types that the
signature names. For any others, it raises `TypeError: No matching definition for argument type(s)`, followed by
the types it was given. Three properties of numbarrow's arrays need care in a signature. The examples below use
`Array`, `Optional`, `int64` and `uint8` from `numba.core.types`.

#### Read-only data

Declare each data parameter with `readonly=True`, as in `Array(int64, 1, "C", readonly=True)`. Such a parameter accepts
a writable array as well. A parameter declared without it refuses numbarrow's data arrays:

```text
TypeError: No matching definition for argument type(s) readonly array(int64, 1d, C)
```

#### Bitmaps that are None

A bitmap can be an array in one batch and `None` in the next. A parameter declared as an array refuses `None`. The
error names the types of the arguments, here an `int64` column and its missing bitmap:

```text
TypeError: No matching definition for argument type(s) readonly array(int64, 1d, C), none
```

Declare the bitmap parameter as `Optional(Array(uint8, 1, "C", readonly=True))`, as
[test_mapinarrow_spark.py][spark-test] does.

#### String widths

The width of a string array is part of its numba type, so a signature that names one width refuses every other width,
wider or narrower:

```text
TypeError: No matching definition for argument type(s) readonly array([unichr x 5], 1d, C)
```

Leave a function that takes strings without a signature.

[batchsize]: https://spark.apache.org/docs/3.5.7/api/python/user_guide/sql/arrow_pandas.html#setting-arrow-batch-size
[chunked]: https://arrow.apache.org/docs/python/generated/pyarrow.ChunkedArray.html
[combine]: https://arrow.apache.org/docs/python/generated/pyarrow.ChunkedArray.html#pyarrow.ChunkedArray.combine_chunks
[eager]: https://numba.readthedocs.io/en/stable/user/jit.html#eager-compilation
[spark-test]: https://github.com/Goykhman/numbarrow/blob/main/test/test_mapinarrow_spark.py
[to-date]: https://spark.apache.org/docs/3.5.7/api/python/reference/pyspark.sql/api/pyspark.sql.functions.to_date.html
[validity]: https://arrow.apache.org/docs/format/Columnar.html#validity-bitmaps

## PySpark mapInArrow

[`DataFrame.mapInArrow`][mia] runs a Python function on an iterator of Arrow record batches and expects an iterator of
batches back. `make_mapinarrow_func` builds that function from a simpler one, `main_func` below, which receives the
columns of each batch as numpy arrays and returns a dict of output columns. `main_func` itself is plain Python: it
passes the arrays to jitted functions, such as `scale` here.

```python
import os
import sys

from numba import njit
from pyspark.sql import SparkSession

from numbarrow.core.mapinarrow_factory import Nullable, make_mapinarrow_func


@njit
def scale(values, factor):
    return values * factor


def main_func(data_dict, bitmap_dict, broadcasts):
    scaled = scale(data_dict["value"], broadcasts["factor"])
    # scaled is null where value is null, so it takes value's bitmap.
    return {"id": data_dict["id"], "scaled": Nullable(scaled, bitmap_dict["value"])}


os.environ["PYSPARK_PYTHON"] = sys.executable  # Spark's Python workers must import numbarrow too.
spark = SparkSession.builder.master("local[1]").getOrCreate()
df_in = spark.createDataFrame([(1, 1.5), (2, None), (3, 4.0)], "id long, value double")
udf = make_mapinarrow_func(main_func, broadcasts={"factor": 2.0})
df_in.mapInArrow(udf, "id long, scaled double").show()
```

With pyspark 3.5, the job prints the following table; pyspark 3.4 prints `null` in place of `NULL`.

```text
+---+------+
| id|scaled|
+---+------+
|  1|   3.0|
|  2|  NULL|
|  3|   8.0|
+---+------+
```

For each batch, numbarrow converts every column with `arrow_array_adapter`, or only the columns that the
`input_columns` parameter names. It then calls `main_func` with three dicts:

- `data_dict` maps each column's name to its data. A struct column, or a list of structs, maps to a dict of its
  fields, as in `data_dict["point"]["x"]`.
- `bitmap_dict` has the same keys and holds the bitmaps, or `None` where a column has no validity buffer in this batch.
  Every key is present in every batch, with or without nulls. For a struct column, the bitmap of each field also marks
  the rows where the whole struct is null, so one `is_null` call per field is enough. `arrow_array_adapter` keeps those
  two bitmaps apart, as [Structs, lists of structs and maps](#structs-lists-of-structs-and-maps) shows.
- `broadcasts` is the dict given to `make_mapinarrow_func`, unchanged.

In the example, `Nullable(scaled, bitmap_dict["value"])` pairs the output column with the bitmap of the input column
`value`, so that `scaled` is null where `value` is null. [Returning nulls](#returning-nulls) explains `Nullable`.

Pass `input_columns`, a list of names, to convert only those columns. Each name must match a column of the batch
exactly, case included. A missing name raises `KeyError`, and the message lists the batch's columns.

`main_func` returns a dict that maps each output column's name to its values: a numpy array, a list or a
[`pyarrow.Array`][pa-array]. A list of dicts, or a numpy record array, becomes a struct column. numbarrow converts most
values with [`pyarrow.array`][pa-array-fn]. It converts a record array one field at a time, and it uses a
`pyarrow.Array` as it is, or casts it to the type that `output_schema` declares.

Spark runs `main_func` in its Python workers, so numba and numbarrow must be installed wherever those workers run.
Spark starts the workers with the `python3` on the `PATH` unless `PYSPARK_PYTHON` names another interpreter. Without
numba there, the job fails with `ModuleNotFoundError: No module named 'numba'`, which is why the example sets
`PYSPARK_PYTHON`. [test_mapinarrow_spark.py][spark-test] has a longer example with struct and list columns.

### Returning nulls

A plain numpy array becomes an output column without nulls, except for a `NaT`, which pyarrow reads as null. A row
that came in null then goes out as a valid row, holding whatever `main_func` calculated from the null row's value:
without `Nullable`, the example would print `0.0` for id 2. To mark rows null, return `Nullable(data, bitmap)`, with a
bitmap in the layout of the bitmaps in `bitmap_dict`.

If the result is null exactly where one input column is null, pass that column's bitmap, as the example does with
`bitmap_dict["value"]`. For a struct field, pass `bitmap_dict["column"]["field"]`. For any other pattern, build the
bitmap from a boolean array `valid` of the valid rows: `np.packbits(valid, bitorder="little")`. `Nullable(data, None)`
adds no nulls: the column then carries only the nulls that `data` holds itself, such as a `None` in a list.

The bitmap must be a one-dimensional `uint8` array of `(rows + 7) // 8` bytes, or numbarrow raises an error that names
the column. A bitmap from `bitmap_dict` fits only a column of the length it came with: the batch's row count for a
column's bitmap, and the field's element count for a struct field's bitmap. A function that changes a column's length
must build a bitmap of its own. numbarrow raises `ValueError` if the column comes with the old bitmap.

Other values can hold nulls too: a list, a tuple or an object array that holds `None`, a `pyarrow.Array` with nulls,
and a [numpy masked array][masked]. A `NaN` in a float array is a value, not a null.

numbarrow refuses a value that `pyarrow.array` would misread, with a `TypeError` that names the column. Such a value is
a dict, a `str` or `bytes`, a pandas DataFrame, or a `Nullable` inside another `Nullable`. Columns of different lengths
raise `ValueError`.

### Columns bind by position, not by name

Spark binds the columns of each returned batch to the schema given to `mapInArrow` by position, not by name. If
`main_func` builds its dict in another order, two columns of the same type, such as two `long` columns, swap without an
error. Struct fields bind by position too, in the order of a dict's keys or of a record array's fields. A dict
`{"lat": 1.0, "lon": 2.0}` under `struct<lon: double, lat: double>` therefore swaps the two values.

Pass `output_schema` to bind columns, and the fields of dicts and record arrays, by name. It is a
[`pyarrow.Schema`][pa-schema], not the Spark `StructType` that `mapInArrow` takes, and numbarrow raises `TypeError` for
a `StructType`. Derive the Arrow schema from the Spark one, so that the two cannot disagree:

```python
import pyarrow as pa
from pyspark.sql.pandas.types import to_arrow_schema
from pyspark.sql.types import DoubleType, LongType, StructField, StructType

from numbarrow.core.mapinarrow_factory import make_mapinarrow_func


def main_func(data_dict, bitmap_dict, broadcasts):
    return {"scaled": data_dict["value"] * 2.0, "id": data_dict["id"]}


spark_schema = StructType([StructField("id", LongType()), StructField("scaled", DoubleType())])
udf = make_mapinarrow_func(main_func, output_schema=to_arrow_schema(spark_schema))

batch = pa.RecordBatch.from_pydict({"id": [1, 2], "value": [1.5, 4.0]})
print(next(udf(iter([batch]))).to_pydict())  # {'id': [1, 2], 'scaled': [3.0, 8.0]}
```

In Spark, pass the same `spark_schema` to `mapInArrow`: `df_in.mapInArrow(udf, spark_schema)`. As the last line
shows, the function that `make_mapinarrow_func` returns takes any iterator of record batches, so a test can call it
without Spark.

### Output column types

With `output_schema`, numbarrow builds each column with its declared type. Without it, pyarrow infers each column's
type from the values, and every later batch of a partition must infer the same types as the first. Spark refuses a
later batch with another schema, and its error names no column: `pyarrow.lib.ArrowInvalid: Tried to write record batch
with different schema`. numbarrow checks each batch first and raises a `ValueError` that names the column whose type
changed. A column returned as an empty list, or as a list that holds only `None`, has the type `null`. Declare
`output_schema` if any batch can return such a column.

Without `output_schema`, a `datetime64` output becomes a timestamp of the same unit, without a time zone. A unit with a
multiplier, such as `datetime64[5s]`, is converted to its base unit, and an hour or minute unit to seconds. A day,
week, month or year unit becomes `date32`. A unit finer than a nanosecond raises `TypeError`. As a result, a zoned
timestamp passed through comes back without its zone, and a `date64` column comes back as `timestamp[ms]`. A numpy
string array becomes `string`, even when the input column was `large_string`.

Spark reads a timestamp column back only in microseconds. A column of another unit fails in the JVM: for
`datetime64[ns]`, the error is `[UNSUPPORTED_ARROWTYPE] Unsupported arrow type Timestamp(NANOSECOND, null)`. Return
`datetime64[us]`, or declare the column in `output_schema`, which converts the unit. A declared type also keeps the
zone of a timestamp and the type of a `date64` column.

A column whose Arrow type differs from its declared Spark type can fail in the JVM, or it can read without an error.
Spark reads the values with the getter of the declared type, such as `getLong` for `LongType` and `TimestampType`.
Spark's reader for each Arrow type provides one getter: `getLong` for `int64` and timestamps, and `getInt` for `int32`
and `date32`. A column whose Arrow type provides the getter of its declared type is read without a check. An `int64`
column declared as a timestamp therefore reads as microseconds since the epoch, and an `int32` column declared as a
date reads as days. A column whose Arrow type provides another getter fails with
`java.lang.UnsupportedOperationException`, even when the widths agree. `float64` under `LongType`, `int64` under
`DoubleType` and `int32` under `LongType` all fail this way.

[masked]: https://numpy.org/doc/stable/reference/maskedarray.html
[pa-array]: https://arrow.apache.org/docs/python/generated/pyarrow.Array.html
[pa-array-fn]: https://arrow.apache.org/docs/python/generated/pyarrow.array.html
[pa-schema]: https://arrow.apache.org/docs/python/generated/pyarrow.Schema.html

## Compilation and the cache

numbarrow compiles `is_null` and its other bitmap functions when your program first imports one of its modules, and
the functions that read Arrow buffers when a conversion first needs them. numba's own default is not to cache, but
numbarrow asks numba to cache the compiled code on disk, so that a later process loads the code instead of compiling
it again. numba writes the cache beside numbarrow's modules if it can, and otherwise to the user cache directory.
[`NUMBA_CACHE_DIR`][cache-dir] overrides both, and [numba's documentation][numba-cache] describes the locations.

`NUMBARROW_JIT_OPTIONS` sets the options that numbarrow passes to `@njit` for its own functions, as a JSON object. If
it is unset or empty, the options are `{"cache": true}`. The object replaces that default rather than adding to it, so
`'{}'` turns caching off. For example, this setting turns on bounds checking, so that `is_null` raises `IndexError`
for an index past the end of its bitmap:

```bash
export NUMBARROW_JIT_OPTIONS='{"boundscheck": true}'
```

The setting also turns caching off, because it leaves out `"cache": true`, and numbarrow then compiles its functions
in every process. To keep the cache under other options, include `"cache": true` and point `NUMBA_CACHE_DIR` at a new
directory. numba's cache does not record the options a function was compiled with. In the old directory, numba would
load the code it compiled under the old options.

numbarrow raises `ValueError` at import if the value is not a JSON object. It also refuses a string for any option
except `error_model` and `inline`, because numba reads any non-empty string as true: `{"cache": "false"}` would turn
caching on.

### Set both variables before numba is imported

numbarrow reads `NUMBARROW_JIT_OPTIONS` when your program first imports one of its modules, and numba reads
`NUMBA_CACHE_DIR` when numba is imported. Another library may import numba before numbarrow does. Set both variables
before the process imports either one.

A cache directory set after numba is imported takes effect late, or not at all. It misses at least the first function
numbarrow compiles. If the cache that numba found at import already holds numbarrow's compiled code, the new directory
misses every function, because numba loads the code from there and reads its environment again only when it compiles.

Spark's Python workers are separate processes. A change to `os.environ` in the driver after Spark has started does not
reach them, and on a cluster the driver's environment never does. Set the variables for the workers with Spark's
[`spark.executorEnv.<name>`][executor-env] setting, as in
`.config("spark.executorEnv.NUMBA_CACHE_DIR", "/tmp/numba-cache")` on the session builder.

### If numbarrow warns that it compiles without a cache

If caching is on and numba cannot write a cache for one of numbarrow's functions, numbarrow compiles the function
without one. It then issues a `RuntimeWarning` that starts with `numba cannot cache` and says how to get a cache. Two
cases cause it:

- numbarrow is installed in a read-only location, and the user cache directory is not writable either. Set
  `NUMBA_CACHE_DIR` to a writable directory.
- numbarrow is imported from an archive: an `.egg` that `spark-submit --py-files` puts on the workers' path, or a
  `.whl` or `.pyz` file placed on `sys.path`. `NUMBA_CACHE_DIR` has no effect there, because numba uses it only for a
  module whose source is a file on disk. Install numbarrow unpacked, or ship it in a file whose name ends in `.zip`.
  numba 0.61 and later cache a module imported from such a file in the user cache directory, if that directory is
  writable, and numba 0.60 does not. numba recognizes the `.zip` name only, although `.whl`, `.pyz` and `.egg` files
  are zip archives too.

In both cases, `NUMBARROW_JIT_OPTIONS='{"cache": false}'` turns caching off and silences the warning.

### NUMBA_DISABLE_JIT is not supported

numbarrow reads Arrow buffers through a numba intrinsic, which works only in compiled code. Under
[`NUMBA_DISABLE_JIT=1`][disable-jit], a boolean or string column, and any column with a validity buffer, therefore
raise `NotImplementedError: <intrinsic _ptr_as_int_to_voidptr> is not usable in pure-python`. A numeric or date
column without a validity buffer still converts, and `is_null` runs as plain Python.

[cache-dir]: https://numba.readthedocs.io/en/stable/reference/envvars.html#envvar-NUMBA_CACHE_DIR
[disable-jit]: https://numba.readthedocs.io/en/stable/reference/envvars.html#envvar-NUMBA_DISABLE_JIT
[executor-env]: https://spark.apache.org/docs/3.5.7/configuration.html#runtime-environment
[numba-cache]: https://numba.readthedocs.io/en/stable/user/jit.html#cache

## Compatibility

| Dependency | Versions |
|---|---|
| Python | 3.12+ |
| numba | 0.60.0 to 0.67.x |
| pyarrow | 14.0.0 to 25.0.1 |
| pyspark | 3.4.0 to 3.5.x (optional; 3.5 on Python 3.13, none on 3.14) |
| pandas | 2.2.2+ (optional; pyspark's `mapInArrow` requires it) |

[`pyproject.toml`][pyproject] declares these ranges and is authoritative, with one exception: it declares no pyarrow
floor, and 14.0.0 is the oldest pyarrow that CI tests. CI runs the test suite on Python 3.12, with the newest numba in
the range, pandas 2.3.2 and pyspark 3.5.7, on Linux x86-64, Linux ARM and Windows. A separate job runs it on Linux with
pyarrow 14.0.0 and 25.0.1, without pandas or pyspark. CI tests no other pandas or pyspark version, and no Python above
3.12.

numbarrow 0.2.1 and later require Python 3.12. On Python 3.10 or 3.11, `pip install numbarrow` does not fail: it
installs numbarrow 0.2.0, the newest release that declares support for them. The current test suite passes on both
when numbarrow is installed with `--ignore-requires-python`, but those versions are not supported.

numba 0.60.0 has wheels for Python up to 3.12. Python 3.13 needs numba 0.61 or later, and Python 3.14 needs numba
0.63 or later.

pyarrow 14 fails to import next to numpy 2, with `ImportError: numpy.core.multiarray failed to import`, and its
requirements do not exclude numpy 2. Install pyarrow 14 with `numpy<2`. pyarrow 15 requires `numpy<2` itself.

pyspark 3.4 does not import on Python 3.13, so Python 3.13 needs pyspark 3.5. pyspark 3.4 also refuses a struct output
column with `NotImplementedError: Invalid return type in mapInPandas`, so a struct output column needs pyspark 3.5 on
any Python. On Python 3.14, pyspark 3.5 cannot serialize a Python function, so no pyspark in the range runs
`mapInArrow` there. The job fails with `PicklingError: Could not serialize object: RecursionError: Stack overflow`.
numbarrow's conversions do not need pyspark.

pyspark 3.3 cannot run a Python UDF on Python 3.12, so 3.4.0 is the floor. Under pyspark 3.3, the driver fails with
`PicklingError: Could not serialize object: IndexError: tuple index out of range`, or a worker fails with
`TypeError: code() argument 13 must be str, not int`.

The pandas floor, 2.2.2, is the first pandas release that works with numpy 2. pandas 2.1.1, for example, installs next
to numpy 2 but fails to import with `ValueError: numpy.dtype size changed, may indicate binary incompatibility`.

[pyproject]: https://github.com/Goykhman/numbarrow/blob/main/pyproject.toml

## Documentation

The [API reference][docs] gives an overview of each module, followed by the module's reference.

[docs]: https://goykhman.github.io/numbarrow/

## License

MIT, with a citation clause: see [LICENSE][license].

[license]: https://github.com/Goykhman/numbarrow/blob/main/LICENSE
