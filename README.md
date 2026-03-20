# numbarrow

Numba adapters for [PyArrow](https://arrow.apache.org/docs/python/) and [PySpark](https://spark.apache.org/docs/latest/api/python/).

numbarrow lets you work with Apache Arrow arrays directly inside Numba `@njit` compiled functions. It converts PyArrow arrays into NumPy views (zero-copy where possible) and extracts validity bitmaps for null handling — bridging PySpark's Arrow-based batch processing with high-performance JIT-compiled code.

## Installation

```bash
pip install numbarrow
```

Optional dependencies for PySpark and pandas support:

```bash
pip install numbarrow[test]       # adds pyspark
pip install numbarrow[mapinarrow] # adds pandas
```

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
| `StringArray` | Fixed-width Unicode | Yes (repacking) |
| `StructArray` | Dict of field arrays | Per-field |
| `ListArray` (of structs) | Dict of field arrays | Per-field |

## PySpark Integration

Use `make_mapinarrow_func` to create functions compatible with PySpark's `mapInArrow`:

```python
from numbarrow.core.mapinarrow_factory import make_mapinarrow_func

def compute(data_dict, bitmap_dict, broadcasts):
    # data_dict: {col_name: np.ndarray}
    # bitmap_dict: {col_name: uint8 bitmap array}
    result = data_dict["value"] * broadcasts["scale"]
    return {"output": result}

udf = make_mapinarrow_func(compute, broadcasts={"scale": 2.0})
df_out = df_in.mapInArrow(udf, output_schema)
```

See [test/demo_map_in_arrow.py](test/demo_map_in_arrow.py) for a complete runnable example.

## Compatibility

| Dependency | Versions |
|---|---|
| Python | 3.10 – 3.12 |
| numba | 0.60 – 0.63 |
| pyarrow | 14 – 18 |
| pyspark | 3.3 – 3.x (optional) |
| pandas | 1.5+ (optional) |

## Documentation

Full API documentation: [numbarrow docs](https://goykhman.github.io/numbarrow)

## License

See [LICENSE](LICENSE).
