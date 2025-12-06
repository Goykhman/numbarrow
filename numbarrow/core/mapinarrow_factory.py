import numpy as np
import pyarrow as pa

from pyspark.broadcast import Broadcast
from typing import Callable, List, Optional, Set

from numbarrow.utils.arrow_array_utils import (
    create_str_array, structured_list_array_adapter, uniform_arrow_array_adapter
)


def make_mapinarrow_func(
    main_func: Callable,
    input_columns: List[str],
    output_columns: List[str],
    broadcast: Optional[Broadcast] = None,
    string_type_columns: Optional[Set[str]] = None,
    struct_type_columns: Optional[Set[str]] = None,
):
    broadcasts = broadcast.value if broadcast is not None else {}
    string_type_columns = string_type_columns or set()
    struct_type_columns = struct_type_columns or set()

    def mapinarrow_func_(iterator):
        for batch in iterator:
            data_dict = {}
            bitmap_dict = {}
            for col in input_columns:
                col_pa = batch.column(col)
                if col in string_type_columns:
                    data_dict[col] = create_str_array(col_pa)
                elif col in struct_type_columns:
                    col_bitmap_dict, col_np_dict = structured_list_array_adapter(col_pa)
                    for c, bm in col_np_dict.items():
                        data_dict[c] = bm
                    for c, bm in col_bitmap_dict.items():
                        bitmap_dict[c] = bm
                else:
                    col_bitmap, col_np = uniform_arrow_array_adapter(col_pa)
                    data_dict[col] = col_np
                    bitmap_dict[col] = col_bitmap
            outputs = main_func(data_dict, bitmap_dict, broadcasts)
            outputs_dict = {}
            for col in output_columns:
                if col in outputs:
                    output = outputs[col]
                    assert isinstance(output, np.ndarray), f"{output} of type {type(output)} but should be numpy array"
                    outputs_dict[col] = pa.array(outputs[col])
                elif col in input_columns:
                    outputs_dict[col] = batch.column(col)
                else:
                    raise ValueError(f"Could not find {col} neither among inputs not in outputs")
            yield pa.RecordBatch.from_pydict(outputs_dict)
    return mapinarrow_func_
