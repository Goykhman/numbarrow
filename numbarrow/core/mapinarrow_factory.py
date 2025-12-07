import numpy as np
import pyarrow as pa

from typing import Callable, Dict, List, Optional

from numbarrow.core.adapters import arrow_array_adapter


def make_mapinarrow_func(
    main_func: Callable,
    input_columns: Optional[List[str]] = None,
    broadcasts: Optional[Dict] = None
):
    """
    Creates a function that can be given as an argument to `mapInArrow`

    :param main_func: should have the following signature:
        - `data_dict: Dict[str, np.ndarray]`, values are arrays of data of various supported types
        - `bitmap_dict: Dict[str, np.ndarray]`, values are uint8 aligned arrays of bitmap data
        - `broadcasts` Optional[Dict[str, Any]]
        returns: Dict[str, np.ndarray] that will be used to create PyArrow `RecordBatch`
    :param input_columns: optional list of column names that will be expected to be needed for in
    `data_dict` for the calculation done by `main_func`. When not given, all columns in the iterated
     over PySpark DataFrame will be used.
    :param broadcasts: optional dictionary of broadcast values
    """
    broadcasts = broadcasts if broadcasts is not None else {}

    def _(iterator):
        for batch in iterator:
            data_dict: Dict[str, np.ndarray] = {}
            bitmap_dict: Dict[str, np.ndarray] = {}
            input_columns_ = input_columns if input_columns is not None else batch.schema.names
            for col in input_columns_:
                col_pa: pa.Array = batch.column(col)
                col_bitmap, col_data = arrow_array_adapter(col_pa)
                col_bitmap = col_bitmap if isinstance(col_bitmap, dict) else {} if col_bitmap is None else {col: col_bitmap}  # noqa: E501
                col_data = col_data if isinstance(col_data, dict) else {col: col_data}
                bitmap_dict = {**bitmap_dict, **col_bitmap}
                data_dict = {**data_dict, **col_data}
            outputs = main_func(data_dict, bitmap_dict, broadcasts)
            yield pa.RecordBatch.from_pydict({col: pa.array(output) for col, output in outputs.items()})
    return _
