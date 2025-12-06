import pyarrow as pa

from pyspark.broadcast import Broadcast
from typing import Callable, List, Optional

from numbarrow.utils.arrow_array_utils import arrow_array_adapter


def make_mapinarrow_func(
    main_func: Callable,
    input_columns: List[str],
    broadcast: Optional[Broadcast] = None
):
    """
    Create function that can be given as argument to `mapInArrow`

    :param main_func: should take the following parameters:
        - `data_dict: Dict[str, np.ndarray]`, values are arrays of data of various supported types
        - `bitmap_dict: Dict[str, np.ndarray]`, values are uint8 aligned arrays of bitmap data
        - `broadcasts` Dict[str, Any]
    :param input_columns: list of column names that matches keys of `data_dict`
    :param broadcast: optional dictionary of broadcast values
    """
    broadcasts = broadcast.value if broadcast is not None else {}

    def _(iterator):
        for batch in iterator:
            data_dict = {}
            bitmap_dict = {}
            for col in input_columns:
                col_pa = batch.column(col)
                col_bitmap, col_data = arrow_array_adapter(col_pa)
                col_bitmap = {col: col_bitmap} if not isinstance(col_bitmap, dict) else col_bitmap
                col_data = {col: col_data} if not isinstance(col_data, dict) else col_data
                bitmap_dict = {**bitmap_dict, **col_bitmap}
                data_dict = {**data_dict, **col_data}
            outputs = main_func(data_dict, bitmap_dict, broadcasts)
            yield pa.RecordBatch.from_pydict({col: pa.array(output) for col, output in outputs.items()})
    return _
