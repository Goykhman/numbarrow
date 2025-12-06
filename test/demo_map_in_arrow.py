import numpy as np

from numba import njit
from numba.core.types import Array, float64, int32, int64, Optional, uint8
from pyspark.sql import functions as sf
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType, IntegerType, LongType, StringType, StructField, StructType
)

from numbarrow.core.is_null import is_null
from numbarrow.core.configurations import default_jit_options
from numbarrow.core.mapinarrow_factory import make_mapinarrow_func


spark = (
    SparkSession
    .builder
    .appName("numba arrow runner")
    .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1000")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .getOrCreate()
)


def create_entities_df():
    entities_schema = StructType([
        StructField("id", StringType()),
        StructField("size", LongType()),
        StructField("coordinate", DoubleType()),
    ])
    entities_df = spark.createDataFrame([
        ("101888", 137, 0.56),
        ("102", 141, 0.12),
        ("103", 172, 0.79),
        ("104", 271, 0.44),
    ], entities_schema)
    entities_df = entities_df.repartition(2, "id")
    entities_df = entities_df.select("*", sf.spark_partition_id().alias("partition_id"))
    # entities_df.show()
    return entities_df, entities_schema


def create_data_df():
    data_schema = StructType([
        StructField("id", StringType()),
        StructField("index", IntegerType()),
        StructField("magnitude", DoubleType()),
    ])
    data_df = spark.createDataFrame([
        ("101888", 904, 12.4),
        ("101888", 905, None),
        ("101888", 906, 13.6),
        ("102", 904, 14.5),
        ("102", 905, 14.8),
        ("102", 906, 15.1),
        ("103", 904, 9.7),
        ("103", 905, 8.8),
        ("103", 906, 9.4),
        ("104", 904, 10.2),
        ("104", 905, 10.4),
        ("104", 906, 10.9),
    ], data_schema)
    data_df = data_df.groupby("id").agg(
        sf.collect_list(sf.struct(*[c for c in data_df.columns if c != "id"])).alias("data")
    )
    data_df = data_df.repartition(2, "id")
    data_df = data_df.select("*", sf.spark_partition_id().alias("partition_id"))
    # data_df.show()
    return data_df, data_schema


def join_data():
    entities_df, entities_schema = create_entities_df()
    data_df, data_schema = create_data_df()
    joined_df = data_df.join(entities_df, on=["id", "partition_id"])
    joined_df.show()
    return joined_df


bitmap_a_ty = Array(uint8, 1, "C")

size_a_ty = Array(int64, 1, "C")
coordinate_a_ty = Array(float64, 1, "C")
index_a_ty = Array(int32, 1, "C")
magnitude_a_ty = Array(float64, 1, "C")
magnitude_bmap_ty = Optional(bitmap_a_ty)
calculate_ret_ty = Array(float64, 1, "C")

calculate_sig = calculate_ret_ty(
    size_a_ty, coordinate_a_ty, index_a_ty, magnitude_a_ty, magnitude_bmap_ty, float64
)


@njit(calculate_sig, **default_jit_options)
def calculate(
    size_a: np.ndarray,
    coordinate_a: np.ndarray,
    index_a: np.ndarray,
    magnitude_a: np.ndarray,
    magnitude_bmap: np.ndarray,
    rescale: float
) -> np.ndarray:
    res = np.empty(size_a.shape, np.float64)
    magnitudes_per_entity = len(magnitude_a) // len(size_a)
    for i in range(size_a.shape[0]):
        area = size_a[i] * coordinate_a[i]
        total_magnitude = 1
        for j in range(magnitudes_per_entity * i, magnitudes_per_entity * (i + 1)):
            if magnitude_bmap is None or not is_null(j, magnitude_bmap):
                total_magnitude *= magnitude_a[j]
        scale = index_a[i] + np.sqrt(total_magnitude) / 121.0
        res[i] = area * scale * rescale
    return res


def main(data_dict: dict, bitmap_dict: dict, broadcasts: dict):
    res = calculate(
        data_dict["size"],
        data_dict["coordinate"],
        data_dict["index"],
        data_dict["magnitude"],
        bitmap_dict["magnitude"],
        broadcasts["rescale"]
    )
    return {"intensity": res}


def run_demo():
    input_columns = ["size", "coordinate", "data", "id"]
    output_schema = StructType([
        StructField("id", StringType()),
        StructField("intensity", DoubleType())
    ])
    output_columns = output_schema.fieldNames()
    string_type_columns = {"id"}
    struct_type_columns = {"data"}
    sc = spark.sparkContext
    broadcast = sc.broadcast({"rescale": 10 ** (-4)})
    mapinarrow_func = make_mapinarrow_func(
        main,
        input_columns,
        output_columns,
        broadcast=broadcast,
        string_type_columns=string_type_columns,
        struct_type_columns=struct_type_columns
    )
    df_in = join_data()
    df_out = df_in.mapInArrow(mapinarrow_func, output_schema)
    df_out.show()


if __name__ == "__main__":
    run_demo()
