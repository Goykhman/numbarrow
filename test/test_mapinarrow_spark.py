import numpy as np
import pytest

from numba import njit
from numba.core.types import Array, float64, int32, int64, Optional, uint8

from numbarrow.core.is_null import is_null
from numbarrow.core.configurations import jit_options
from numbarrow.core.mapinarrow_factory import Nullable, make_mapinarrow_func
from test.conftest import spark_leg_required

if spark_leg_required():
    # A missing pyspark is then a collection error, not a silent skip.
    import pandas  # noqa: F401
    import pyspark  # noqa: F401
else:
    pytest.importorskip("pyspark")
    pytest.importorskip("pandas")

from pyspark.sql import functions as sf  # noqa: E402
from pyspark.sql.types import (  # noqa: E402
    BooleanType, DoubleType, IntegerType, LongType, StringType, StructField, StructType
)


def create_entities_df(spark):
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
    return entities_df.select("*", sf.spark_partition_id().alias("partition_id"))


def create_data_df(spark):
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
    return data_df.select("*", sf.spark_partition_id().alias("partition_id"))


def join_data(spark):
    entities_df = create_entities_df(spark)
    data_df = create_data_df(spark)
    return data_df.join(entities_df, on=["id", "partition_id"])


# Every array numbarrow hands a UDF is a view over an Arrow buffer the UDF does
# not own, so the parameters are declared readonly. A readonly parameter accepts
# a writable array as well, while a writable one rejects a readonly array at
# compile time.
bitmap_a_ty = Array(uint8, 1, "C", readonly=True)

size_a_ty = Array(int64, 1, "C", readonly=True)
coordinate_a_ty = Array(float64, 1, "C", readonly=True)
index_a_ty = Array(int32, 1, "C", readonly=True)
magnitude_a_ty = Array(float64, 1, "C", readonly=True)
magnitude_bmap_ty = Optional(bitmap_a_ty)
calculate_ret_ty = Array(float64, 1, "C")

calculate_sig = calculate_ret_ty(
    size_a_ty, coordinate_a_ty, index_a_ty, magnitude_a_ty, magnitude_bmap_ty, float64
)


@njit(calculate_sig, **jit_options)
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
    # `data` is a list-of-struct column, so its fields sit under its own name.
    res = calculate(
        data_dict["size"],
        data_dict["coordinate"],
        data_dict["data"]["index"],
        data_dict["data"]["magnitude"],
        bitmap_dict["data"]["magnitude"],
        broadcasts["rescale"]
    )
    return {"id": data_dict["id"], "intensity": res}


def collect_intensities(spark):
    output_schema = StructType([
        StructField("id", StringType()),
        StructField("intensity", DoubleType())
    ])
    broadcast = spark.sparkContext.broadcast({"rescale": 10 ** (-4)})
    mapinarrow_func = make_mapinarrow_func(main, broadcasts=broadcast.value)
    df_out = join_data(spark).mapInArrow(mapinarrow_func, output_schema)
    return {row["id"]: row["intensity"] for row in df_out.collect()}


def test_mapinarrow_round_trips_every_entity(spark):
    intensities = collect_intensities(spark)
    # `calculate` reads index_a[i] for row i while index_a is the flattened
    # list child array, so an entity is scored with whichever of the three
    # index values sits at its own position within its batch.
    expected = {
        "101888": (137 * 0.56, 12.4 * 13.6),
        "102": (141 * 0.12, 14.5 * 14.8 * 15.1),
        "103": (172 * 0.79, 9.7 * 8.8 * 9.4),
        "104": (271 * 0.44, 10.2 * 10.4 * 10.9),
    }
    assert sorted(intensities) == sorted(expected)
    for entity_id, (area, total_magnitude) in expected.items():
        scale = np.sqrt(total_magnitude) / 121.0
        candidates = [area * (index + scale) * 10 ** (-4) for index in (904, 905, 906)]
        assert any(
            np.isclose(intensities[entity_id], candidate, rtol=1e-12, atol=0.0)
            for candidate in candidates
        ), "%s: %r matched none of %r" % (entity_id, intensities[entity_id], candidates)


def test_null_magnitude_is_excluded_from_the_product(spark):
    intensities = collect_intensities(spark)
    # 101888 carries a null magnitude at index 905. Reading it as a value
    # rather than skipping it on the bitmap would fold a zero into the
    # product and collapse the intensity to zero.
    area = 137 * 0.56
    scale = np.sqrt(12.4 * 13.6) / 121.0
    candidates = [area * (index + scale) * 10 ** (-4) for index in (904, 905, 906)]
    assert any(
        np.isclose(intensities["101888"], candidate, rtol=1e-12, atol=0.0)
        for candidate in candidates
    )
    # Reading the null as a value folds a zero into the product, so the scale
    # collapses to the bare index. `> 0.0` could not fail and did not say this.
    collapsed = [area * index * 10 ** (-4) for index in (904, 905, 906)]
    assert not any(
        np.isclose(intensities["101888"], candidate, rtol=1e-9, atol=0.0)
        for candidate in collapsed
    )


def test_struct_null_row_survives_the_arrow_transport(spark):
    # A struct-null row reaches the worker with no child validity buffer, so
    # the field's own bitmap cannot see it. Spark's transport is an Arrow IPC
    # round trip, which preserves that shape, and the folded bitmap catches it.
    schema = StructType([
        StructField("id", StringType()),
        StructField("point", StructType([StructField("v", LongType())])),
    ])
    df = spark.createDataFrame([("a", {"v": 10}), ("b", None), ("c", {"v": 30})], schema)

    def main(data_dict, bitmap_dict, broadcasts):
        bitmap = bitmap_dict["point"]["v"]
        n = len(data_dict["point"]["v"])
        flags = np.array(
            [False] * n if bitmap is None else [is_null(i, bitmap) for i in range(n)],
            dtype=np.bool_
        )
        return {"id": data_dict["id"], "was_null": flags}

    out_schema = StructType([
        StructField("id", StringType()), StructField("was_null", BooleanType())
    ])
    rows = df.repartition(1).mapInArrow(make_mapinarrow_func(main), out_schema).collect()
    assert {row["id"]: row["was_null"] for row in rows} == {"a": False, "b": True, "c": False}


def test_null_free_batch_still_carries_its_bitmap_key(spark):
    # The Arrow IPC reader normalises null_count == 0 to validity=None, so a
    # batch that happens to contain no nulls arrives with no validity buffer.
    # The key must still be there, or a job passes on one batch and raises
    # KeyError on the next batch of the same schema.
    schema = StructType([StructField("id", StringType()), StructField("magnitude", DoubleType())])
    df = spark.createDataFrame([("a", 1.0), ("b", 2.0), ("c", 3.0)], schema)

    def main(data_dict, bitmap_dict, broadcasts):
        present = np.array(["magnitude" in bitmap_dict] * len(data_dict["id"]), dtype=np.bool_)
        return {"id": data_dict["id"], "key_present": present}

    out_schema = StructType([
        StructField("id", StringType()), StructField("key_present", BooleanType())
    ])
    rows = df.repartition(1).mapInArrow(make_mapinarrow_func(main), out_schema).collect()
    assert [row["key_present"] for row in rows] == [True, True, True]


def test_a_struct_field_sharing_a_column_name_reaches_the_udf_through_spark(spark):
    # Four ordinary Spark StructTypes convert to this shape: a top-level column
    # and a struct field sharing a name. The field sits under its column, so
    # both arrive, each under its own key.
    schema = StructType([
        StructField("id", LongType()),
        StructField("order", StructType([
            StructField("id", LongType()), StructField("total", DoubleType())
        ])),
    ])
    df = spark.createDataFrame([(1, {"id": 10, "total": 1.5}), (2, {"id": 20, "total": 2.5})], schema)

    def main(data_dict, bitmap_dict, broadcasts):
        return {"id": data_dict["id"], "order_id": data_dict["order"]["id"]}

    out_schema = StructType([StructField("id", LongType()), StructField("order_id", LongType())])
    rows = df.repartition(1).mapInArrow(make_mapinarrow_func(main), out_schema).collect()
    assert sorted((row["id"], row["order_id"]) for row in rows) == [(1, 10), (2, 20)]


def test_nullable_carries_nulls_back_through_spark(spark):
    # A bare array republishes a null as whatever sat under it, so a row of
    # (1, None, None, None) came back (1, 0, 0.0, ''). Nullable carries each
    # column's validity out, and Spark's Arrow transport carries it back.
    schema = StructType([
        StructField("id", LongType()), StructField("n", LongType()),
        StructField("x", DoubleType()), StructField("s", StringType()),
    ])
    df = spark.createDataFrame([(1, None, None, None), (2, 20, 2.5, "b")], schema)

    def main(data_dict, bitmap_dict, broadcasts):
        out = {"id": data_dict["id"]}
        for name in ("n", "x", "s"):
            out[name] = Nullable(data_dict[name], bitmap_dict[name])
        return out

    rows = df.repartition(1).mapInArrow(make_mapinarrow_func(main), schema).collect()
    assert sorted((row["id"], row["n"], row["x"], row["s"]) for row in rows) == [
        (1, None, None, None), (2, 20, 2.5, "b")
    ]


def test_a_column_of_another_accessor_family_fails_whatever_its_width(spark):
    # The docstring said only a width mismatch fails, so an operator chasing
    # UnsupportedOperationException was told the types could not be the cause.
    # float64 and int64 are both 64 bits wide and neither reads through the
    # other's accessor.
    doc = " ".join(make_mapinarrow_func.__doc__.split())
    assert "read through the accessor of another family fails in the JVM" in doc
    assert "float64 under ``LongType`` and int64 under ``DoubleType``" in doc
    df = spark.createDataFrame([(1,), (2,)], StructType([StructField("v", LongType())]))

    def collect_as(dtype, declared):
        def main(data_dict, bitmap_dict, broadcasts):
            return {"n": data_dict["v"].astype(dtype)}

        out_schema = StructType([StructField("n", declared)])
        return df.repartition(1).mapInArrow(make_mapinarrow_func(main), out_schema).collect()

    refused = [
        ("float64 under LongType", np.float64, LongType()),
        ("int64 under DoubleType", np.int64, DoubleType()),
        ("int32 under LongType", np.int32, LongType()),
    ]
    for label, dtype, declared in refused:
        with pytest.raises(Exception) as excinfo:
            collect_as(dtype, declared)
        assert "UnsupportedOperationException" in str(excinfo.value), label
    assert [row["n"] for row in collect_as(np.float64, DoubleType())] == [1.0, 2.0]
