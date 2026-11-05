"""End-to-end tests of ``make_mapinarrow_func`` through ``mapInArrow`` in a local Spark session.

In ``mapInArrow``, Spark sends each partition's rows to Python as Arrow batches and reads the returned batches back in
the JVM. The tests run a jitted UDF over a joined DataFrame that has a list-of-struct column and a null. They check that
the null row of a struct column, the bitmap key of a null-free column, a struct field named like a column, and the
nulls of a :class:`Nullable` all survive that transport. They also check that a numeric column fails in the JVM when
its type differs from the declared numeric type, and that a Spark ``StructType`` is refused as ``output_schema``.
pytest skips the module without pyspark or pandas. The ``spark`` fixture in ``conftest.py`` skips a test when Spark
cannot start or its Arrow transport fails. With ``NUMBARROW_REQUIRE_SPARK=1``, each of these cases fails instead of
skipping. ``test_mapinarrow_factory.py`` tests the factory on batches built in Python.
"""
import numpy as np
import pytest

from numba import njit
from numba.core.types import Array, float64, int32, int64, Optional, uint8

from numbarrow.core.is_null import is_null
from numbarrow.core.configurations import jit_options
from numbarrow.core.mapinarrow_factory import Nullable, make_mapinarrow_func
from test.conftest import spark_leg_required

if spark_leg_required():
    # With NUMBARROW_REQUIRE_SPARK=1, a missing pyspark or pandas fails collection instead of skipping the module.
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
    """Return each entity's three ``(index, magnitude)`` rows as one list-of-struct column, ``data``.

    The DataFrame has two partitions by id and a ``partition_id`` column. Entity 101888 has a null magnitude at index
    905.
    """
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


# The data arrays the factory passes to a UDF are read-only, whether they view an Arrow buffer or are copies, so the
# signature declares each array parameter readonly. A readonly array type accepts a writable array too, such as the
# bitmap here. A writable array type makes numba refuse a read-only argument when the function is called.
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
    # data is a list-of-struct column, so data_dict holds its fields under data_dict["data"].
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
    """Every entity comes back from ``mapInArrow`` with the intensity that ``calculate`` gives it."""
    intensities = collect_intensities(spark)
    # calculate reads index_a[i] for entity i, but index_a is the flattened child of the list column. An entity is
    # therefore scored with whichever index value sits at its own position in its batch, so the test accepts each of
    # the three values that every entity holds.
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
    """The null magnitude of entity 101888 is left out of the product of its magnitudes.

    ``calculate`` skips a magnitude that the bitmap marks null. Spark leaves 0.0 under the null, so reading it as a
    value would make the product zero, and the scale would drop to the bare index.
    """
    intensities = collect_intensities(spark)
    # The product holds only the two valid magnitudes, 12.4 and 13.6.
    area = 137 * 0.56
    scale = np.sqrt(12.4 * 13.6) / 121.0
    candidates = [area * (index + scale) * 10 ** (-4) for index in (904, 905, 906)]
    assert any(
        np.isclose(intensities["101888"], candidate, rtol=1e-12, atol=0.0)
        for candidate in candidates
    )
    # A zero in the product would leave the bare index as the scale.
    collapsed = [area * index * 10 ** (-4) for index in (904, 905, 906)]
    assert not any(
        np.isclose(intensities["101888"], candidate, rtol=1e-9, atol=0.0)
        for candidate in collapsed
    )


def test_struct_null_row_survives_the_arrow_transport(spark):
    """A null row of a struct column reaches the UDF through Spark as a null in its field's bitmap.

    Spark's Arrow writer also sets the child null under a null struct row, so the field's own validity shows the row
    before the factory folds in the struct's validity. ``test_struct_null_row_reaches_the_udf`` and
    ``test_both_null_layers_are_folded_together`` in ``test_mapinarrow_factory.py`` test that fold on a child that is
    valid under the null struct row.
    """
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
    """A column with no null in its batch still has a key in ``bitmap_dict`` when it comes through Spark.

    pyarrow's IPC reader drops the validity buffer of a column whose ``null_count`` is 0, so the column arrives with
    no bitmap. Without the key, a UDF that indexes ``bitmap_dict`` would pass on a batch with nulls and raise
    ``KeyError`` on a null-free batch of the same schema.
    """
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
    """A column ``id`` and a struct field ``order.id`` both reach the UDF through Spark, each under its own key.

    ``data_dict`` holds the field under its struct column, as ``data_dict["order"]["id"]``.
    """
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
    """``Nullable`` columns of long, double and string values carry their nulls back through Spark.

    As bare arrays, the row ``(1, None, None, None)`` would come back ``(1, 0, 0.0, '')``, the values that the UDF
    receives under the nulls.
    """
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
    """A numeric column whose Arrow type differs from its declared Spark type fails in the JVM, whatever its width.

    Spark wraps each returned Arrow vector in an accessor for the vector's type, and reads it with the getter of the
    column's declared type, such as ``getLong`` for ``LongType``. An accessor without that getter throws
    ``UnsupportedOperationException``. float64 and int64 are both 64 bits wide, and neither reads through the other's
    accessor; int32 under ``LongType`` fails the same way. The test also asserts the two phrases of
    ``make_mapinarrow_func``'s docstring that state this.
    """
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


def test_a_spark_struct_type_as_output_schema_is_refused():
    """A Spark ``StructType`` passed as ``output_schema`` raises ``TypeError`` when the function is made.

    ``mapInArrow`` takes a ``StructType`` as its output schema, while the factory's ``output_schema`` is a
    ``pyarrow.Schema``. Without the check, the factory would fail on a field without ``.type``, with an error that
    names neither the parameter nor the type it needs. The test needs pyspark's types but no Spark session.
    """
    schema = StructType([StructField("n", LongType())])
    with pytest.raises(TypeError, match="output_schema must be a pyarrow.Schema, not a StructType"):
        make_mapinarrow_func(lambda d, b, br: {}, output_schema=schema)
