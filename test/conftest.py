"""Fixtures and helpers that several test files share.

``spark`` is a session-scoped fixture that starts one local SparkSession with Arrow enabled. A test that takes it is
skipped where Spark cannot run, and fails instead when ``NUMBARROW_REQUIRE_SPARK`` is ``1``. ``Wrapped`` is a minimal
extension type for the tests of code that unwraps an extension type to reach its storage.
"""
import os
import shutil
import sys

import pyarrow as pa
import pytest


class Wrapped(pa.ExtensionType):
    """An extension type over any storage type, for the tests of code that unwraps an extension type.

    pyarrow 18 and later provide ``pa.opaque`` for this, but CI also runs older pyarrow versions. This class works on
    every version, so the code that unwraps an extension type is tested on each pyarrow that CI runs.
    """

    def __init__(self, storage_type):
        super().__init__(storage_type, "test.wrapped")

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return cls(storage_type)


def spark_leg_required():
    """Return True when a Spark test that cannot run must fail instead of being skipped.

    The ``spark`` fixture skips its tests wherever Spark cannot run, so a machine without Java can still run the rest
    of the suite. A run that loses every end-to-end test would then still pass. A CI job that is expected to run Spark
    sets ``NUMBARROW_REQUIRE_SPARK`` to ``1``, and each Spark test then fails with the reason. Any other value leaves
    the skips in place. ``test_mapinarrow_spark.py`` reads this function too, so that a missing pyspark or pandas
    fails the collection of that file instead of skipping it.
    """
    return os.environ.get("NUMBARROW_REQUIRE_SPARK") == "1"


def skip_or_fail(reason):
    """Skip the current test with ``reason``, or fail it with the reason when ``spark_leg_required()`` is true."""
    if spark_leg_required():
        pytest.fail("NUMBARROW_REQUIRE_SPARK=1 but " + reason, pytrace=False)
    pytest.skip(reason)


def spark_unavailable_reason():
    """Return the reason a SparkSession cannot start here, or None when one can."""
    if sys.platform not in ("linux", "darwin"):
        return "spark tests run on linux and macos only, this platform is %s" % sys.platform
    # The check follows Spark's launcher script, bin/spark-class, in choosing a java binary. A non-empty JAVA_HOME
    # wins, and the script never checks that JAVA_HOME/bin/java exists, so a JAVA_HOME that points nowhere fails even
    # when java is on PATH. An unset or empty JAVA_HOME falls back to the java on PATH.
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        runner = os.path.join(java_home, "bin", "java")
        if not os.access(runner, os.X_OK):
            return "JAVA_HOME is set but %s is not executable" % runner
    elif shutil.which("java") is None:
        return "JAVA_HOME is unset and there is no java on PATH"
    # pyspark's Arrow path runs its own dependency checks, which import distutils. PEP 632 removed distutils from the
    # standard library in Python 3.12, and setuptools is its only supplier there. The checks also require pandas and
    # pyarrow at minimum versions. A stand-in, such as a bare import of distutils, can pass where the real checks
    # fail, so this function calls the checks themselves.
    try:
        from pyspark.sql.pandas.utils import (
            require_minimum_pandas_version,
            require_minimum_pyarrow_version,
        )
        require_minimum_pandas_version()
        require_minimum_pyarrow_version()
    except Exception as exc:
        return "pyspark cannot run its arrow path here: %s" % exc
    return None


def arrow_transport_unusable(session):
    """Return the reason Spark's own Arrow transport fails here, or None when it works.

    An unsupported JVM breaks the JVM half of the transport, not the Python half. The Arrow jars that pyspark bundles
    call a ``java.nio.DirectByteBuffer`` constructor that Java 17 has and newer JVMs, such as Java 21, lack, so on such
    a JVM every batch fails in Arrow's ``MemoryUtil`` before numbarrow sees any data. The check runs a round trip with
    no numbarrow code in it. It tests the JVM itself instead of judging by its version number, and a failure here
    cannot be mistaken for a numbarrow defect.
    """
    def identity(batches):
        for batch in batches:
            yield batch

    try:
        session.range(1).mapInArrow(identity, "id long").collect()
    except Exception as exc:
        return "spark's own arrow transport fails here: %s" % " ".join(
            str(exc).split("\n")[:2])
    return None


@pytest.fixture(scope="session")
def spark():
    """A local SparkSession with Arrow enabled, shared by every test of the run and stopped at its end.

    Each test that takes the fixture is skipped where Spark cannot run here, or fails when ``spark_leg_required()`` is
    true.
    """
    reason = spark_unavailable_reason()
    if reason is not None:
        skip_or_fail(reason)
    # pyspark is imported here and not at the top of conftest.py, which pytest imports for every test. A top-level
    # import would stop the collection of the whole suite where pyspark is missing. In that case the checks above have
    # already skipped or failed the test.
    from pyspark.sql import SparkSession
    # Spark starts its Python workers with PYSPARK_PYTHON, or with the python3 on PATH when that is unset, and that
    # interpreter may not have numbarrow installed. The workers and the driver both use the interpreter that runs the
    # tests.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    # The tests bind Spark's driver to the loopback address, and keep a SPARK_LOCAL_IP that the caller has set.
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    session = (
        SparkSession
        .builder
        .appName("numba arrow runner")
        .master("local[2]")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1000")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    unusable = arrow_transport_unusable(session)
    if unusable is not None:
        session.stop()
        skip_or_fail(unusable)
    yield session
    session.stop()
