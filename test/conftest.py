import os
import shutil
import sys

import pytest


def spark_leg_required():
    """True when a Spark leg that cannot run must fail the run rather than skip it.

    Every Spark or JVM failure below turns into a skip, which keeps a machine
    without java useful but also means that losing every end-to-end test is a
    green run. A CI cell that is expected to run Spark sets
    NUMBARROW_REQUIRE_SPARK=1 and gets a failure that says why instead.
    """
    return os.environ.get("NUMBARROW_REQUIRE_SPARK") == "1"


def skip_or_fail(reason):
    if spark_leg_required():
        pytest.fail("NUMBARROW_REQUIRE_SPARK=1 but " + reason, pytrace=False)
    pytest.skip(reason)


def spark_unavailable_reason():
    """Reason a SparkSession cannot start here, or None when one can."""
    if sys.platform not in ("linux", "darwin"):
        return "spark tests run on linux and macos only, this platform is %s" % sys.platform
    # Mirrors how pyspark picks a java binary: a non-empty JAVA_HOME wins
    # outright and is never checked against the filesystem, so a JAVA_HOME
    # pointing nowhere fails even when java is on PATH. Only an unset
    # JAVA_HOME falls back to PATH.
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        runner = os.path.join(java_home, "bin", "java")
        if not os.access(runner, os.X_OK):
            return "JAVA_HOME is set but %s is not executable" % runner
    elif shutil.which("java") is None:
        return "JAVA_HOME is unset and there is no java on PATH"
    # The arrow path runs pyspark's own dependency checks, and those import
    # distutils, which PEP 632 removed from the standard library in 3.12 and
    # which only setuptools supplies there. Run the very checks the tests will
    # hit rather than approximating them: a stand-in for the dependency can
    # pass while the real check fails, which is exactly what a bare
    # `import distutils` did.
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
    """Reason spark's own arrow transport cannot run here, or None when it can.

    An unsupported java breaks the jvm half of the transport rather than
    anything on the python side: the arrow jars pyspark bundles reach for a
    `java.nio.DirectByteBuffer` constructor that a newer jvm no longer
    provides, and every batch then dies in `MemoryUtil` before numbarrow sees
    the data. Provoke it with a round trip that carries no numbarrow in it, so
    the jvm is measured rather than inferred from its version number, and so a
    failure here is never read as one of ours.
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
    reason = spark_unavailable_reason()
    if reason is not None:
        skip_or_fail(reason)
    # Imported here rather than at module scope so a missing pyspark skips
    # these tests instead of aborting collection for the whole suite.
    from pyspark.sql import SparkSession
    # Workers launch whatever python3 PATH resolves to unless this is set,
    # and that interpreter cannot import numbarrow.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    # Bind the driver to loopback rather than letting spark pick whichever
    # interface the hostname resolves to.
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
