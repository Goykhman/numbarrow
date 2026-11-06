.. numbarrow documentation master file, created by sphinx-quickstart


numbarrow
=========

numbarrow lets numba ``@njit`` functions read `Apache Arrow <https://arrow.apache.org/docs/python/>`_ data, such as the
record batches that Spark sends to Python in ``mapInArrow``. It converts an Arrow array into numpy arrays that numba
accepts: the values, and a validity bitmap that marks the null rows.

The user guide is the README, on `GitHub <https://github.com/Goykhman/numbarrow>`_ and on
`PyPI <https://pypi.org/project/numbarrow/>`_. It covers installation, the supported types, nulls, strings and time
zones, PySpark's ``mapInArrow``, the compilation cache and the supported versions. This site is the API reference.
:doc:`modules` lists the modules, and the documentation of each module opens with an overview of what the module is
for and when to use it.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
