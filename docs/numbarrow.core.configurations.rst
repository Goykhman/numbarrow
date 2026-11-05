numbarrow.core.configurations
=============================

Overview
++++++++

numbarrow compiles its own numba functions, such as :func:`~numbarrow.core.is_null.is_null` and the functions that read
Arrow buffers, with the options that this module holds. The ``NUMBARROW_JIT_OPTIONS`` environment variable sets them
as a JSON object, which numbarrow passes to ``@njit`` as keyword arguments. If the variable is unset or empty, the
options are ``{"cache": true}``. Set the variable to turn on bounds checks while you debug, or to turn caching off
where numba cannot write a cache.

:func:`~numbarrow.core.configurations.get_jit_options` reads and checks the variable. numbarrow calls it once, when this
module is first imported, and keeps the result in ``jit_options``. The modules that you import from numbarrow, such as
:doc:`numbarrow.core.adapters`, all import this one, so set the variable before your program imports any of them. A
change after that import has no effect.

:func:`~numbarrow.core.configurations.jit_with_options` is the decorator that numbarrow's own functions use: ``@njit``
with ``jit_options`` and a signature. If caching is on and numba cannot write a cache for a function, the decorator
compiles the function without one and issues a ``RuntimeWarning``. The warning says how to get a cache: set
``NUMBA_CACHE_DIR`` to a writable directory, or, if numbarrow is imported from an archive, install it unpacked or
import it from a ``.zip``, which numba 0.61 and later can cache. ``NUMBARROW_JIT_OPTIONS='{"cache": false}'`` turns
caching off instead and silences the warning.

The README section on `compilation and the cache`_ explains when numba cannot write a cache, and why a change of
options needs a new ``NUMBA_CACHE_DIR`` while caching stays on.

.. _compilation and the cache: https://github.com/Goykhman/numbarrow#compilation-and-the-cache

Module
++++++

.. automodule:: numbarrow.core.configurations
   :members:
   :show-inheritance:
   :undoc-members:
