numbarrow.core.adapters
=======================

Overview
++++++++

numba cannot compile a function that takes a ``pyarrow.Array``. :func:`~numbarrow.core.adapters.arrow_array_adapter`
converts such an array into a pair of numpy arrays, ``(bitmap, data)``, that an ``@njit`` function can take: the
validity bitmap that marks the null rows, and the values. Call it outside jitted code, once per array, and pass the
results to the jitted function:

.. code-block:: python

    import pyarrow as pa
    from numbarrow.core.adapters import arrow_array_adapter

    bitmap, data = arrow_array_adapter(pa.array([1.5, None, 4.0]))
    print(bitmap, data)   # [5] [1.5 0.  4. ]

``bitmap`` is a ``uint8`` array with one bit per row, set for a valid row. In the example, ``[5]`` is ``0b101``: rows
0 and 2 are valid, and row 1 is null. ``bitmap`` is ``None`` when the array has no validity buffer, the Arrow memory
that marks its nulls, so code that reads a bitmap must accept ``None`` as well as an array. ``data`` is a read-only
numpy array. For a fixed-width type such as ``int64``, ``float64`` or a timestamp, it is a view of the Arrow buffer,
and it keeps that buffer alive. For booleans, ``date32`` values and strings, which numpy stores differently, it is a
copy.

A ``StructArray`` converts to a 3-tuple instead: the struct's own bitmap, then two dicts keyed by field name, one of
field bitmaps and one of field data. A ``ListArray`` whose elements are structs converts to the same 3-tuple, over the
elements of all its rows, and its first item marks null elements rather than null rows. The adapter keeps the struct's
bitmap apart from the field bitmaps, so pass both to :func:`~numbarrow.core.is_null.is_null_struct`.
:func:`~numbarrow.core.mapinarrow_factory.make_mapinarrow_func` combines them for the function it calls instead.

The function accepts the array classes that the README lists, and their subclasses. A ``MapArray``, for example, is a
``ListArray`` of structs with the fields ``key`` and ``value``, and it converts as one. Any other array, and any object
that is not a pyarrow array, raises ``NotImplementedError`` with the type in the message.

The README's `Supported Types`_ section has the table of classes and what each becomes. Its subsections cover what code
that reads the results must handle: `nulls and validity bitmaps`_, `strings`_, `timestamps and time zones`_,
`structs, lists of structs and maps`_, `read-only arrays`_ and `explicit signatures`_.

.. _Supported Types: https://github.com/Goykhman/numbarrow#supported-types
.. _nulls and validity bitmaps: https://github.com/Goykhman/numbarrow#nulls-and-validity-bitmaps
.. _strings: https://github.com/Goykhman/numbarrow#strings
.. _timestamps and time zones: https://github.com/Goykhman/numbarrow#timestamps-and-time-zones
.. _structs, lists of structs and maps: https://github.com/Goykhman/numbarrow#structs-lists-of-structs-and-maps
.. _read-only arrays: https://github.com/Goykhman/numbarrow#read-only-arrays
.. _explicit signatures: https://github.com/Goykhman/numbarrow#explicit-signatures

Module
++++++

.. automodule:: numbarrow.core.adapters
   :members:
   :show-inheritance:
   :undoc-members:
