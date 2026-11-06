numbarrow.utils
===============

The two modules of this package hold the code that :doc:`numbarrow.core.adapters` is built on. Most programs need
neither: :func:`~numbarrow.core.adapters.arrow_array_adapter` chooses the right conversion, and the arrays it returns
stay valid for as long as the caller holds them.

numbarrow.utils.utils
---------------------

Overview
''''''''

The functions of this module let compiled code view raw memory as a numpy array. The dict ``arrays_viewers`` maps a
numpy dtype to a jitted function, called a viewer, that takes a memory address and an element count. The viewer returns
a numpy array of that dtype over the memory at the address, without a copy. The adapters pass a viewer the ``address``
of a ``pyarrow.Buffer``, such as a validity buffer or the offsets of a string array, and they return copies, never the
viewer's array itself. ``arrays_viewers`` compiles a viewer the first time its dtype is requested and keeps it for the
rest of the process.

A viewer's array does not own its memory, does not keep that memory alive, and is not read-only. The array is valid
only while that memory stays allocated, and a write through it changes the memory.
:func:`~numbarrow.core.adapters.arrow_array_adapter` returns read-only arrays that stay valid while you hold them: a
view that keeps its Arrow buffer alive, or a copy that owns its memory. Use the adapter unless your code manages the
memory itself.

.. automodule:: numbarrow.utils.utils
   :members:
   :show-inheritance:
   :undoc-members:

numbarrow.utils.arrow_array_utils
---------------------------------

Overview
''''''''

This module holds the conversions that :func:`~numbarrow.core.adapters.arrow_array_adapter` calls for numeric, string,
struct and list arrays. :func:`~numbarrow.utils.arrow_array_utils.uniform_arrow_array_adapter` views an array of
fixed-width values, and :func:`~numbarrow.utils.arrow_array_utils.create_str_array` copies a string array into
fixed-width Unicode. :func:`~numbarrow.utils.arrow_array_utils.structured_array_adapter` converts a struct array field
by field, and :func:`~numbarrow.utils.arrow_array_utils.structured_list_array_adapter` converts a list of structs. Each
of them copies the validity bitmap with :func:`~numbarrow.utils.arrow_array_utils.create_bitmap`, which takes an Arrow
buffer, an offset and a length rather than an array.

Each of the four conversions takes the pyarrow array directly, but ``arrow_array_adapter`` is the function to call: it
chooses the conversion by the class of the array and accepts every supported type. The reference below documents what
each function returns and what it raises.

.. automodule:: numbarrow.utils.arrow_array_utils
   :members:
   :show-inheritance:
   :undoc-members:
