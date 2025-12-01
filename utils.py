import numpy as np
from numba import carray, float64, int32, int64, intp, njit
from numba.core.types import Array, Type, uint8, voidptr
from numba.extending import intrinsic

from configurations import default_jit_options


@intrinsic
def _ptr_as_int_to_voidptr(typingctx, arg_type):
    def codegen(context, builder, signature, args):
        return builder.inttoptr(args[0], context.get_value_type(voidptr))
    return voidptr(arg_type), codegen


def numpy_array_from_ptr_factory(dtype_: Type):
    @njit(Array(dtype_, 1, "C")(intp, int64), **default_jit_options)
    def _(ptr_as_int: int, sz: int):
        return carray(_ptr_as_int_to_voidptr(ptr_as_int), shape=(sz,), dtype=dtype_)
    return _


arrays_viewers = {
    np.uint8: numpy_array_from_ptr_factory(uint8),
    np.float64: numpy_array_from_ptr_factory(float64),
    np.int32: numpy_array_from_ptr_factory(int32),
    np.int64: numpy_array_from_ptr_factory(int64)
}
