#
# Copyright (c) 2022
#

from typing import Any
from types import FunctionType

import builtins

def contains(obj: Any, item: Any) -> bool:
    return (item in obj)

builtins.contains = contains

_reflective_ops = {
    '__lt__'       : '__gt__',
    '__le__'       : '__ge__',
    '__gt__'       : '__lt__',
    '__ge__'       : '__le__',
    '__eq__'       : '__eq__',
    '__ne__'       : '__ne__',
    '__add__'      : '__radd__',
    '__sub__'      : '__rsub__',
    '__mul__'      : '__rmul__',
    '__matmul__'   : '__rmatmul__',
    '__truediv__'  : '__rtruediv__',
    '__floordiv__' : '__rfloordiv__',
    '__mod__'      : '__rmod__',
    '__divmod__'   : '__rdivmod__',
    '__pow__'      : '__rpow__',
    '__lshift__'   : '__rlshift__',
    '__rshift__'   : '__rrshift__',
    '__and__'      : '__rand__',
    '__xor__'      : '__rxor__',
    '__or__'       : '__ror__',
    '__radd__'     : '__add__',
    '__rsub__'     : '__sub__',
    '__rmul__'     : '__mul__',
    '__rmatmul__'  : '__matmul__',
    '__rtruediv__' : '__truediv__',
    '__rfloordiv__': '__floordiv__',
    '__rmod__'     : '__mod__',
    '__rdivmod__'  : '__divmod__',
    '__rpow__'     : '__pow__',
    '__rlshift__'  : '__lshift__',
    '__rrshift__'  : '__rshift__',
    '__rand__'     : '__and__',
    '__rxor__'     : '__xor__',
    '__ror__'      : '__or__',
}

def emulate(method: FunctionType, _self: Any) -> FunctionType:
    if method.__name__ in _reflective_ops:
        def emul_reflective(*args, **kwargs) -> Any:
            try:
                out = method(*args, **kwargs)
                if out is NotImplemented:
                    raise AttributeError

                return out

            except AttributeError:
                pass

            rop = _reflective_ops[method.__name__]
            o = args[0]

            method = getattr(o, rop)
            out = method(_self)

            return out

        return emul_reflective

    else:
        return method
