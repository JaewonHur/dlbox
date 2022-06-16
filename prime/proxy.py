#
# Copyright (c) 2022
#
from __future__ import annotations
from typing import List, Dict, Union, Any
from types import NotImplementedType

import prime
from prime import utils
from prime.client import PrimeClient

from prime_pb2 import *
from prime_pb2_grpc import *

# Run PrimeServer and PrimeClient
utils.run_server()
_client = PrimeClient()

"""
Proxy type is used to wrap the referenced variable allocated in DE.
This type variables behave as if they are in FE,
but they are actually evaluated in DE.

Supported types of referenced variable:

  None
  NotImplemented
  Ellipsis

+ Numbers
+   int
+   bool
+   float
+   complex
  Immutable sequences
    String
    Tuple
    Bytes
  Mutable sequences
    List
    Byte array
  Sets
    Set
    Frozen set
  Mappings
    Dictionary

  Callable types
    User-defined function
    Instance method
    Generator function
    Coroutine function
    Asynchronous generator function
    Built-in function
    Built-in method
-   Class
    Class instance

  Module
- Custom class
  Class instance

  I/O object
  Internal types
    Code object
    Frame object
    Traceback object
    Slice object
    Static method object
    Class method object
"""


class Proxy(object):
    _client: PrimeClient = _client

    def __init__(self, ref: str):
        # TODO: Need to check referenced variable is not class definition
        self._ref = ref

    def __getattr__(self, name: str) -> Proxy:
        name_d = self._client.AllocateObj(name)
        attr_d = self._client.InvokeMethod('__main__', 'getattr',
                                           [name_d])

        return Proxy(attr_d)

    def _get_ref(self, obj: Union[Any, Proxy]) -> str:
        if isinstance(obj, Proxy):
            return obj._ref
        else:
            return self._client.AllocateObj(obj)

    def _reflective_prime_op(func):
        __reflective_ops = {
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


        def wrapper(self, *args, **kwargs) -> Union[Proxy, NotImplementedType]:
            assert len(args) == 1 and not kwargs, ''

            op = func.__name__

            args_d = [ self._get_ref(i) for i in args ]
            kwargs_d = { k:self._get_ref(v) for k, v in kwargs.items() }
            res = self._client.InvokeMethod(self._ref, op, args_d, kwargs_d)

            # NOTE: Prime assumes that __ne__ is delegated to inverse of __eq__
            if res is NotImplemented:
                rop = __reflective_ops[op]

                o_d = args_d[0]

                res = self._client.InvokeMethod(o_d, rop, [self._ref])

            if res is NotImplemented:
                return res
            else:
                return Proxy(func(self, res))

        return wrapper

    def _prime_op(func):
        def wrapper(self, *args, **kwargs) -> Union[Proxy, NotImplementedType]:
            args_d = [ self._get_ref(i) for i in args ]
            kwargs_d = { k:self._get_ref(v) for k, v in kwargs.items() }

            res = self._client.InvokeMethod(self._ref, func.__name__,
                                            args_d, kwargs_d)

            if res is NotImplemented:
                return res
            else:
                return Proxy(func(self, res))

        return wrapper


    """ Special methods should be emulated """

    # NOTE: Proxy cannot emulate metaclass

    # def __new__(cls, *args, **kwargs):
    #     raise NotImplementedError()

    # def __init__(self, *args, **kwargs):
    #     raise NotImplementedError()

    # TODO: Implement later
    # def __del__(self):
    #     raise NotImplementedError()

    def __repr__(self) -> str:
        return f"'Proxy({self._ref})'"

    def __str__(self):
        return f'Proxy@{self._ref}'

    def __bytes__(self):
        raise TypeError("'Proxy' does not support bytes() conversion")

    def __format__(self, format_spec) -> str:
        return self.__str__()

    @_reflective_prime_op
    def __lt__(self, o):
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __le__(self, o):
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __eq__(self, o):
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __ne__(self, o):
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __gt__(self, o):
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __ge__(self, o):
        if isinstance(o, Exception):
            raise o
        return o

    def __hash__(self) -> int:
        return hash(self._ref)

    # TODO: Bool operation
    def __bool__(self):
        raise TypeError("'Proxy' does not support bool() conversion")

    # TODO: Attribute related operation
    # def __getattr__(self, name):
    #     raise NotImplementedError()

    # def __getattribute__(self, name):
    #     raise NotImplementedError()

    # TODO: Implement later
    # def __setattr__(self, name, value):
    #     raise NotImplementedError()

    def __delattr__(self, name):
        raise NotImplementedError()

    def __dir__(self):
        raise NotImplementedError()

    # NOTE: Proxy does not implement descriptor

    # def __get__(self, instance, owner):
    #     raise NotImplementedError()

    # def __set__(self, instance, value):
    #     raise NotImplementedError()

    # def __delete__(self, instance):
    #     raise NotImplementedError()

    # NOTE: Proxy does not implement Class

    # def __init_subclass__(cls):
    #     raise NotImplementedError()

    # def __set_name__(self, owner, name):
    #     raise NotImplementedError()

    # NOTE: Proxy does not implement instance and subclass checks

    # @classmethod
    # def __instancecheck__(cls, instance):
    #     raise NotImplementedError()

    # @classmethod
    # def __subclasscheck__(cls, subclass):
    #     raise NotImplementedError()

    # NOTE: Proxy does not emulate generic types

    # @classmethod
    # def __class_getitem__(cls, key):
    #     raise NotImplementedError()

    # TODO: Emulate callable object
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    # TODO: Emulate container types
    def __len__(self):
        raise NotImplementedError()

    def __length_hint__(self):
        raise NotImplementedError()

    def __getitem__(self, key):
        raise NotImplementedError()

    def __setitem__(self, key, value):
        raise NotImplementedError()

    def __delitem__(self, key):
        raise NotImplementedError()

    def __missing__(self, key):
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()

    def __reversed__(self):
        raise NotImplementedError()

    def __contains__(self, item):
        raise NotImplementedError()

    @_reflective_prime_op
    def __add__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __sub__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __mul__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __matmul__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __truediv__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __floordiv__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __mod__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __divmod__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __pow__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __lshift__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __rshift__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __and__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __xor__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __or__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __radd__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __rsub__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __rmul__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __rmatmul__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __rtruediv__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __rfloordiv__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __rmod__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __rdivmod__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __rpow__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __rlshift__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __rrshift__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __rand__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __rxor__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_reflective_prime_op
    def __ror__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    # NOTE: These fall back to the normal operations
    # def __iadd__(self, o):
    #     raise NotImplementedError()

    # def __isub__(self, o):
    #     raise NotImplementedError()

    # def __imul__(self, o):
    #     raise NotImplementedError()

    # def __imatmul__(self, o):
    #     raise NotImplementedError()

    # def __itruediv__(self, o):
    #     raise NotImplementedError()

    # def __ifloordiv__(self, o):
    #     raise NotImplementedError()

    # def __imod__(self, o):
    #     raise NotImplementedError()

    # def __ipow__(self, o):
    #     raise NotImplementedError()

    # def __ilshift__(self, o):
    #     raise NotImplementedError()

    # def __irshift__(self, o):
    #     raise NotImplementedError()

    # def __iand__(self, o):
    #     raise NotImplementedError()

    # def __ixor__(self, o):
    #     raise NotImplementedError()

    # def __ior__(self, o):
    #     raise NotImplementedError()

    @_prime_op
    def __neg__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_prime_op
    def __pos__(self) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_prime_op
    def __abs__(self) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_prime_op
    def __invert__(self) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    # TODO: These should return correct type
    def __complex__(self):
        raise TypeError("'Proxy' does not support complex() conversion")

    def __int__(self):
        raise TypeError("'Proxy' does not support int() conversion")

    def __float__(self):
        raise TypeError("'Proxy' does not support float() conversion")

    def __index__(self):
        raise NotImplementedError()

    @_prime_op
    def __round__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_prime_op
    def __trunc__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_prime_op
    def __floor__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_prime_op
    def __ceil__(self) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    # NOTE: Proxy does not implement context managers

    # def __enter__(self):
    #     raise NotImplementedError()

    # def __exit__(self, exc_type, exc_value, traceback):
    #     raise NotImplementedError()

    # NOTE: Proxy does not support class pattern matching

    # __match_args_ = None

    # NOTE: Proxy does not implement coroutines

    # def __await__(self):
    #     raise NotImplementedError()

    # def __aiter__(self):
    #     raise NotImplementedError()

    # def __anext__(self):
    #     raise NotImplementedError()

    # # NOTE: Proxy does not implement asynchronous context managers

    # def __aenter__(self):
    #     raise NotImplementedError()

    # def __aexit__(self, exc_type, exc_value, traceback):
    #     raise NotImplementedError()
