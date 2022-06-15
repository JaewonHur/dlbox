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

    # _to_server directly invokes the same function on prime server, and returns the result.
    def _to_server(func):
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

    def __del__(self):
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

    def __bytes__(self):
        raise NotImplementedError()

    def __format__(self, format_spec):
        raise NotImplementedError()

    # TODO: rich comparison methods
    def __lt__(self, o):
        raise NotImplementedError()

    def __le__(self, o):
        raise NotImplementedError()

    def __eq__(self, o):
        raise NotImplementedError()

    def __ne__(self, o):
        raise NotImplementedError()

    def __gt__(self, o):
        raise NotImplementedError()

    def __ge__(self, o):
        raise NotImplementedError()

    # TODO: Hash operation
    def __hash__(self):
        raise NotImplementedError()

    # TODO: Bool operation
    def __bool__(self):
        raise TypeError("'Proxy' does not support bool() conversion")

    # TODO: Attribute related operation
    # def __getattr__(self, name):
    #     raise NotImplementedError()

    # def __getattribute__(self, name):
    #     raise NotImplementedError()

    def __setattr__(self, name, value):
        raise NotImplementedError()

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

    # TODO: Emulate numeric type
    @_to_server
    def __add__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __sub__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __mul__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __matmul__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __truediv__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __floordiv__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __mod__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __divmod__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __pow__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __lshift__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __rshift__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __and__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __xor__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __or__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __radd__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __rsub__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __rmul__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __rmatmul__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __rtruediv__(self, o) -> str:
        if isinstance(o, Exception):
            raise o

        return o

    @_to_server
    def __rfloordiv__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __rmod__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __rdivmod__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __rpow__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __rlshift__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __rrshift__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __rand__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __rxor__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
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

    @_to_server
    def __neg__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __pos__(self) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __abs__(self) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
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

    @_to_server
    def __round__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __trunc__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
    def __floor__(self, o) -> str:
        if isinstance(o, Exception):
            raise o
        return o

    @_to_server
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
