#
# Copyright (c) 2022
#
from typing import List, Dict

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

  Numbers
    int
    bool
    float
    complex
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
        raise NotImplementedError()

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

    # __slots__ = None

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
    def __add__(self, o):
        raise NotImplementedError()

    def __sub__(self, o):
        raise NotImplementedError()

    def __mul__(self, o):
        raise NotImplementedError()

    def __matmul__(self, o):
        raise NotImplementedError()

    def __truediv__(self, o):
        raise NotImplementedError()

    def __floordiv__(self, o):
        raise NotImplementedError()

    def __mod__(self, o):
        raise NotImplementedError()

    def __divmod__(self, o):
        raise NotImplementedError()

    # TODO: check
    def __pow__(self, o, modulo=None):
        raise NotImplementedError()

    def __lshift__(self, o):
        raise NotImplementedError()

    def __rshift__(self, o):
        raise NotImplementedError()

    def __and__(self, o):
        raise NotImplementedError()

    def __xor__(self, o):
        raise NotImplementedError()

    def __or__(self, o):
        raise NotImplementedError()

    def __radd__(self, o):
        raise NotImplementedError()

    def __rsub__(self, o):
        raise NotImplementedError()

    def __rmul__(self, o):
        raise NotImplementedError()

    def __rmatmul__(self, o):
        raise NotImplementedError()

    def __rtruediv__(self, o):
        raise NotImplementedError()

    def __rfloordiv__(self, o):
        raise NotImplementedError()

    def __rmod__(self, o):
        raise NotImplementedError()

    def __rdivmod__(self, o):
        raise NotImplementedError()

    def __rpow__(self, o):
        raise NotImplementedError()

    def __rlshift__(self, o):
        raise NotImplementedError()

    def __rrshift__(self, o):
        raise NotImplementedError()

    def __rand__(self, o):
        raise NotImplementedError()

    def __rxor__(self, o):
        raise NotImplementedError()

    def __ror__(self, o):
        raise NotImplementedError()

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

    def __neg__(self):
        raise NotImplementedError()

    def __pos__(self):
        raise NotImplementedError()

    def __abs__(self):
        raise NotImplementedError()

    def __invert__(self):
        raise NotImplementedError()

    # TODO: These should return correct type
    def __complex__(self):
        raise NotImplementedError()

    def __int__(self):
        raise NotImplementedError()

    def __float__(self):
        raise NotImplementedError()

    def __index__(self):
        raise NotImplementedError()

    def __round__(self, ndigits=None):
        raise NotImplementedError()

    def __trunc__(self):
        raise NotImplementedError()

    def __floor__(self):
        raise NotImplementedError()

    def __ceil__(self):
        raise NotImplementedError()

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
