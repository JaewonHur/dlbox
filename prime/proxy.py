#
# Copyright (c) 2022
#
from __future__ import annotations

import re
import sys
from typing import List, Dict, Union, Any
from types import NoneType, NotImplementedType, FunctionType
from functools import partial

import prime
from prime import utils
from prime.utils import logger
from prime.client import PrimeClient
from prime.exceptions import PrimeNotSupportedError
from prime.hasref import HasRef

from prime_pb2 import *
from prime_pb2_grpc import *

# Run PrimeServer and PrimeClient
utils.run_server()
_client = PrimeClient()

# TODO: HasRef of Prime client does not set ctx
# delattr(HasRef, '_set_ctx')
# delattr(HasRef, '_set_export')

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
+ Immutable sequences
+   String
+   Tuple
+   Bytes
+ Mutable sequences
+   List
+   Byte array
+ Sets
+   Set
+   Frozen set
+ Mappings
+   Dictionary

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

def get_path(obj: Any) -> str:
    return f'{obj.__module__}.{obj.__name__}'

def _prime_op(func):
    def wrapper(self: Proxy, *args, **kwargs) -> Union[Proxy, NotImplementedType]:
        res = self._client.InvokeMethod(self._ref, func.__name__,
                                        args, kwargs)

        if res is NotImplemented:
            return res
        elif isinstance(res, str):
            return func(self, res, *args, **kwargs)
        else: # bytes
            return res

    return wrapper


class Proxy(HasRef):
    __slots__ = ('_ref',)
    __refcnt = {}

    _client: PrimeClient = _client

    def __init__(self, ref: str):
        # TODO: Need to check referenced variable is not class definition
        super().__init__(ref)

        # TODO: Need lock?
        self.__refcnt[ref] = self.__refcnt.get(ref, 0) + 1

    def __getattribute__(self, name: str) -> Any:
        __methods = (
            '__init__',
            '__getattribute__',
            '__getattr__',
            '__repr__',
            '__str__',
            '__bytes__',
            '__format__',
            '__lt__',
            '__le__',
            '__eq__',
            '__ne__',
            '__gt__',
            '__ge__',
            '__hash__',
            '__bool__',
            '__delattr__',
            '__dir__',
            '__call__',
            '__len__',
            '__length_hint__',
            '__getitem__',
            '__setitem__',
            '__delitem__',
            '__iter__',
            '__next__',
            '__reversed__',
            '__contains__',
            '__add__',
            '__sub__',
            '__mul__',
            '__matmul__',
            '__truediv__',
            '__floordiv__',
            '__mod__',
            '__divmod__',
            '__pow__',
            '__lshift__',
            '__rshift__',
            '__and__',
            '__xor__',
            '__or__',
            '__radd__',
            '__rsub__',
            '__rmul__',
            '__rmatmul__',
            '__rtruediv__',
            '__rfloordiv__',
            '__rmod__',
            '__rdivmod__',
            '__rpow__',
            '__rlshift__',
            '__rrshift__',
            '__rand__',
            '__rxor__',
            '__ror__',
            '__iadd__',
            '__isub__',
            '__imul__',
            '__imatmul__',
            '__itruediv__',
            '__ifloordiv__',
            '__imod__',
            '__ipow__',
            '__ilshift__',
            '__irshift__',
            '__iand__',
            '__ixor__',
            '__ior__',
            '__neg__',
            '__pos__',
            '__abs__',
            '__invert__',
            '__complex__',
            '__int__',
            '__float__',
            '__index__',
            '__round__',
            '__trunc__',
            '__floor__',
            '__ceil__',
        )

        # if name in __methods:
        #     print('Direct access to special methods of Proxy is discouraged', file=sys.stderr)

        return object.__getattribute__(self, name)

    def __getattr__(self, name: str) -> Proxy:
        if name == 'mro':
            raise AttributeError

        attr_d = self._client.InvokeMethod('', get_path(getattr),
                                           [self, name])

        if isinstance(attr_d, Exception):
            raise attr_d

        return Proxy(attr_d)

    def __setattr__(self, name: str, attr: Any):
        if name == '_ref':
            object.__setattr__(self, name, attr)
        else:
            null_d = self._client.InvokeMethod('', get_path(setattr),
                                               [self, name, attr])

            if isinstance(null_d, Exception):
                raise null_d

            return Proxy(null_d)


    """ Special methods should be emulated """

    # NOTE: Proxy cannot emulate metaclass

    # def __new__(cls, *args, **kwargs):
    #     raise NotImplementedError()

    # def __init__(self, *args, **kwargs):
    #     raise NotImplementedError()

    def __del__(self):
        assert self.__refcnt[self._ref] > 0
        self.__refcnt[self._ref] -= 1

        if self.__refcnt[self._ref] == 0:
            try:
                self._client.DeleteObj(self._ref)
            except:
                pass

    def __repr__(self) -> str:
        return f"'Proxy({self._ref})'"

    def __str__(self) -> str:
        return f'Proxy@{self._ref}'

    def __bytes__(self):
        raise PrimeNotSupportedError("'Proxy' does not support bytes() conversion")

    def __format__(self, format_spec) -> str:
        return self.__str__()

    @_prime_op
    def __lt__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __le__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __eq__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __ne__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __gt__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __ge__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    def __hash__(self) -> int:
        raise PrimeNotSupportedError("'Proxy' does not support hash()")

    # TODO: Bool operation
    def __bool__(self):
        raise PrimeNotSupportedError("'Proxy' does not support bool() conversion")

    # TODO
    # def __getattribute__(self, name):
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

    @_prime_op
    def __call__(self, res: Union[Exception, str], *args, **kwargs) -> Proxy:
        if isinstance(res, Exception):
            raise res

        return Proxy(res)

    @_prime_op
    def __len__(self, res: Union[Exception, str]) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    def __length_hint__(self):
        return NotImplemented

    @_prime_op
    def __getitem__(self, res: Union[Exception, str], key) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __setitem__(self, res: Union[Exception, str], key, val) -> Proxy:
        if isinstance(res, AttributeError):
            tpe = re.match(f"^'(.*)'.*", str(res)).group(1)
            raise TypeError(f"'{tpe}' object does not support item assignment")
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __delitem__(self, res: Union[Exception, str], key) -> Proxy:
        if isinstance(res, AttributeError):
            tpe = re.match(f"^'(.*)'.*", str(res)).group(1)
            raise TypeError(f"'{tpe}' object does not support item deletion")
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    # def __missing__(self, key):
    #     raise NotImplementedError()

    @_prime_op
    def __iter__(self, res: Union[Exception, str]) -> Proxy:
        if isinstance(res, AttributeError):
            res = self._client.InvokeMethod('', get_path(iter),
                                            [self._ref])

        if isinstance(res, Exception):
            raise res

        return Proxy(res)

    # TODO: This method is added to make Proxy iterator type
    #       Is it the best way?
    @_prime_op
    def __next__(self, res: Union[Exception, str]) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __reversed__(self, res: Union[Exception, str]) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __contains__(self, res: Union[Exception, str], item) -> Proxy:
        if isinstance(res, AttributeError):
            res = self._client.InvokeMethod('', 'builtins.contains',
                                            [self, item])

        if isinstance(res, Exception):
            raise res

        return Proxy(res)

    @_prime_op
    def __add__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __sub__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __mul__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __matmul__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __truediv__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __floordiv__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __mod__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __divmod__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __pow__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __lshift__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __rshift__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __and__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __xor__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __or__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __radd__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __rsub__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __rmul__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __rmatmul__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __rtruediv__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __rfloordiv__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __rmod__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __rdivmod__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __rpow__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __rlshift__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __rrshift__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __rand__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __rxor__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __ror__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __iadd__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, AttributeError):
            return NotImplemented
        elif isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __isub__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, AttributeError):
            return NotImplemented
        elif isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __imul__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, AttributeError):
            return NotImplemented
        elif isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __imatmul__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, AttributeError):
            return NotImplemented
        elif isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __itruediv__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, AttributeError):
            return NotImplemented
        elif isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __ifloordiv__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, AttributeError):
            return NotImplemented
        elif isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __imod__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, AttributeError):
            return NotImplemented
        elif isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __ipow__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, AttributeError):
            return NotImplemented
        elif isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __ilshift__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, AttributeError):
            return NotImplemented
        elif isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __irshift__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, AttributeError):
            return NotImplemented
        elif isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __iand__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, AttributeError):
            return NotImplemented
        elif isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __ixor__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, AttributeError):
            return NotImplemented
        elif isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __ior__(self, res: Union[Exception, str], o) -> Proxy:
        if isinstance(res, AttributeError):
            return NotImplemented
        elif isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __neg__(self, res: Union[Exception, str]) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __pos__(self, res: Union[Exception, str]) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __abs__(self, res: Union[Exception, str]) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __invert__(self, res: Union[Exception, str]) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    # TODO: These should return correct type
    def __complex__(self):
        raise PrimeNotSupportedError("'Proxy' does not support complex() conversion")

    def __int__(self):
        raise PrimeNotSupportedError("'Proxy' does not support int() conversion")

    def __float__(self):
        raise PrimeNotSupportedError("'Proxy' does not support float() conversion")

    # TODO: This should be resolved in cpython
    def __index__(self):
        raise PrimeNotSupportedError("'Proxy' does not support __index__()")

    @_prime_op
    def __round__(self, res: Union[Exception, str]) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __trunc__(self, res: Union[Exception, str]) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __floor__(self, res: Union[Exception, str]) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

    @_prime_op
    def __ceil__(self, res) -> Proxy:
        if isinstance(res, Exception):
            raise res
        return Proxy(res)

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
