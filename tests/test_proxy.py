#
# Copyright (c) 2022
#

import pytest
import sys
import os
import re
import random
import time
from math import nan
from typing import Union, Any

import prime
from prime.proxy import Proxy, _client

from tests.common import export_f_output, read_val, import_class

export_f_output(_client)

def _randbool() -> (bool, Proxy):
    r = bool(random.randint(0, 1))
    r_d = Proxy(_client.AllocateObj(r))

    assert read_val(_client, r_d._ref) == r

    return (r, r_d)

def _randint() -> (int, Proxy):
    r = random.randint(1, 100)
    r_d = Proxy(_client.AllocateObj(r))

    assert read_val(_client, r_d._ref) == r

    return (r, r_d)

def _randfloat() -> (float, Proxy):
    r = random.uniform(0.1, 100.0)
    r_d = Proxy(_client.AllocateObj(r))

    assert read_val(_client, r_d._ref) == r

    return (r, r_d)

def _any() -> (Any, Union[Proxy, Any]):
    x = random.choice([True,
                       random.randint(0, 100),
                       random.uniform(0.0, 100.0)])

    if random.random() < 0.5:
        return (x, x)
    else:
        return (x, Proxy(_client.AllocateObj(x)))


def _add(a, b):      return a+b
def _sub(a, b):      return a-b
def _mul(a, b):      return a*b
def _matmul(a, b):   return a@b
def _truediv(a, b):  return a/b
def _floordiv(a, b): return a//b
def _mod(a, b):      return a%b
# def _divmod_a(a, b): return divmod(a, b)[0]
# def _divmod_b(a, b): return divmod(a, b)[1]
def _pow(a, b):      return a**b
def _lshift(a, b):   return a<<b
def _rshift(a, b):   return a>>b
def _and(a, b):      return a&b
def _xor(a, b):      return a^b
def _or(a, b):       return a|b

bool_op_list = [_and, _xor, _or]
int_op_list = [_and, _xor, _or, _lshift, _rshift]
float_op_list = [_add, _sub, _mul, _truediv, _floordiv,
                 _mod, _pow] #_divmod_a, _divmod_b,
complex_op_list = [_add, _sub, _mul, _truediv, _pow]

def random_compute(n: int):
    x, x_d = _randint()
    for i in range(n):
        y, y_d = _any()

        if isinstance(x, complex) or isinstance(y, complex):
            op = random.choice(complex_op_list)
        elif isinstance(x, float) or isinstance(y, float):
            op = random.choice(float_op_list)
        else:
            op = random.choice(list(set(bool_op_list +
                                        int_op_list +
                                        float_op_list)))

        (a_d, b_d, a, b) = ((x_d, y_d, x, y) if random.random() < 0.5
                            else (y_d, x_d, y, x))

        try:
            x = op(a, b)
            x_d = op(a_d, b_d)
            x_f = read_val(_client, x_d._ref)

            if x is nan and x_f is nan:
                break

            assert x_f == x

        except ZeroDivisionError as ze:
            with pytest.raises(ZeroDivisionError, match=str(ze)):
                op(a_d, b_d)

            break
        except OverflowError as oe:
            with pytest.raises(OverflowError, match=str(oe)):
                op(a_d, b_d)

            break
        except TypeError as te:
            with pytest.raises(TypeError, match=str(te)):
                op(a_d, b_d)

            break
        except ValueError as ve:
            with pytest.raises(ValueError, match=str(ve)):
                op(a_d, b_d)

            break
        except MemoryError as me:
            with pytest.raises(MemoryError, match=str(me)):
                op(a_d, b_d)

            break


def test_NumberTypes():

    # Bool test
    x, x_d = _randbool()

    # Type conversion should raise TypeError
    def msg(tpe):
        return re.escape(f"'Proxy' does not support {tpe.__name__}() conversion")

    with pytest.raises(TypeError, match=msg(bool)):
        o = bool(x_d)
    with pytest.raises(TypeError, match=msg(int)):
        o = int(x_d)
    with pytest.raises(TypeError, match=msg(float)):
        o = float(x_d)
    with pytest.raises(TypeError, match=msg(complex)):
        o = complex(x_d)

    # Random computation on numeric type should return the same result
    for i in range(100):
        random_compute(10)


################################################################################
# This test should be performed last to kill the server                        #
################################################################################

def test_KillServer():
    prime.utils.kill_server()
