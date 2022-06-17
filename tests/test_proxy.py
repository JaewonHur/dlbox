#
# Copyright (c) 2022
#

import pytest
import sys
import os
import re
import signal
import time
import string
from math import nan
from random import randint, random, uniform, choice, choices
from typing import Union, Any, List, Tuple

import prime
from prime.proxy import Proxy, _client
from prime.exceptions import PrimeNotSupportedError

from tests.common import export_f_output, read_val, import_class

export_f_output(_client)

def _randbool() -> (bool, Proxy):
    r = bool(randint(0, 1))
    r_d = Proxy(_client.AllocateObj(r))

    assert read_val(_client, r_d._ref) == r

    return (r, r_d)

def _randint() -> (int, Proxy):
    r = randint(1, 100)
    r_d = Proxy(_client.AllocateObj(r))

    assert read_val(_client, r_d._ref) == r

    return (r, r_d)

def _randfloat() -> (float, Proxy):
    r = uniform(0.1, 100.0)
    r_d = Proxy(_client.AllocateObj(r))

    assert read_val(_client, r_d._ref) == r

    return (r, r_d)

def _any_number(proxy=False) -> (Any, Union[Proxy, Any]):
    x = choice([True, randint(0, 100), uniform(0.0, 100.0)])

    if random() < 0.5 and not proxy:
        return (x, x)
    else:
        return (x, Proxy(_client.AllocateObj(x)))


def _lt(a, b):       return a<b
def _le(a, b):       return a<=b
def _eq(a, b):       return a==b
def _ne(a, b):       return a!=b
def _gt(a, b):       return a>b
def _ge(a, b):       return a>=b

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

def _neg(a):         return -a
def _pos(a):         return +a
def _abs(a):         return abs(a)
def _invert(a):      return ~a

cmp_list = [_lt, _le, _eq, _ne, _gt, _ge]
complex_cmp_list = [_eq, _ne]

bool_op_list = [_and, _xor, _or]
int_op_list = [_and, _xor, _or, _lshift, _rshift]
float_op_list = [_add, _sub, _mul, _truediv, _floordiv,
                 _mod, _pow] #_divmod_a, _divmod_b,
complex_op_list = [_add, _sub, _mul, _truediv, _pow]

uop_list = [_neg, _pos, _abs, _invert]
float_complex_uop_list = [_neg, _pos, _abs]

def random_compute(n: int, comparison: bool):
    x, x_d = _randint()
    for i in range(n):
        y, y_d = _any_number()

        if isinstance(x, complex) or isinstance(y, complex):
            op = choice(complex_op_list)
        elif isinstance(x, float) or isinstance(y, float):
            op = choice(float_op_list)
        else:
            op = choice(list(set(bool_op_list +
                                        int_op_list +
                                        float_op_list)))

        (a_d, b_d, a, b) = ((x_d, y_d, x, y) if random() < 0.5
                            else (y_d, x_d, y, x))

        try:
            print(f'{op.__name__}({a}, {b})')
            x = op(a, b)

        except TimeoutError as te:
            raise te
        except Exception as e:
            with pytest.raises(Exception, match=str(e)):
                op(a_d, b_d)
            break

        x_d = op(a_d, b_d)
        x_f = read_val(_client, x_d._ref)

        if x != x and x_f != x_f:
            break

        assert x_f == x

        if comparison:
            y, y_d = ((x, Proxy(x_d._ref)) if random() < 0.5
                      else _any_number())

            if isinstance(x, complex) or isinstance(y, complex):
                op = choice(complex_cmp_list)
            else:
                op = choice(cmp_list)

            (a_d, b_d, a, b) = (x_d, y_d, x, y)

            c = op(a, b)
            c_d = op(a_d, b_d)
            c_f = read_val(_client, c_d._ref)

            assert c == c_f

def alarm_handler(signum, frame):
    raise TimeoutError()


def test_NumberTypes():

    # Bool test
    x, x_d = _randbool()

    # Type conversion should raise TypeError
    def msg(tpe):
        return re.escape(f"'Proxy' does not support {tpe.__name__}() conversion")

    with pytest.raises(PrimeNotSupportedError, match=msg(bool)):
        o = bool(x_d)
    with pytest.raises(PrimeNotSupportedError, match=msg(int)):
        o = int(x_d)
    with pytest.raises(PrimeNotSupportedError, match=msg(float)):
        o = float(x_d)
    with pytest.raises(PrimeNotSupportedError, match=msg(complex)):
        o = complex(x_d)

    # Random computation on numeric type should return the same result
    signal.signal(signal.SIGALRM, alarm_handler)
    for i in range(100):

        signal.alarm(10)
        try:
            random_compute(10, False)
        except TimeoutError:
            continue

    # Rich comparison on numeric type should return the same result
    for i in range(100):

        signal.alarm(10)
        try:
            random_compute(10, True)
        except TimeoutError:
            continue

    signal.alarm(0)

    # Unary computation on numeric type should return the same result
    for i in range(100):
        x, x_d = _any_number(True)

        if isinstance(x, float) or isinstance(x, complex):
            op = choice(float_complex_uop_list)
        else:
            op = choice(uop_list)

        x = op(x)
        x_d = op(x_d)

        assert x == read_val(_client, x_d._ref)


# def _randlist(n=None) -> (List, Proxy):
#     n = randint(0, 10) if not n else n

#     # TODO: How to export mutable variable containing Proxy
#     r = [ _any_number()[0] for i in range(n) ]
#     r_d = Proxy(_client.AllocateObj(r))

#     return r, r_d

# def _randbytesarray(n=None) -> (bytearray, Proxy):
#     n = randint(0, 10) if not n else n
#     r = bytearray([ randint(0, 255) for i in range(n) ])

#     r_d = Proxy(_client.AllocateObj(r))

#     return r, r_d

# def _randstring(n=None) -> (str, Proxy):
#     n = randint(0, 10) if not n else n

#     r = ''.join(choices(string.ascii_letters + string.digits, k=n))
#     r_d = Proxy(_client.AllocateObj(r))

#     return r, r_d

# def _randtuple(n=None) -> (Tuple, Proxy):
#     r, _ = _randlist(n)
#     r = tuple(r)

#     r_d = Proxy(_client.AllocateObj(r))

#     return r, r_d

# def _randbytes(n=None) -> (bytes, Proxy):
#     n = randint(0, 10) if not n else n
#     r = bytes([ randint(0, 255) for i in range(n) ])

#     r_d = Proxy(_client.AllocateObj(r))

#     return r, r_d

# def _any_imm_seq(n=None) -> (Any, Proxy):
#     _rand_seq = choice([_randstring, _randtuple, _randbytes])

#     return _rand_seq(n)

# def _any_mut_seq(n=None) -> (Any, Proxy):
#     _rand_seq = choice([_randlist, _randbytesarray])

#     return _rand_seq(n)

# def _any_seq(n=None) -> (Any, Proxy):
#     _rand_seq = choice([_any_imm_seq, _any_mut_seq])

#     return _rand_seq(n)


# def _contains(r, x, X, i, j, v, V, k=None):     return (x in r)
# def _not_contains(r, x, X, i, j, v, V, k=None): return (x not in r)
# def _add(r, x, X, i, j, v, V, k=None):          return (r + X)
# def _mul(r, x, X, i, j, v, V, k=None):          return (r * x)
# def _rmul(r, x, X, i, j, v, V, k=None):         return (x * r)
# def _getitem(r, x, X, i, j, v, V, k=None):      return r[i]
# def _getslice(r, x, X, i, j, v, V, k=None):
#     return (r[i:j] if not k else r[i:j:k])
# # def _len(a):         return len(a)
# # def _min(a):         return min(a)
# # def _max(a):         return max(a)
# # def _index(a):       return a.index(random())
# # def _count(a):       return a.count(random())

# def _setitem(r, x, X, i, j, v, V, k=None):      r[i] = v
# def _setslice(r, x, X, i, j, v, V, k=None):
#     if not k: r[i:j] = V
#     else:     r[i:j:k] = V
# def _delitem(r, x, X, i, j, v, V, k=None):      del r[i]
# def _delslice(r, x, X, i, j, v, V, k=None):
#     if not k: del r[i:j]
#     else:     del r[i:j:k]
# # def _append(r, x):                 r.append(x)
# # def _clear(r):                     r.clear()
# # def _copy(r):                      return r.copy()
# def _iadd(r, x, X, i, j, v, V, k=None):         r += x
# # def _extend(r, x):                 r.extend(x)
# def _imul(r, x, X, i, j, v, V, k=None):         r *= x
# # def _insert(r, i, x):              r.insert(i, x)
# # def _pop(r, i=None):
# #     if not i: return r.pop()
# #     else:     return r.pop(i)
# # def _remove(r, i):                 r.remove(i)
# def _reverse(r, x, X, i, j, v, V, k=None):      r.reverse()

# imm_op_list = [_contains, _not_contains, _add,
#                _mul, _rmul, _getitem, _getslice]
# mut_op_list = imm_op_list + [_setitem, _setslice, _delitem, _delslice,
#                              _iadd, _imul, _reverse]

# def _get_args(r):
#     x = choice((choice(r + [None]), _any_number()[0]))
#     X = choice((_any_seq(len(r))[0], _any_seq()[0]))
#     i = randint(0, int(1.2 * len(r)))
#     j = randint(0, int(1.2 * len(r)))
#     v = _any_number()[0]
#     V = _any_seq(choice((None, len(r))))[0]
#     k = choice((None, randint(0, int(1.2 * len(r)))))

#     return (x, X, i, j, v, V, k)

# def test_SeqTypes():
#     # List
#     for i in range(10):
#         r, r_d = _randlist()

#         args = _get_args(r)
#         op = choice(mut_op_list)

#         try:
#             o = op(r, *args)
#             o_d = op(r_d, *args)

#         except Exception as e:
#             print(e)
#             with pytest.raises(type(e)):
#                 op(r_d, *args)

#             continue

#         if o: assert o == read_val(_client, o_d._ref)
#         assert r == read_val(_client, r_d._ref)




################################################################################
# This test should be performed last to kill the server                        #
################################################################################

def test_KillServer():
    prime.utils.kill_server()
