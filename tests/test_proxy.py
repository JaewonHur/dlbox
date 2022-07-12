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

from tests.common import export_f_output, read_val, import_class, fullname

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


class NUM():
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

    # Additional methods on Integer
    def _bit_length(a):  return a.bit_length()
    def _bit_count(a):   return a.bit_count()
    def _to_bytes(a):    return a.to_bytes(a.bit_length() // 8 + 1, 'little')
    # def _from_bytes(a):  return int.from_bytes(a)
    # def _as_integer_ratio(a)

    # Additional methods on Float
    # def _as_integer_ratio(a)
    def _is_integer(a):  return a.is_integer()
    def _hex(a):         return a.hex()
    # def _fromhex(a):     return float.fromhex(a)

cmp_list = [NUM._lt, NUM._le, NUM._eq, NUM._ne, NUM._gt, NUM._ge]
complex_cmp_list = [NUM._eq, NUM._ne]

bool_op_list = [NUM._and, NUM._xor, NUM._or]
int_op_list = [NUM._and, NUM._xor, NUM._or, NUM._lshift, NUM._rshift]
float_op_list = [NUM._add, NUM._sub, NUM._mul, NUM._truediv, NUM._floordiv,
                 NUM._mod, NUM._pow] #NUM._divmodNUM._a, NUM._divmodNUM._b,
complex_op_list = [NUM._add, NUM._sub, NUM._mul, NUM._truediv, NUM._pow]

uop_list = [NUM._neg, NUM._pos, NUM._abs, NUM._invert]
float_complex_uop_list = [NUM._neg, NUM._pos, NUM._abs]

int_method_list = [NUM._bit_length, NUM._bit_count, NUM._to_bytes]
float_method_list = [NUM._is_integer, NUM._hex]

def random_NumberType_test(n: int, comparison: bool):
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
            random_NumberType_test(10, False)
        except TimeoutError:
            continue

    # Rich comparison on numeric type should return the same result
    for i in range(100):

        signal.alarm(10)
        try:
            random_NumberType_test(10, True)
        except TimeoutError:
            continue

    signal.alarm(0)

    # Unary computation on numeric type should return the same result
    for i in range(100):
        signal.alarm(10)
        try:

            x, x_d = _any_number(True)

            if isinstance(x, float) or isinstance(x, complex):
                op = choice(float_complex_uop_list)
            else:
                op = choice(uop_list)

                x = op(x)
                x_d = op(x_d)

            assert x == read_val(_client, x_d._ref)
        except TimeoutError:
            continue

    signal.alarm(0)

    # Additional methods on Integer
    for op in int_method_list:
        x, x_d = _randint()

        o, o_d = op(x), op(x_d)
        assert o == read_val(_client, o_d._ref)

    # Additional methods on Float
    for op in float_method_list:
        x, x_d = _randfloat()

        o, o_d = op(x), op(x_d)
        assert o == read_val(_client, o_d._ref)


def _randlist(n=None) -> (List, Proxy):
    n = randint(0, 10) if not n else n

    r = [ _any_number()[0] for i in range(n) ]
    r_d = Proxy(_client.AllocateObj(r))

    return r, r_d

def _randbytearray(n=None) -> (bytearray, Proxy):
    n = randint(0, 10) if not n else n
    r = bytearray([ randint(0, 255) for i in range(n) ])

    r_d = Proxy(_client.AllocateObj(r))

    return r, r_d

def _randstring(n=None) -> (str, Proxy):
    n = randint(0, 10) if not n else n

    r = ''.join(choices(string.ascii_letters + string.digits, k=n))
    r_d = Proxy(_client.AllocateObj(r))

    return r, r_d

def _randtuple(n=None) -> (Tuple, Proxy):
    r, _ = _randlist(n)
    r = tuple(r)

    r_d = Proxy(_client.AllocateObj(r))

    return r, r_d

def _randbytes(n=None) -> (bytes, Proxy):
    n = randint(0, 10) if not n else n
    r = bytes([ randint(0, 255) for i in range(n) ])

    r_d = Proxy(_client.AllocateObj(r))

    return r, r_d

def _randset(n=None) -> (set, Proxy):
    n = randint(0, 10) if not n else n

    r = set([ randint(0, 10) for i in range(n) ])
    r_d = Proxy(_client.AllocateObj(r))

    return r, r_d

def _randfrozenset(n=None) -> (frozenset, Proxy):
    n = randint(0, 10) if not n else n

    r = frozenset([ randint(0, 10) for i in range(n) ])
    r_d = Proxy(_client.AllocateObj(r))

    return r, r_d

def _randdict(n=None) -> (dict, Proxy):
    n = randint(0, 10) if not n else n

    r = {randint(0, 10):randint(0, 10) for i in range(n)}
    r_d = Proxy(_client.AllocateObj(r))

    return r, r_d

def _any_imm_seq(n=None) -> (Any, Proxy):
    _rand_seq = choice([_randstring, _randtuple, _randbytes])

    return _rand_seq(n)

def _any_mut_seq(n=None) -> (Any, Proxy):
    _rand_seq = choice([_randlist, _randbytearray])

    return _rand_seq(n)

def _any_seq(n=None) -> (Any, Proxy):
    _rand_seq = choice([_any_imm_seq, _any_mut_seq])

    return _rand_seq(n)


class SEQ():
    def _contains(r, x, X, i, j, v, V, k):     return (x in r)
    def _not_contains(r, x, X, i, j, v, V, k): return (x not in r)
    def _add(r, x, X, i, j, v, V, k):          return (r + X)
    def _mul(r, x, X, i, j, v, V, k):          return (r * x)
    def _rmul(r, x, X, i, j, v, V, k):         return (x * r)
    def _getitem(r, x, X, i, j, v, V, k):      return r[i]
    def _getslice(r, x, X, i, j, v, V, k):
        return (r[i:j] if not k else r[i:j:k])
    def _len(r, x, X, i, j, v, V, k):          return len(r)
    def _min(r, x, X, i, j, v, V, k):          return min(r)
    def _max(r, x, X, i, j, v, V, k):          return max(r)
    def _index(r, x, X, i, j, v, V, k):        return r.index(i)
    def _count(r, x, X, i, j, v, V, k):        return r.count(i)

    def _setitem(r, x, X, i, j, v, V, k):      r[i] = v
    def _setslice(r, x, X, i, j, v, V, k):
        if not k: r[i:j] = V
        else:     r[i:j:k] = V
    def _delitem(r, x, X, i, j, v, V, k):      del r[i]
    def _delslice(r, x, X, i, j, v, V, k):
        if not k: del r[i:j]
        else:     del r[i:j:k]
    def _append(r, x, X, i, j, v, V, k):       r.append(x)
    def _clear(r, x, X, i, j, v, V, k):        r.clear()
    def _copy(r, x, X, i, j, v, V, k):         return r.copy()
    def _iadd(r, x, X, i, j, v, V, k):         r += x
    def _extend(r, x, X, i, j, v, V, k):       r.extend(X)
    def _imul(r, x, X, i, j, v, V, k):         r *= x
    def _insert(r, x, X, i, j, v, V, k):       r.insert(i, x)
    def _pop(r, x, X, i, j, v, V, k):          r.pop(i)
    def _remove(r, x, X, i, j, v, V, k):       r.remove(i)
    def _reverse(r, x, X, i, j, v, V, k):      r.reverse()

imm_op_list = [SEQ._contains, SEQ._not_contains, SEQ._add, SEQ._mul,
               SEQ._rmul, SEQ._getitem, SEQ._getslice, SEQ._len, SEQ._min,
               SEQ._max, SEQ._index, SEQ._count]
mut_op_list = imm_op_list + [SEQ._setitem, SEQ._setslice, SEQ._delitem,
                             SEQ._delslice, SEQ._append,
                             SEQ._clear, SEQ._copy, SEQ._iadd, SEQ._extend,
                             SEQ._imul, SEQ._insert, SEQ._pop, SEQ._remove,
                             SEQ._reverse]

# Additional methods on List
# def _list():
def _sort(r):    r.sort()

list_method_list = [_sort]

# Additional methods on Tuple
# def _tuple():

# Ranges
# def _range():

# Additional methods on String
# def _str():
class STR():
    def _capitalize(r):                       return r.capitalize()
    def _casefold(r):                         return r.casefold()
    def _center(r):                           return r.center(100)
    def _count(r):                            return r.count(r[0:2])
    def _encode(r):                           return r.encode()
    def _endswith(r):                         return r.endswith(r[-1])
    def _expandtabs(r):                       return r.expandtabs()
    def _find(r):                             return r.find(r[0:2])
    def _format(r):                           return r.format()
    def _format_map(r):                       return r.format_map({})
    def _index(r):                            return r.index(r[0:2])
    def _isalnum(r):                          return r.isalnum()
    def _isalpha(r):                          return r.isalpha()
    def _isascii(r):                          return r.isascii()
    def _isdecimal(r):                        return r.isdecimal()
    def _isdigit(r):                          return r.isdigit()
    def _isidentifier(r):                     return r.isidentifier()
    def _islower(r):                          return r.islower()
    def _isnumeric(r):                        return r.isnumeric()
    def _isprintable(r):                      return r.isprintable()
    def _isspace(r):                          return r.isspace()
    def _istitle(r):                          return r.istitle()
    def _isupper(r):                          return r.isupper()
    def _join(r):                             return r.join(['a', 'b', 'c'])
    def _ljust(r):                            return r.ljust(10)
    def _lower(r):                            return r.lower()
    def _lstrip(r):                           return r.lstrip()
    def _partition(r):                        return r.partition(r[0])
    def _removeprefix(r):                     return r.removeprefix(r[0:2])
    def _removesuffix(r):                     return r.removesuffix(r[-1])
    def _replace(r):                          return r.replace(r[0], r[-1])
    def _rfind(r):                            return r.rfind(r[0:2])
    def _rindex(r):                           return r.rindex(r[0:2])
    def _rjust(r):                            return r.rjust(10)
    def _rpartition(r):                       return r.rpartition(r[0])
    def _rsplit(r):                           return r.rsplit()
    def _rstrip(r):                           return r.rstrip()
    def _split(r):                            return r.split()
    def _splitlines(r):                       return r.splitlines()
    def _startswith(r):                       return r.startswith(r[0:2])
    def _strip(r):                            return r.strip()
    def _swapcase(r):                         return r.swapcase()
    def _title(r):                            return r.title()
    def _translate(r):                        return r.translate({})
    def _upper(r):                            return r.upper()
    def _zfill(r):                            return r.zfill(5)

str_method_list = [
    STR._capitalize,  STR._casefold,  STR._center,  STR._count,  STR._encode,
    STR._endswith, STR._expandtabs,  STR._find,  STR._format,
    STR._format_map, STR._index,  STR._isalnum,  STR._isalpha, STR._isascii,
    STR._isdecimal, STR._isdigit,  STR._isidentifier,  STR._islower,
    STR._isnumeric, STR._isprintable,  STR._isspace,  STR._istitle,
    STR._isupper,  STR._join, STR._ljust,  STR._lower, STR._lstrip,
    STR._partition,  STR._removeprefix, STR._removesuffix,  STR._replace,
    STR._rfind, STR._rindex,  STR._rjust, STR._rpartition,  STR._rsplit,
    STR._rstrip,  STR._split,  STR._splitlines, STR._startswith,  STR._strip,
    STR._swapcase,  STR._title,  STR._translate, STR._upper,  STR._zfill
]

# Additional methods on bytes and bytearrays
class BYTES():
    def _count(r):                            return r.count(r[0])
    def _removeprefix(r):                     return r.removeprefix(r[0:2])
    def _removesuffix(r):                     return r.removesuffix(r[0:2])
    # def _decode(r):                           return r.decode()
    def _endswith(r):                         return r.endswith(r[-2:])
    def _find(r):                             return r.find(r[0:2])
    def _index(r):                            return r.index(r[0:2])
    def _join(r):                             return r.join([b'\x00', b'\x01'])
    def _partition(r):                        return r.partition(r[0:1])
    def _replace(r):                          return r.replace(r[0:2], r[-2:])
    def _rfind(r):                            return r.rfind(r[0:2])
    def _rindex(r):                           return r.rindex(r[0:2])
    def _rpartition(r):                       return r.rpartition(r[0:1])
    def _startswith(r):                       return r.startswith(r[0:2])
    def _translate(r):                        return r.translate(None)
    def _center(r):                           return r.center(100)
    def _ljust(r):                            return r.ljust(10)
    def _lstrip(r):                           return r.lstrip()
    def _rjust(r):                            return r.rjust(10)
    def _rsplit(r):                           return r.rsplit()
    def _rstrip(r):                           return r.rstrip()
    def _split(r):                            return r.split()
    def _strip(r):                            return r.strip()
    def _capitalize(r):                       return r.capitalize()
    def _expandtabs(r):                       return r.expandtabs()
    def _isalnum(r):                          return r.isalnum()
    def _isalpha(r):                          return r.isalpha()
    def _isascii(r):                          return r.isascii()
    def _isdigit(r):                          return r.isdigit()
    def _islower(r):                          return r.islower()
    def _isspace(r):                          return r.isspace()
    def _istitle(r):                          return r.istitle()
    def _isupper(r):                          return r.isupper()
    def _lower(r):                            return r.lower()
    def _splitlines(r):                       return r.splitlines()
    def _swapcase(r):                         return r.swapcase()
    def _title(r):                            return r.title()
    def _upper(r):                            return r.upper()
    def _zfill(r):                            return r.zfill(10)

bytes_method_list = [
    BYTES._count, BYTES._removeprefix, BYTES._removesuffix, BYTES._endswith,
    BYTES._find, BYTES._index, BYTES._join, BYTES._partition, BYTES._replace,
    BYTES._rfind, BYTES._rindex, BYTES._rpartition, BYTES._startswith,
    BYTES._translate, BYTES._center, BYTES._ljust, BYTES._lstrip, BYTES._rjust,
    BYTES._rsplit, BYTES._rstrip, BYTES._split, BYTES._strip, BYTES._capitalize,
    BYTES._expandtabs, BYTES._isalnum, BYTES._isalpha, BYTES._isascii,
    BYTES._isdigit, BYTES._islower, BYTES._isspace, BYTES._istitle,
    BYTES._isupper, BYTES._lower, BYTES._splitlines, BYTES._swapcase,
    BYTES._title, BYTES._upper, BYTES._zfill ] # BYTES._decode

bytearrays_method_list = bytes_method_list

def _get_args(r):
    x = choice(r) if len(r) > 0 else None
    x = choice((x, _any_number()[0]))
    X = choice((_any_seq(len(r))[0], _any_seq()[0]))
    i = randint(0, int(1.2 * len(r)))
    j = randint(0, int(1.2 * len(r)))
    v = _any_number()[0]
    V = _any_seq(choice((None, len(r))))[0]
    k = choice((None, randint(0, int(1.2 * len(r)))))

    return (x, X, i, j, v, V, k)

def random_SeqType_test(n: int, _randseq: callable):
    if _randseq in (_randstring, _randtuple, _randbytes):
        mutable = False
    elif _randseq in (_randlist, _randbytearray):
        mutable = True

    for i in range(n):
        r, r_d = _randseq()

        args = _get_args(r)
        if mutable: op = choice(mut_op_list)
        else:
            if random() < 0.2: op = choice(mut_op_list)
            else: op = choice(imm_op_list)

        try:
            o = op(r, *args)

        except Exception as e:
            with pytest.raises(type(e)):
                op(r_d, *args)
            continue

        try:
            o_d = op(r_d, *args)
        except PrimeNotSupportedError as pe:
            assert op in (SEQ._contains, SEQ._not_contains, SEQ._len, SEQ._min,
                          SEQ._max)
            continue

        if o: assert o == read_val(_client, o_d._ref)

        assert r == read_val(_client, r_d._ref)


def test_SeqTypes():
    # String
    random_SeqType_test(100, _randstring)

    # Bytes
    random_SeqType_test(100, _randbytes)

    # Tuple
    random_SeqType_test(100, _randtuple)

    # List
    random_SeqType_test(100, _randlist)

    # ByteArray
    random_SeqType_test(100, _randbytearray)

    # Additional methods on String
    for op in list_method_list:
        r, r_d = _randlist(10)

        o, o_d = op(r), op(r_d)
        assert r == read_val(_client, r_d._ref)

    # Additional methods on String
    for op in str_method_list:
        r, r_d = _randstring(10)

        o, o_d = op(r), op(r_d)
        assert o == read_val(_client, o_d._ref)

    # Additional methods on Bytes
    for op in bytes_method_list:
        r, r_d = _randbytes(10)

        o, o_d = op(r), op(r_d)
        assert o == read_val(_client, o_d._ref)

    # Additional methods on ByteArrays
    for op in bytearrays_method_list:
        r, r_d = _randbytes(10)

        o, o_d = op(r), op(r_d)
        assert o == read_val(_client, o_d._ref)


class SET():
    def _len(r, x, X):                           return len(r)
    def _contains(r, x, X):                      return (x in r)
    def _not_contains(r, x, X):                  return (x not in r)
    def _lessoreq(r, x, X):                      return (r <= X)
    def _less(r, x, X):                          return (r < X)
    def _issubset(r, x, X):                      return r.issubset(X)
    def _largeroreq(r, x, X):                    return (r >= X)
    def _larger(r, x, X):                        return (r > X)
    def _union(r, x, X):                         return r.union(X)
    def _union_r(r, x, X):                       return (r | X)
    def _intersection(r, x, X):                  return r.intersection(X)
    def _intersection_r(r, x, X):                return (r & X)
    def _difference(r, x, X):                    return r.difference(X)
    def _difference_r(r, x, X):                  return (r - X)
    def _symmetric_difference(r, x, X):          return r.symmetric_difference(X)
    def _symmetric_difference_r(r, x, X):        return (r ^ X)
    def _copy(r, x, X):                          return r.copy()

    def _update(r, x, X):                        r.update(X)
    def _update_r(r, x, X):                      r |= X
    def _intersection_update(r, x, X):           r.intersection_update(X)
    def _intersection_update_r(r, x, X):         r &= X
    def _difference_update(r, x, X):             r.difference_update(X)
    def _difference_update_r(r, x, X):           r -= X
    def _symmetric_difference_update(r, x, X):   r.symmetric_difference_update(X)
    def _symmetric_difference_update_r(r, x, X): r ^= X
    def _add(r, x, X):                           r.add(x)
    def _remove(r, x, X):                        r.remove(x)
    def _discard(r, x, X):                       r.discard(x)
    def _pop(r, x, X):                           return r.pop()
    def _clear(r, x, X):                         r.clear()

set_op_list = [
    SET._len, SET._contains, SET._not_contains, SET._lessoreq, SET._less,
    SET._issubset, SET._largeroreq, SET._larger, SET._union, SET._union_r,
    SET._intersection, SET._intersection_r, SET._difference, SET._difference_r,
    SET._symmetric_difference, SET._symmetric_difference_r, SET._copy,
    SET._update, SET._update_r, SET._intersection_update,
    SET._intersection_update_r, SET._difference_update,
    SET._difference_update_r, SET._symmetric_difference_update,
    SET._symmetric_difference_update_r, SET._add, SET._remove, SET._discard,
    SET._pop, SET._clear
]

def random_SetType_test(n: int, _randsettype: callable):
    for i in range(n):
        r, r_d = _randsettype()

        x = randint(0, 10)
        x_d = Proxy(_client.AllocateObj(x))

        X, X_d = _randsettype()

        op = choice(set_op_list)

        try:
            o = op(r, x, X)
        except Exception as e:
            with pytest.raises(type(e)):
                op(r_d, x_d, X_d)
            continue

        try:
            o_d = op(r_d, x_d, X_d)
        except PrimeNotSupportedError as pe:
            assert op in (SET._contains, SET._not_contains, SET._len)
            continue

        if o: assert o == read_val(_client, o_d._ref)

        assert r == read_val(_client, r_d._ref)

def test_SetTypes():
    # Proxy objects cannot be an element of SetType
    a, a_d = _randint()
    b, b_d = _randint()

    with pytest.raises(PrimeNotSupportedError):
        c_d = set([a_d, b_d])

    with pytest.raises(PrimeNotSupportedError):
        c_d = frozenset([a_d, b_d])

    # Set
    random_SetType_test(100, _randset)

    # FrozenSet
    random_SetType_test(100, _randfrozenset)


class MAP():
    def _list(r, x, X):                      return list(r)
    def _len(r, x, X):                       return len(r)
    def _getitem(r, x, X):                   return r[x]
    def _setitem(r, x, X):                   r[x] = x
    def _delitem(r, x, X):                   del r[x]
    def _contains(r, x, X):                  return (x in r)
    def _not_contains(r, x, X):              return (x not in r)
    # def _iter(r, x, X):                      return iter(r)
    def _clear(r, x, X):                     r.clear()
    def _copy(r, x, X):                      return r.copy()
    def _fromkeys(r, x, X):                  return dict.fromkeys([])
    def _get(r, x, X):                       return r.get(x)
    # def _items(r, x, X):                     return r.items()
    # def _keys(r, x, X):                      return r.keys()
    def _pop(r, x, X):                       return r.pop(k)
    def _popitem(r, x, X):                   return popitem()
    # def _reversed(r, x, X):                  return reversed(r)
    def _setdefault(r, x, X):                return setdefault(x)
    def _update(r, x, X):                    return r.update(X)
    # def _values(r, x, X):                    return r.values()
    def _or(r, x, X):                        return (r | X)
    def _update_r(r, x, X):                  r |= X

map_op_list = [
    MAP._list, MAP._len, MAP._getitem, MAP._setitem, MAP._delitem,
    MAP._contains, MAP._not_contains, MAP._clear, MAP._copy,
    MAP._fromkeys, MAP._get, MAP._pop, MAP._popitem, MAP._setdefault,
    MAP._update, MAP._or, MAP._update_r
] # _iter, _reversed, _items, _keys, _values

def random_MapType_test(n: int, _randmap):
    for i in range(n):
        r, r_d = _randmap()

        x = randint(0, 10)
        x_d = Proxy(_client.AllocateObj(x))

        X, X_d = _randmap()

        op = choice(map_op_list)

        try:
            o = op(r, x, X)
        except Exception as e:
            with pytest.raises(type(e)):
                op(r_d, x_d, X_d)
            continue

        try:
            o_d = op(r_d, x_d, X_d)
        except PrimeNotSupportedError as pe:
            assert op in (MAP._contains, MAP._not_contains, MAP._len, MAP._list)
            continue

        print(f'{op}({r}, {r_d})')
        if o: assert read_val(_client, (o == o_d)._ref)

        assert r == read_val(_client, r_d._ref)


def test_MapTypes():
    # Proxy objects cannot be a key of MapType
    a, a_d = _randint()
    b, b_d = _randint()

    with pytest.raises(PrimeNotSupportedError):
        c_d = dict({a_d: 'a_d', b_d: 'b_d'})

    # Dict
    random_MapType_test(100, _randdict)


MY_CLASS = """
class MyClass():
    def __init__(self, x):
        self.x = x
    def __eq__(self, o):
        return (self.x == o.x)
"""

def test_Containers():
    # List is container type
    r, r_d = _randlist(10)
    x, x_d = _any_number(True)

    for (a, b) in ((x, x), (x, x_d), (x_d, x), (x_d, x_d)):
        i = randint(0, 9)
        r[i] = a
        r_d[i] = b

        assert read_val(_client, (r == r_d)._ref)

    # Tuple is container type
    r, r_d = _randtuple(10)
    x, x_d = _any_number(True)

    for (a, b) in ((x, x), (x, x_d), (x_d, x), (x_d, x_d)):
        r = r + (a,)
        r_d = r_d + (b,)

        assert read_val(_client, (r == r_d)._ref)

    # Custom class can be container type
    name = 'MyClass'
    import_class(name, MY_CLASS, globals())
    _client.ExportDef(fullname(MyClass.MyClass), type, MY_CLASS)

    o = MyClass.MyClass(3)
    o_d_partial = MyClass.MyClass(Proxy(_client.AllocateObj(3)))
    o_d = Proxy(_client.AllocateObj(o))

    assert read_val(_client, (o == o_d_partial)._ref)
    assert read_val(_client, (o == o_d)._ref)
    assert read_val(_client, (o_d == o_d_partial)._ref)

################################################################################
# This test should be performed last to kill the server                        #
################################################################################

def test_KillServer():
    prime.utils.kill_server()
