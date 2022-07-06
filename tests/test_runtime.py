#
# Copyright (c) 2022
#

import pytest
import sys
import os
import re
import dill
import types

import prime
from prime import PrimeClient
from prime.exceptions import PrimeError, UserError

from tests.common import export_f_output, read_val, import_class, import_prime

F_BENIGN = """
benign = 3/1

def benign():
    val = benign + 3
    return val
"""

F_ERROR = """
err = 3/0

def error():
    val = err + 3
    return val
"""

MY_CLASS = """
class MyClass():
    def __init__(self, x):
        self.x = x
    def doublex(self):
        self.x *= 2
"""

class NotInDE():
    def __init__(self, x):
        self.x = x

@import_prime
def test_ExportDef(_client: PrimeClient):

    name = 'foo'
    tpe = types.FunctionType
    source = """def foo():\n\tprint('foo')"""

    foo = _client.ExportDef(name, tpe, source)

    assert foo == 'foo'

@import_prime
def test_AllocateObj(_client: PrimeClient):

    """ Export Function to check allocated variables """
    # name = 'output'
    # tpe = types.FunctionType
    # output = _client.ExportDef(name, tpe, F_OUTPUT)
    # assert output == 'output'
    export_f_output(_client)

    """ Built-in types """
    a = 3
    new_a = _client.AllocateObj(a)
    assert isinstance(new_a, str)

    b = 'abcd'
    new_b = _client.AllocateObj(b)
    assert isinstance(new_b, str)

    c = [1,2,3]
    new_c = _client.AllocateObj(c)
    assert isinstance(new_c, str)

    assert read_val(_client, new_a) == a
    assert read_val(_client, new_b) == b
    assert read_val(_client, new_c) == c

    """ Custom types """

    # First define MyClass
    name = 'MyClass'
    import_class(name, MY_CLASS, globals())

    x = 1
    class_in_fe = MyClass.MyClass(x)
    assert _client.ExportDef(name, type, MY_CLASS) == 'MyClass'

    class_in_de = _client.AllocateObj(class_in_fe)
    assert isinstance(class_in_de, str)

    attr = _client.AllocateObj('x')
    new_x = _client.InvokeMethod('__main__', 'builtins.getattr', [class_in_de, attr], {})

    assert read_val(_client, new_x) == x


# TODO: Test FitModel
@import_prime
def test_PrimeError(_client: PrimeClient):

    # Mistmatch in function def and name should raise PrimeError
    name = 'not_output'
    tpe = types.FunctionType
    with pytest.raises(PrimeError, match="module 'not_output' has no attribute 'not_output'"):
        output = _client.ExportDef(name, tpe, F_BENIGN)

    # Allocate object which is not defined in DE should raise PrimeError
    not_in_de = NotInDE(1)
    with pytest.raises(PrimeError,
                       match="type not defined: <class 'tests.test_runtime.NotInDE'>"):
        output = _client.AllocateObj(not_in_de)

    # Invoke a method on not existing object should raise PrimeError
    with pytest.raises(PrimeError):
        output = _client.InvokeMethod('not_in_de', 'not_method', [], {})

    # Invoke an existing method on existing object with non-existing argument should
    # raise PrimeError
    name = 'MyClass'
    import_class(name, MY_CLASS, globals())
    _client.ExportDef(name, type, MY_CLASS)

    class_in_fe = MyClass.MyClass(1)
    class_in_de = _client.AllocateObj(class_in_fe)

    with pytest.raises(PrimeError, match='not_in_de'):
        output = _client.InvokeMethod(class_in_de, 'doublex', ['not_in_de'], {})

    with pytest.raises(PrimeError, match='not_in_de'):
        output = _client.InvokeMethod(class_in_de, 'doublex', [], {'not_key': 'not_in_de'})


# TODO: Test FitModel
@import_prime
def test_UserError(_client: PrimeClient):

    # Error in exported module should give UserError
    name = 'error'
    tpe = types.FunctionType
    output = _client.ExportDef(name, tpe, F_ERROR)

    with pytest.raises(ZeroDivisionError, match='division by zero'):
        raise output

    # Invoke a non-existing method on existing object should raise UserError
    name = 'MyClass'
    import_class(name, MY_CLASS, globals())
    _client.ExportDef(name, type, MY_CLASS)

    class_in_fe = MyClass.MyClass(1)
    class_in_de = _client.AllocateObj(class_in_fe)

    output = _client.InvokeMethod(class_in_de, 'triplex', [], {})
    with pytest.raises(AttributeError, match="'MyClass' object has no attribute 'triplex'"):
        raise output

    # Invoke a method on object with invalid argument should raise UserError
    new_one = _client.AllocateObj(1)

    output = _client.InvokeMethod(class_in_de, 'doublex', [new_one], {})
    with pytest.raises(TypeError,
                       match=re.escape('MyClass.doublex() takes 1 positional argument but 2 were given')):
        raise output
