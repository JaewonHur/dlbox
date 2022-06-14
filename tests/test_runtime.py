#
# Copyright (c) 2022
#

import pytest
import sys
import os
import dill
import time
import types
import datetime
import importlib.util

import prime
from prime import PrimeClient
from prime.exceptions import PrimeError, UserError

F_OUTPUT = """
def output(x):
    import dill
    with open('/tmp/x.txt', 'wb') as fd:
        dill.dump(x, fd)
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

def export_class(name: str, src: str):
    module = f'/tmp/_{name}.py'
    with open(module, 'w') as fd:
        fd.write(src)

    spec = importlib.util.spec_from_file_location(name, module)
    module = importlib.util.module_from_spec(spec)

    sys.modules[name] = module
    globals()[name] = module
    spec.loader.exec_module(module)

class NotInDE():
    def __init__(self, x):
        self.x = x


logdir = f'{os.getcwd()}/test-logs'
os.makedirs(logdir, exist_ok=True)
now = datetime.datetime.now()

def import_prime(func):
    def wrapper():
        prime.utils.log_to_file(logdir + '/' + now.strftime('%Y-%m-%d-%X') + '_'
                                + func.__name__ + '.txt')

        prime.utils.run_server()

        time.sleep(1)
        _client = PrimeClient()

        func(_client)
        prime.utils.kill_server()

    return wrapper


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
    name = 'output'
    tpe = types.FunctionType
    output = _client.ExportDef(name, tpe, F_OUTPUT)
    assert output == 'output'

    def read(x):
        _client.InvokeMethod('__main__', 'output', [x], {})
        with open('/tmp/x.txt', 'rb') as fd:
            x = dill.load(fd)

        return x

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

    assert read(new_a) == a
    assert read(new_b) == b
    assert read(new_c) == c

    """ Custom types """

    # First define MyClass
    name = 'MyClass'
    export_class(name, MY_CLASS)

    x = 1
    class_in_fe = MyClass.MyClass(x)
    assert _client.ExportDef(name, type, MY_CLASS) == 'MyClass'

    class_in_de = _client.AllocateObj(class_in_fe)
    assert isinstance(class_in_de, str)

    attr = _client.AllocateObj('x')
    new_x = _client.InvokeMethod('__main__', 'getattr', [class_in_de, attr], {})

    assert read(new_x) == x


# TODO: Test FitModel
@import_prime
def test_PrimeError(_client: PrimeClient):

    # Mistmatch in function def and name should raise PrimeError
    name = 'not_output'
    tpe = types.FunctionType
    with pytest.raises(PrimeError, match="module 'not_output' has no attribute 'not_output'"):
        output = _client.ExportDef(name, tpe, F_OUTPUT)

    # Allocate object which is not defined in DE should raise PrimeError
    not_in_de = NotInDE(1)
    with pytest.raises(PrimeError,
                       match="type not defined: <class 'tests.test_runtime.NotInDE'>"):
        output = _client.AllocateObj(not_in_de)

    # Invoke a method on not existing object should raise PrimeError
    with pytest.raises(PrimeError):
        output = _client.InvokeMethod('not_in_de', 'not_method', [], {})

    # Invoke an existing method on existing object with invalid argument should
    # raise PrimeError
    name = 'MyClass'
    export_class(name, MY_CLASS)
    _client.ExportDef(name, type, MY_CLASS)

    class_in_fe = MyClass.MyClass(1)
    class_in_de = _client.AllocateObj(class_in_fe)

    with pytest.raises(PrimeError, match='not_in_de'):
        output = _client.InvokeMethod(class_in_de, 'doublex', ['not_in_de'], {})

    with pytest.raises(PrimeError, match='not_in_de'):
        output = _client.InvokeMethod(class_in_de, 'doublex', [], {'not_key': 'not_in_de'})

