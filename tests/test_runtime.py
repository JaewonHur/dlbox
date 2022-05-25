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

F_OUTPUT = """
def output(x):
    import dill
    with open('/tmp/x.txt', 'wb') as fd:
        dill.dump(x, fd)
"""

MY_CLASS = """
class MyClass():
    def __init__(self, x):
        self.x = x
"""

logdir = f'{os.getcwd()}/test-logs'
os.makedirs(logdir, exist_ok=True)
now = datetime.datetime.now()

def import_prime(func):
    def wrapper():
        prime.utils.log_to_file(logdir + '/' + now.strftime('%Y-%m-%d-%X') + '_'
                                + func.__name__ + '.txt')

        prime.utils.run_server()
        time.sleep(1)
        func()
        prime.utils.kill_server()

    return wrapper

@import_prime
def test_ExportDef():
    fe = prime.PrimeClient()

    name = 'foo'
    tpe = types.FunctionType
    source = """def foo():\n\tprint('foo')"""

    foo = fe.ExportDef(name, tpe, source)

    assert foo == 'foo'

@import_prime
def test_AllocateObj():
    fe = prime.PrimeClient()

    """ Export Function to check allocated variables """
    name = 'output'
    tpe = types.FunctionType
    output = fe.ExportDef(name, tpe, F_OUTPUT)
    assert output == 'output'

    def read(x):
        fe.InvokeMethod('__main__', 'output', [x], {})
        with open('/tmp/x.txt', 'rb') as fd:
            x = dill.load(fd)

        return x

    """ Built-in types """
    a = 3
    assert fe.AllocateObj('a', type(a), a) == 'a'

    b = 'abcd'
    assert fe.AllocateObj('b', type(b), b) == 'b'

    c = [1,2,3]
    assert fe.AllocateObj('c', type(c), c) == 'c'

    assert read('a') == a
    assert read('b') == b
    assert read('c') == c

    """ Custom types """

    # First define MyClass
    name = 'MyClass'
    module = f'/tmp/_MyClass.py'
    with open(module, 'w') as fd:
        fd.write(MY_CLASS)

    spec = importlib.util.spec_from_file_location(name, module)
    module = importlib.util.module_from_spec(spec)

    sys.modules[name] = module
    globals()[name] = module
    spec.loader.exec_module(module)

    x = 1
    myclass = MyClass.MyClass(x)
    assert fe.ExportDef(name, type, MY_CLASS) == 'MyClass'
    assert fe.AllocateObj('myclass', MyClass.MyClass, myclass) == 'myclass'

    attr = fe.AllocateObj('attr', str, 'x')
    new_x = fe.InvokeMethod('__main__', 'getattr', ['myclass', attr], {})

    assert read(new_x) == x
