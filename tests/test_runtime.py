#
# Copyright (c) 2022
#

import pytest
import sys
import os
import re
import dill
import types

import torch

import prime
from prime import PrimeClient
from prime.proxy import Proxy, _client
from prime.exceptions import PrimeError, UserError

from tests.common import *

samples_d = Proxy('_SAMPLES')
labels_d = Proxy('_LABELS')

F_BENIGN = """
v = 3/1

def benign():
    val = v + 3
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

################################################################################
# Init server before starting tests                                            #
################################################################################

def test_initServer():
    time.sleep(1)
    if not _client.check_server():
        raise Exception('Server not running')

    export_f_output(_client)


def test_ExportDef():
    name = 'foo'
    tpe = types.FunctionType
    source = """def foo():\n\tprint('foo')"""

    foo = _client.ExportDef('__main__.foo', tpe, source)

    assert foo == 'foo'

def test_InvokeMethod():
    export_f_output(_client)

    samples = torch.Tensor(range(6 * 10)).reshape(10, 2, 3)
    labels = torch.Tensor([0, 1] * 5)

    assert torch.equal(read_val(_client, samples_d), samples)
    assert torch.equal(read_val(_client, labels_d), labels)

def test_PrimeError():

    # Mistmatch in function def and name should raise PrimeError
    name = 'not_output'
    tpe = types.FunctionType
    with pytest.raises(PrimeError, match="'not_output'"):
        output = _client.ExportDef('__main__.not_output', tpe, F_BENIGN)

    # Invoke a method with an object whose prototype is not in DE should raise PrimeError
    not_in_de = NotInDE(1)
    with pytest.raises(PrimeError,
                       match="type not trusted: <class 'tests.test_runtime.NotInDE'>"):
        output = _client.InvokeMethod('__main__', 'builtins.str', [not_in_de])

    # Invoke a method on not existing object should raise PrimeError
    with pytest.raises(PrimeError):
        output = _client.InvokeMethod('not_in_de', 'not_method', [], {})

    # Invoke a non-existing __main__ method should raise PrimeError
    with pytest.raises(PrimeError,
                       match="not_existing is not from trusted packages"):
        output = _client.InvokeMethod('__main__', 'not_existing')

def test_UserError():

    # Error in exported module should give UserError
    name = 'error'
    tpe = types.FunctionType
    output = _client.ExportDef('__main__.error', tpe, F_ERROR)

    with pytest.raises(ZeroDivisionError, match='division by zero'):
        raise output

    # Invoke a non-existing method on existing object should raise UserError
    output = _client.InvokeMethod('_SAMPLES', 'not_existing')
    with pytest.raises(AttributeError, match="'Tensor' object has no attribute 'not_existing'"):
        raise output

    # Invoke a method on object with invalid argument should raise UserError
    output = _client.InvokeMethod('_SAMPLES', 'reshape')
    with pytest.raises(TypeError,
                       match=re.escape('reshape() missing 1 required positional arguments: "shape"')):
        raise output

################################################################################
# Kill server after all tests are completed                                    #
################################################################################

def test_KillServer():
    prime.utils.kill_server()
