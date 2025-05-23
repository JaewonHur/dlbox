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


def test_ExportDef():
    name = 'foo'
    tpe = types.FunctionType
    source = """def foo():\n\tprint('foo')"""

    foo = _client.ExportDef('__main__.foo', tpe, source)

    assert foo == 'foo'

def test_AllocateObj():
    # cannot allocate untrusted type
    class bar(): pass
    bar.__module__ = 'bar'

    with pytest.raises(PrimeError,
                       match=re.compile("type not trusted: .*")):
        output = _client.AllocateObj(bar())

    # cannot allocate non-callable type
    with pytest.raises(PrimeError,
                       match='cannot allocate non-callable: 3'):
        output = _client.AllocateObj(3)

    # TODO: can allocate callable, trusted instance
    # import torchvision
    # ref = Proxy(_client.AllocateObj(torchvision.transforms.ToTensor()))
    # assert isinstance(read_val(_client, ref), torchvision.transforms.ToTensor)


@with_args
def test_InvokeMethod(samples_d, labels_d, samples, labels):

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
                       match=re.compile("type not trusted:.*")):
        output = _client.InvokeMethod('', 'builtins.str', [not_in_de])

    # Invoke a method on not existing object should raise PrimeError
    with pytest.raises(PrimeError):
        output = _client.InvokeMethod('not_in_de', 'not_method', [], {})

    # Invoke a non-existing __main__ method should raise PrimeError
    with pytest.raises(PrimeError,
                       match="not_existing is not from trusted packages"):
        output = _client.InvokeMethod('', 'not_existing')

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

@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    def kill_server():
        try: prime.utils.kill_server()
        except: pass
    request.addfinalizer(kill_server)
