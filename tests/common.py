#
# Copyright (c) 2022
#

import sys
import os
import dill
import importlib.util
import datetime
import time
import types
from torch import Tensor

from typing import Any, Union
from types import FunctionType

import prime
from prime.proxy import _client, Proxy
from prime import PrimeClient

logdir = f'{os.getcwd()}/test-logs'
os.makedirs(logdir, exist_ok=True)
now = datetime.datetime.now()


F_OUTPUT = """
def output(x):
    import dill
    with open('/tmp/x.txt', 'wb') as fd:
        dill.dump(x, fd)
"""

def export_f_output(_client: PrimeClient):
    output = _client.ExportDef('__main__.output', types.FunctionType, F_OUTPUT)
    assert output == 'output'

def read_val(_client: PrimeClient, x: Proxy) -> Any:
    o = _client.InvokeMethod('', '__main__.output', [x])
    with open('/tmp/x.txt', 'rb') as fd:
        x = dill.load(fd)
    _client.DeleteObj(o)

    return x

def read_tag(_client: PrimeClient, x: Proxy) -> str:
    o = _client.InvokeMethod('', 'get_tag', [x])
    with open('/tmp/x.txt', 'r') as fd:
        x = fd.read()

    return x

def fullname(cls: type) -> str:
    return f'{cls.__module__}.{cls.__name__}'

def import_class(name: str, src: str, __global: dict):
    module = f'/tmp/_{name}.py'
    with open(module, 'w') as fd:
        fd.write(src)

    spec = importlib.util.spec_from_file_location(name, module)
    module = importlib.util.module_from_spec(spec)

    sys.modules[name] = module
    __global[name] = module
    spec.loader.exec_module(module)

def sample_init() -> (Tensor, Tensor):
    samples = Tensor(range(6 * 10)).reshape(10, 2, 3)
    labels = Tensor([0, 1] * 5)

    return samples, labels

def reset_server():

    port = os.environ.get('PRIMEPORT', None)

    prime.utils.kill_server()
    prime.utils.run_server(port)

    global samples, labels
    samples, labels = sample_init()

samples_d = Proxy('_SAMPLES')
labels_d = Proxy('_LABELS')

samples, labels = sample_init()

def with_args(func):
    def wrapper():
        global samples_d, labels_d, samples, labels

        func(samples_d, labels_d, samples, labels)

    return wrapper

def R(func: str) -> FunctionType:
    def fun(*args, **kwargs) -> Union[Proxy, Any]:
        ret = _client.InvokeMethod('', func, args, kwargs)

        if isinstance(ret, Exception):
            raise ret
        elif isinstance(ret, str):
            return Proxy(ret)
        else:
            return ret

    return fun
