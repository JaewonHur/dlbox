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

from typing import Any

import prime
from prime.proxy import _client
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

def read_val(_client: PrimeClient, x: Any) -> Any:
    _client.InvokeMethod('__main__', '__main__.output', [x])
    with open('/tmp/x.txt', 'rb') as fd:
        x = dill.load(fd)

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

