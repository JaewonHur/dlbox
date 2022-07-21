#
# Copyright (c) 2022
#

from __future__ import annotations

import os
import sys
import dill
import shutil
import builtins
import importlib.util
from typing import types, List, Dict, Any, Optional, Type, Tuple
from types import FunctionType

from functools import partial

from prime.utils import logger
from prime.exceptions import catch_xcpt, UserError, NotImplementedOutputError
from prime.hasref import HasRef
from prime.data import build_dataloader

VAR_SFX = 'VAL'

################################################################################
# Trusted Libraries                                                            #
################################################################################

TRUSTED_PKGS = {}

def _trust(pkg: str):
    module = __import__(pkg)

    global TRUSTED_PKGS
    TRUSTED_PKGS[pkg] = module


_trust('builtins')
_trust('torch')
_trust('pytorch_lightning')

################################################################################

def get_fullname(obj: type) -> str:
    try:
        return f'{obj.__module__}.{obj.__name__}'
    except:
        return ''

def path_exists(m: ModuleType, n: str) -> bool:
    if hasattr(m, '__path__'):
        path = '/'.join(m.__path__[0].split('/')[:-1])
        if (os.path.exists(f'{path}/{n}') or
            os.path.exists(f'{path}/{n}.py')):
            return True

    return False


def get_from(name: str) -> FunctionType:
    pkg = name.split('.')[0]
    if pkg not in TRUSTED_PKGS:
        raise Exception(f'{name} is not from trusted packages')

    m = TRUSTED_PKGS[pkg]
    for n in name.split('.')[1:]:
        if hasattr(m, n):
            m = getattr(m, n)
        elif path_exists(m, n):
            m = __import__(f'{m.__name__}.{n}')
        else:
            raise Exception(f'{name} is not from trusted packages')

    return m


def from_trusted(tpe: type) -> bool:
    if tpe is type(None): return True

    try:
        name = f'{tpe.__module__}.{tpe.__name__}'
    except:
        return False

    pkg = name.split('.')[0]
    if pkg not in TRUSTED_PKGS:
        return False

    m = TRUSTED_PKGS[pkg]
    for n in name.split('.')[1:]:
        if hasattr(m, n):
            m = getattr(m, n)
        elif path_exists(m, n):
            m = __import__(f'{m.__name__}.{n}')
        else:
            return False

    if tpe is m:
        return True
    else:
        return True


class ExecutionRuntime():
    __initialized = False
    __ctx = {}

    def __init__(self, g_ctx: Dict[str, Any]):
        self.g_ctx = g_ctx

        if self.__initialized:
            raise RuntimeError('There can exist only one ExecutionRuntime')
        self.__init()

        self.ctr = 0

    @classmethod
    def __init(cls: ExecutionRuntime):
        cls.__initialized = True
        HasRef._set_ctx(cls.__ctx)

    # TODO: Need lock?
    def _add_to_ctx(self, obj: Any, name: str = None) -> str:
        if name:
            # Force add, may replace already an existing content
            self.__ctx[name] = obj
        else:
            name = f'{self.ctr}{VAR_SFX}'
            self.ctr += 1

            self.__ctx[name] = obj

        logger.debug(f'{name}: {hex(id(obj))}')
        return name

    # TODO: Need lock?
    def _del_from_ctx(self, name: str):

        logger.debug(f'{name}')
        del self.__ctx[name]

    def _from_ctx(self, tpe: type) -> bool:
        if get_fullname(tpe) not in self.__ctx:
            return False

        if tpe is not self.__ctx[get_fullname(tpe)]:
            return False

        return True

    # TODO: Still need to check a class instance from trusted package does
    # not contain malicious method
    def _deserialize(self, val: bytes) -> Any:
        obj = dill.loads(val)
        tpe = type(obj)

        assert not tpe in [NotImplemented, type, FunctionType], \
            f'invalid type: {obj}'
        assert (from_trusted(tpe) or self._from_ctx(tpe)), \
            f'type not trusted: {tpe}, {self.__ctx}'

        return obj


    @catch_xcpt(False)
    def ExportDef(self, fullname: str, tpe: bytes, source: str) -> str:
        tpe = dill.loads(tpe)

        assert tpe in (type, types.FunctionType), f'not supported type: {tpe}'

        logger.debug(f'{fullname} ({tpe})')

        # TODO: Sandbox codes
        # Malicious codes can change states of global variables

        if fullname.startswith('__main__'):
            # NOTE: Do not support nested definition
            assert len(fullname.split('.')) == 2
            name = fullname.split('.')[1]
            assert name not in globals()

            try:
                locals()['__name__'] = '__main__'
                exec(source, locals())
                del locals()['__name__']
            except Exception as e:
                raise UserError(e)

            obj = locals()[name]
            self.g_ctx[name] = obj

        else:
            module_path = fullname.split('.')[:-1]
            name = fullname.split('.')[-1]

            for i in range(len(module_path) -1):
                p = f'/tmp/{"/".join(module_path[:i+1])}'
                os.mkdir(f'{p}')

                with open(p + '/__init__.py', 'w') as fd:
                    fd.write(f'import {".".join(module_path[:i+2])}')

            with open(f'/tmp/{"/".join(module_path)}.py', 'w') as fd:
                fd.write(source)

            root = module_path[0]
            path = (f'/tmp/{root}.py' if len(module_path) == 1
                    else f'/tmp/{root}/__init__.py')
            spec = importlib.util.spec_from_file_location(root, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[root] = module

            try:
                spec.loader.exec_module(module)
            except Exception as e:
                raise UserError(e)

            globals()[root] = sys.modules[root]

            if len(module_path) == 1: os.remove(f'/tmp/{root}.py')
            else: shutil.rmtree(f'/tmp/{root}')

            obj = module
            for n in fullname.split('.')[1:]:
                obj = getattr(obj, n)

        self._add_to_ctx(obj, fullname)
        return name

    def DeleteObj(self, name: str):
        self._del_from_ctx(name)

    @catch_xcpt(False)
    def InvokeMethod(self, obj: str, method: str,
                     args: List[bytes], kwargs: Dict[str,bytes]) -> str:

        if obj == '__main__':
            if method in self.__ctx.keys():
                method = self.__ctx[method]

            else:
                method = get_from(method)

        else: # obj is allocated in context
            obj = self.__ctx[obj]

            try:
                method = getattr(obj, method)
            except Exception as e:
                raise UserError(e)

        logger.debug(f'{method}')
        args = [ self._deserialize(i) for i in args ]
        kwargs = { k:self._deserialize(v) for k, v in kwargs.items() }

        try:
            out = method(*args, **kwargs)
        except Exception as e:
            raise UserError(e)

        if out is NotImplemented:
            raise NotImplementedOutputError()

        name = self._add_to_ctx(out)
        return name

    @catch_xcpt(True)
    def FitModel(self, trainer: bytes, model: bytes,
                 epoch: Tuple[List[str], List[str]],
                 d_args: List[bytes], d_kwargs: Dict[str, bytes],
                 args: List[bytes], kwargs: Dict[str,bytes]) -> bytes:

        import torch
        import pytorch_lightning as pl

        HasRef._set_export(False)

        trainer = self._deserialize(trainer)
        model = self._deserialize(model)

        logger.debug(f'{model}')
        d_args = [ self._deserialize(i) for i in d_args ]
        d_kwargs = { k:self._deserialize(v) for k, v in d_kwargs.items() }

        samples, labels = epoch
        assert len(samples) == len(labels), \
            'Numbers of samples and labels should be the same'

        tagged_samples = [ (s, self.__ctx[s]) for s in samples ]
        tagged_labels = [ (l, self.__ctx[l]) for l in labels ]
        tagged_epoch = (tagged_samples, tagged_labels)

        dataloader = build_dataloader(tagged_epoch, d_args, d_kwargs)

        args = [ self._deserialize(i) for i in args ]
        kwargs = { k:self._deserialize(v) for k, v in kwargs.items() }

        HasRef._set_export(True)

        try:
            trainer.fit(model, dataloader, *args, **kwargs)
        except Exception as e:
            raise UserError(e)

        model = dill.dumps(model)
        return model

