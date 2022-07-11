#
# Copyright (c) 2022
#

# from prime.
from __future__ import annotations

import sys
import dill
import builtins
import importlib.util
from typing import types, List, Dict, Any, Optional, Type, Tuple
from types import FunctionType

from functools import partial

from prime.utils import logger
from prime.exceptions import catch_xcpt, UserError, NotImplementedOutputError
from prime.hasref import HasRef

VAR_SFX = 'VAL'

BUILTIN_TYPES = [
    getattr(builtins, d) for d in dir(builtins)
    if isinstance(getattr(builtins, d), type)
] + [ type(None) ]

# TODO: import trusted libraries
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

def get_name(obj: Any) -> str:
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


def get_from(name: str) -> Any:
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


def from_trusted(name: str) -> bool:
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

    return True


class ExecutionRuntime():
    __initialized = False
    __ctx = {}

    def __init__(self):
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

        return name

    # TODO: Need lock?
    def _del_from_ctx(self, name: str):
        del self.__ctx[name]

    # TODO: Still need to check a class instance from trusted package does
    # not contain malicious method
    def _deserialize(self, val: bytes) -> Any:
        obj = dill.loads(val)
        tpe = type(obj)

        assert obj is not NotImplemented, f'invalid type: {obj}'
        assert tpe == type(obj), f'type mismatch: {tpe} vs {type(obj)} of {obj}'
        assert (tpe in BUILTIN_TYPES or tpe.__name__ in self.__ctx
                or from_trusted(get_name(tpe))), \
                f'type not defined: {tpe}'

        return obj


    @catch_xcpt(False)
    def ExportDef(self, name: str, tpe: bytes, source: str) -> str:
        tpe = dill.loads(tpe)

        assert tpe in (type, types.FunctionType), f'not supported type: {tpe}'

        module = f'/tmp/__{name}.py'

        with open(module, 'w') as fd:
            fd.write(source)

        spec = importlib.util.spec_from_file_location(name, module)
        module = importlib.util.module_from_spec(spec)

        sys.modules[name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise UserError(e)

        self._add_to_ctx(getattr(module, name), name)

        logger.debug(f'{name}: {self.__ctx[name]}')

        return name

    @catch_xcpt(False)
    def AllocateObj(self, val: bytes) -> str:

        obj = self._deserialize(val)
        name = self._add_to_ctx(obj)

        logger.debug(f'{name}: {obj}')

        return name

    @catch_xcpt(False)
    def InvokeMethod(self, obj: str, method: str,
                     args: List[str], kwargs: Dict[str,str]) -> str:

        if obj == '__main__':
            if method in self.__ctx.keys():
                method = self.__ctx[method]

            elif method == '_del_from_ctx':
                self._del_from_ctx(*[self.__ctx[k] for k in args])
                self._del_from_ctx(*args)

                return ""

            elif from_trusted(method):
                method = get_from(method)

            else:
                raise Exception(f'{method} is not trusted')

        else: # obj is allocated in context
            obj = self.__ctx[obj]

            try:
                method = getattr(obj, method)
            except Exception as e:
                raise UserError(e)

        args = [self.__ctx[k] for k in args]
        kwargs = {k:self.__ctx[v] for k, v in kwargs.items()}

        try:
            out = method(*args, **kwargs)
        except Exception as e:
            raise UserError(e)

        if out is NotImplemented:
            raise NotImplementedOutputError()

        name = self._add_to_ctx(out)

        logger.debug(f'{name}: {out}')

        return name

    @catch_xcpt(True)
    def FitModel(self, trainer: bytes, model: bytes,
                 epochs: Dict[int, Tuple[List[str], List[str]]],
                 d_args: List[bytes], d_kwargs: Dict[str, bytes],
                 args: List[bytes], kwargs: Dict[str,bytes]) -> bytes:

        import pytorch_lightning as pl
        from torch.utils.data import DataLoader

        HasRef._set_export(False)

        trainer: pl.Trainer = dill.loads(trainer)
        assert isinstance(trainer, pl.Trainer), \
            f'incorrect trainer type: {type(trainer)}'

        model: pl.LightningModule = dill.loads()
        assert isinstance(model, pl.LightningModule), \
            f'incorrect model type: {type(model)}'

        dataloader: DataLoader = dill.loads(dataloader)
        assert isinstance(dataloader, DataLoader), \
            f'incorrect dataloader type: {type(dataloader)}'

        # TODO: Sanitize epochs and construct DataSet

        d_args = [ self._deserialize(i) for i in d_args ]
        d_kwargs = { k:self._deserialize(v) for k, v in d_kwargs.items() }

        args = [ self._deserialize(i) for i in args ]
        kwargs = { k:self._deserialize(v) for k, v in kwargs.items() }

        HasRef._set_export(True)

        try:
            trainer.fit(model, dataloader, *args, **kwargs)
        except Exception as e:
            raise UserError(e)

        model = dill.dumps(model)
        return model
