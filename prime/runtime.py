#
# Copyright (c) 2022
#

# from prime.
from __future__ import annotations

import sys
import dill
import builtins
import importlib.util
from typing import types, List, Dict, Any

from functools import partial

from prime.utils import logger
from prime.exceptions import catch_xcpt, UserError, NotImplementedOutputError
from prime.hasref import HasRef

# TODO: import trusted libraries
################################################################################
# Trusted Libraries                                                            #
################################################################################

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

################################################################################

VAR_SFX = 'VAL'

BUILTIN_TYPES = [
    getattr(builtins, d) for d in dir(builtins)
    if isinstance(getattr(builtins, d), type)
] + [ type(None) ]


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

    def _add_to_ctx(self, obj: Any, name: str = None) -> str:
        if name:
            # Force add, may replace already an existing content
            self.__ctx[name] = obj
        else:
            name = f'{self.ctr}{VAR_SFX}'
            self.ctr += 1

            self.__ctx[name] = obj

        return name

    def _del_from_ctx(self, name: str):
        del self.__ctx[name]

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
    def AllocateObj(self, tpe: bytes, val: bytes) -> str:
        tpe, obj = dill.loads(tpe), dill.loads(val)
        tpe = type(obj) if issubclass(tpe, HasRef) else tpe

        assert (tpe in BUILTIN_TYPES or tpe.__name__ in self.__ctx), \
            f'type not defined: {tpe}'
        assert tpe == type(obj), f'type mismatch: {tpe} vs {type(obj)}'
        assert obj is not NotImplemented, f'invalid type: {obj}'

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
            else:
                method = getattr(builtins, method)

        else:
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
    def FitModel(self, trainer: str, model: bytes, dataloader: str,
                 args: List[str], kwargs: Dict[str,str]) -> bytes:

        trainer: pl.Trainer = self.__ctx[trainer]
        assert type(trainer) == pl.Trainer, \
            f'incorrect trainer type: {type(trainer)}'

        dataloader: DataLoader = self.__ctx[dataloader]
        assert type(dataloader) == DataLoader, \
            f'incorrect dataloader type: {type(dataloader)}'

        path = '/tmp/__model.pt'
        with open(path, 'wb') as fd:
            fd.write(model)

        model = torch.load(path)

        args = [self.__ctx[k] for k in args]
        kwargs = {k:self.__ctx[v] for k, v in kwargs.items()}

        try:
            trainer.fit(model, dataloader, *args, **kwargs)
        except Exception as e:
            raise UserError(e)

        torch.save(model, path)
        with open(path, 'rb') as fd:
            model = fd.read()

        return model
