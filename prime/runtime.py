#
# Copyright (c) 2022
#

# from prime.

import sys
import dill
import builtins
import importlib.util
from typing import types, List, Dict

from prime.utils import logger
from prime.exceptions import catch_xcpt, UserError

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
]

class ExecutionRuntime():
    def __init__(self):
        self.__ctx = {}

        self.ctr = 0

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

        self.__ctx[name] = getattr(module, name)

        logger.debug(f'{name}: {self.__ctx[name]}')

        return name

    @catch_xcpt(False)
    def AllocateObj(self, tpe: bytes, val: bytes) -> str:
        tpe = dill.loads(tpe)
        obj = dill.loads(val)

        assert (tpe in BUILTIN_TYPES or tpe.__name__ in self.__ctx), \
            f'type not defined: {tpe}'
        assert tpe == type(obj), f'type mismatch: {tpe} vs {type(obj)}'

        name = f'{self.ctr}{VAR_SFX}'
        self.ctr += 1

        self.__ctx[name] = obj

        logger.debug(f'{name}: {obj}')

        return name

    @catch_xcpt(False)
    def InvokeMethod(self, obj: str, method: str,
                     args: List[str], kwargs: Dict[str,str]) -> str:

        if obj == '__main__':
            if method in self.__ctx.keys():
                method = self.__ctx[method]

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

        name = f'{self.ctr}{VAR_SFX}'
        self.ctr += 1

        try:
            out = method(*args, **kwargs)
        except Exception as e:
            raise UserError(e)

        self.__ctx[name] = out

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
