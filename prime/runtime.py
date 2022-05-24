#
# Copyright (c) 2022
#

# from prime.

import sys
import dill
import importlib.util
from typing import types, List, Dict

# TODO: add prime prefix when modularize
from utils import logger

# TODO: import ml libraries
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

VAR_SFX = 'VAL'

class ExecutionRuntime():
    def __init__(self):
        self.__ctx = {}

        self.ctr = 0

    def ExportDef(self, name: str, tpe: bytes, source: str) -> str:
        tpe = dill.loads(tpe)

        assert tpe in (type, types.FunctionType), f'not supported type: {tpe}'

        module = f'/tmp/__{name}.py'

        with open(module, 'w') as fd:
            fd.write(source)

        spec = importlib.util.spec_from_file_location(name, module)
        module = importlib.util.module_from_spec(spec)

        sys.modules[name] = module
        spec.loader.exec_module(module)

        self.__ctx[name] = getattr(module, name)

        logger.debug(f'{name}: {self.__ctx[name]}')

        return name

    def AllocateObj(self, name: str, tpe: bytes, val: bytes) -> str:
        tpe = dill.loads(tpe)
        obj = dill.loads(val)

        assert tpe == type(obj), f'type mismatch: {tpe} vs {type(obj)}'

        self.__ctx[name] = obj

        logger.debug(f'{name}: {obj}')

        return name

    def InvokeMethod(self, obj: str, method: str,
                     args: List[str], kwargs: Dict[str,str]) -> str:

        if obj == '__main__': # TODO change name?
            if method in self.__ctx.keys():
                method = self.__ctx[method]

            else:
                raise NotImplementedError('')

        else:
            obj = self.__ctx[obj]
            method = getattr(obj, method)

        args = [self.__ctx[k] for k in args]
        kwargs = {k:self.__ctx[v] for k, v in kwargs.items()}

        name = f'{self.ctr}{VAR_SFX}'
        self.ctr += 1

        out = method(*args, **kwargs)

        self.__ctx[name] = out

        logger.debug(f'{name}: {out}')

        return name

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

        trainer.fit(model, dataloader, *args, **kwargs)

        torch.save(model, path)
        with open(path, 'rb') as fd:
            model = fd.read()

        return model
