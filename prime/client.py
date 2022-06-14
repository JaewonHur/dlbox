#
# Copyright (c) 2022
#

import grpc
import dill

from typing import types, Type, List, Dict

from prime.utils import logger
from prime.exceptions import retrieve_xcpt

from prime_pb2 import *
from prime_pb2_grpc import *


class PrimeClient:
    def __init__(self, port=50051):
        # TODO: construct secure channel
        self.channel = grpc.insecure_channel(f'[::]:{port}')

        self.stub = PrimeServerStub(self.channel)

    @retrieve_xcpt(False)
    def ExportDef(self, name: str, tpe: types, source: str) -> Ref:
        assert name.isidentifier(), f'not identifer: {name}'
        assert tpe in (type, types.FunctionType), f'not supported type: {tpe}'

        tpe = dill.dumps(tpe)
        arg = ExportDefArg(name=name, type=tpe, source=source)

        ref = self.stub.ExportDef(arg)

        return ref

    @retrieve_xcpt(False)
    def AllocateObj(self, obj: object) -> Ref:

        tpe = type(obj)
        tpe = dill.dumps(tpe)
        val = dill.dumps(obj)
        arg = AllocateObjArg(type=tpe, val=val)

        ref = self.stub.AllocateObj(arg)

        return ref

    @retrieve_xcpt(False)
    def InvokeMethod(self, obj: str, method: str, args: List[str],
                     kwargs: Dict[str,str]) -> Ref:
        arg = InvokeMethodArg()
        arg.obj = obj
        arg.method = method
        arg.args.extend(args)

        for k, v in kwargs.items():
            arg.kwargs[k] = v

        ref = self.stub.InvokeMethod(arg)

        return ref

    @retrieve_xcpt(True)
    def FitModel(self, trainer: str, model: bytes, dataloader: str,
                 args: List[str], kwargs: Dict[str,str]) -> Model:
        arg = FitModelArg()
        arg.trainer = trainer
        arg.model = model
        arg.dataloader = dataloader
        arg.args.extend(args)

        for k, v in kwargs.items():
            arg.kwargs[k] = v

        model = self.stub.FitModel(arg)

        return model
