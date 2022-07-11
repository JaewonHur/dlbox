#
# Copyright (c) 2022
#

import grpc
import dill

from typing import types, Type, List, Dict, Any, Tuple

from prime.utils import logger
from prime.exceptions import retrieve_xcpt
from prime.hasref import HasRef

from prime_pb2 import *
from prime_pb2_grpc import *


TIMEOUT_SEC = 10


class PrimeClient:
    def __init__(self, port=50051):
        # TODO: construct secure channel
        self.channel = grpc.insecure_channel(f'[::]:{port}')
        if not self.check_server():
            raise RuntimeError('grpc server is not ready')

        self.stub = PrimeServerStub(self.channel)

    def check_server(self) -> bool:
        try:
            grpc.channel_ready_future(self.channel).result(timeout=TIMEOUT_SEC)
            return True
        except grpc.FutureTimeoutError:
            return False

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
        val = dill.dumps(obj)
        arg = AllocateObjArg(val=val)

        ref = self.stub.AllocateObj(arg)

        return ref

    @retrieve_xcpt(False)
    def InvokeMethod(self, obj: str, method: str, args: List[str]=[],
                     kwargs: Dict[str,str]={}) -> Ref:
        arg = InvokeMethodArg()
        arg.obj = obj
        arg.method = method
        arg.args.extend(args)

        for k, v in kwargs.items():
            arg.kwargs[k] = v

        ref = self.stub.InvokeMethod(arg)

        return ref

    @retrieve_xcpt(True)
    def FitModel(self, trainer: bytes, model: bytes,
                 epochs: Dict[int, Tuple[List[str], List[str]]],
                 d_args: List[Any], d_kwargs: Dict[str,Any],
                 args: List[Any], kwargs: Dict[str,Any]) -> Model:

        _epochs = {}
        for k, v in epochs.items():
            assert len(v[0]) == len(v[1]), \
                'Numbers of samples and labels in epoch should be the same'

            epoch = Epoch(samples=v[0], labels=v[1])
            _epochs[k] = epoch

        d_args = [ dill.dumps(i) for i in d_args ]
        d_kwargs = { k:dill.dumps(v) for k, v in d_kwargs.items() }

        args = [ dill.dumps(i) for i in args ]
        kwargs = { k:dill.dumps(v) for k, v in kwargs.items() }

        arg = FitModelArg(trainer=trainer, model=model,
                          epochs=_epochs,
                          d_args=d_args, d_kwargs=d_kwargs,
                          args=args, kwargs=kwargs)

        model = self.stub.FitModel(arg)

        return model
