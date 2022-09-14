#
# Copyright (c) 2022
#

import grpc
import dill

from typing import types, Type, List, Dict, Any, Tuple, Callable, Union
import pytorch_lightning as pl

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
    def ExportDef(self, fullname: str, tpe: types, source: str) -> Ref:
        assert tpe in (type, types.FunctionType), f'not supported type: {tpe}'

        tpe = dill.dumps(tpe)
        arg = ExportDefArg(fullname=fullname, type=tpe, source=source)

        ref = self.stub.ExportDef(arg)

        return ref

    def DeleteObj(self, name: str):
        arg = DeleteObjArg(name=name)

        self.stub.DeleteObj(arg)

    @retrieve_xcpt(False)
    def AllocateObj(self, obj: Any) -> Ref:
        val = dill.dumps(obj)
        arg = AllocateObjArg(val=val)

        ref = self.stub.AllocateObj(arg)

        return ref

    @retrieve_xcpt(False)
    def InvokeMethod(self, obj: str, method: str, args: List[Any]=[],
                     kwargs: Dict[str,Any]={}) -> Ref:
        arg = InvokeMethodArg()
        arg.obj = obj
        arg.method = method

        args = [ dill.dumps(i) for i in args ]
        kwargs = { k:dill.dumps(v) for k, v in kwargs.items() }

        arg.args.extend(args)

        for k, v in kwargs.items():
            arg.kwargs[k] = v

        ref = self.stub.InvokeMethod(arg)
        return ref

    @retrieve_xcpt(True)
    def FitModel(self, trainer: pl.Trainer, model: pl.LightningModule,
                 d_args: List[Any], d_kwargs: Dict[str, Any],
                 args: List[Any], kwargs: Dict[str, Any]) -> Model:

        trainer = dill.dumps(trainer)
        model   = dill.dumps(model)

        d_args   = [ dill.dumps(i) for i in d_args ]
        d_kwargs = { k:dill.dumps(v) for k, v in d_kwargs.items() }

        args   = [ dill.dumps(i) for i in args ]
        kwargs = { k:dill.dumps(v) for k, v in kwargs.items() }

        arg = FitModelArg(trainer=trainer, model=model,
                          d_args=d_args, d_kwargs=d_kwargs,
                          args=args, kwargs=kwargs)

        model = self.stub.FitModel(arg)
        return model

    @retrieve_xcpt(False)
    def SupplyData(self, datapairs: List[Tuple['Proxy']]) -> Ref:

        datapairs = [ DataPair(sample=dill.dumps(p[0]), label=dill.dumps(p[1]))
                      for p in datapairs ]

        arg = SupplyDataArg(datapairs=datapairs)

        ref = self.stub.SupplyData(arg)
        return ref

    @retrieve_xcpt(False)
    def StreamData(self, samples: 'Proxy', labels: 'Proxy',
                   transforms: List[Union[Callable, str]], args: List[Tuple], kwargs: List[Dict],
                   max_epoch: int):

        samples = dill.dumps(samples)
        labels  = dill.dumps(labels)

        transforms = [ dill.dumps(t) for t in transforms ]
        args = [ dill.dumps(i) for i in args ]
        kwargs = [ dill.dumps(i) for i in kwargs ]

        max_epoch  = dill.dumps(max_epoch)

        arg = StreamDataArg(samples=samples, labels=labels,
                            transforms=transforms, args=args, kwargs=kwargs,
                            max_epoch=max_epoch)

        none = self.stub.StreamData(arg)
        return none
