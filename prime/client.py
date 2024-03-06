#
# Copyright (c) 2022
#

import os
import grpc
import dill

from typing import types, Type, List, Dict, Any, Tuple, Callable, Union
import pytorch_lightning as pl

from prime.utils import logger, MAX_MESSAGE_LENGTH
from prime.exceptions import retrieve_xcpt, retrieve_xcpts
from prime.hasref import HasRef

from prime_pb2 import *
from prime_pb2_grpc import *


TIMEOUT_SEC = 10


class PrimeClient:
    def __init__(self, ipaddr=None, port=None, cert=None, secure=False):
        ipaddr = ipaddr or 'localhost'
        port = port or 50051

        options = [
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)
        ]

        if secure:
            cert = cert or f'{os.getcwd()}/certs/cert.pem'
    
            with open(cert, 'rb') as fd:
                creds = grpc.ssl_channel_credentials(fd.read())

            self.channel = grpc.secure_channel(f'{ipaddr}:{port}', creds, 
                            options=options)
        else:
            self.channel = grpc.insecure_channel(f'{ipaddr}:{port}',
                            options=options)

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

    @retrieve_xcpts()
    def InvokeMethods(self,             #  obj  method args       kwargs 
                      lineages: Dict[str, 
                                     Tuple[str, str,   List[Any], Dict[str, Any]]]) -> Ref:

        tot_args = InvokeMethodsArg()

        for ref, lineage in lineages.items():
            ref_arg = RefInvokeMethodArg()
            arg = InvokeMethodArg()

            obj, method, args, kwargs = lineage

            args = [ dill.dumps(i) for i in args ]
            kwargs = { k:dill.dumps(v) for k, v in kwargs.items() }

            ref_arg.ref = ref
            arg.obj = obj
            arg.method = method
            arg.args.extend(args)

            for k, v in kwargs.items():
                arg.kwargs[k] = v

            ref_arg.arg.CopyFrom(arg)

            tot_args.lineages.extend([ref_arg])

        refs = self.stub.InvokeMethods(tot_args)
        return refs

    @retrieve_xcpt(False)
    def ExportModel(self, fullname: str, source: str) -> Ref:
        arg = ExportModelArg(fullname=fullname, source=source)

        ref = self.stub.ExportModel(arg)

        return ref

    @retrieve_xcpt(True)
    def FitModel(self, trainer: str, model: str, 
                 args: List[Any], kwargs: Dict[str, Any]) -> Model:

        arg = FitModelArg()
        arg.trainer = trainer
        arg.model = model
        
        args = [ dill.dumps(i) for i in args ]
        kwargs = { k:dill.dumps(v) for k, v in kwargs.items() }

        arg.args.extend(args)
        
        for k, v in kwargs.items():
            arg.kwargs[k] = v
            
        ref = self.stub.FitModel(arg)
        return ref