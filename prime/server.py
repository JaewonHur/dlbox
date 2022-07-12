#
# Copyright (c) 2022
#

import grpc
import click
from concurrent import futures

from prime.utils import logger, is_server
from prime.runtime import ExecutionRuntime

from prime_pb2 import *
from prime_pb2_grpc import *


class PrimeServer(PrimeServerServicer):
    def __init__(self):
        self._runtime = ExecutionRuntime()

        is_server()

    def ExportDef(self, arg: ExportDefArg, ctx: grpc.ServicerContext) -> Ref:

        fullname = arg.fullname
        tpe = arg.type
        source = arg.source

        ref = self._runtime.ExportDef(fullname, tpe, source)

        return ref

    def AllocateObj(self, arg: AllocateObjArg, ctx: grpc.ServicerContext) -> Ref:

        val = arg.val

        ref = self._runtime.AllocateObj(val)

        return ref

    def InvokeMethod(self, arg: InvokeMethodArg, ctx: grpc.ServicerContext) -> Ref:

        obj = arg.obj
        method = arg.method
        args = arg.args
        kwargs = arg.kwargs

        ref = self._runtime.InvokeMethod(obj, method, args, kwargs)

        return ref

    def FitModel(self, arg: FitModelArg, ctx: grpc.ServicerContext) -> Model:

        trainer = arg.trainer
        model = arg.model

        epochs = { k:(v.samples, v.labels) for k, v in arg.epochs.items() }

        d_args = arg.d_args
        d_kwargs = arg.d_kwargs

        args = arg.args
        kwargs = arg.kwargs

        model = self._runtime.FitModel(trainer, model, epochs,
                                       d_args, d_kwargs,
                                       args, kwargs)

        return model


@click.command()
@click.option('--port', default=50051, help='grpc port number')
def run(port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    add_PrimeServerServicer_to_server(PrimeServer(), server)

    # TODO: Add credential and open a secure port
    server.add_insecure_port(f'[::]:{port}')

    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    print('Server start...')
    run()
