#
# Copyright (c) 2022
#

import grpc
import click
from concurrent import futures

# from wrapper import Wrapper

from prime.utils import logger
from prime.runtime import ExecutionRuntime

from prime_pb2 import *
from prime_pb2_grpc import *

class PrimeServer(PrimeServerServicer):
    def __init__(self):
        self._runtime = ExecutionRuntime()

    def ExportDef(self, arg: ExportDefArg, ctx: grpc.ServicerContext) -> Ref:

        name = arg.name
        tpe = arg.type
        source = arg.source

        ref = self._runtime.ExportDef(name, tpe, source)

        return ref

    def AllocateObj(self, arg: AllocateObjArg, ctx: grpc.ServicerContext) -> Ref:

        name = arg.name
        tpe = arg.type
        val = arg.val

        ref = self._runtime.AllocateObj(name, tpe, val)

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
        dataloader = arg.dataloader
        args = arg.args
        kwargs = arg.kwargs

        model = self._runtime.FitModel(trainer, model, dataloader, args, kwargs)

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
