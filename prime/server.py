#
# Copyright (c) 2022
#

import grpc
import click
from concurrent import futures
from typing import Optional

from prime.utils import logger, is_server
from prime.runtime import ExecutionRuntime

from prime_pb2 import *
from prime_pb2_grpc import *


class PrimeServer(PrimeServerServicer):
    def __init__(self, ci: Optional[str] = None):
        self._runtime = ExecutionRuntime(globals(), ci)

        is_server()

    def ExportDef(self, arg: ExportDefArg, ctx: grpc.ServicerContext) -> Ref:

        fullname = arg.fullname
        tpe = arg.type
        source = arg.source

        ref = self._runtime.ExportDef(fullname, tpe, source)

        return ref

    def DeleteObj(self, arg: DeleteObjArg, ctx: grpc.ServicerContext):

        name = arg.name
        self._runtime.DeleteObj(name)

        from google.protobuf.empty_pb2 import Empty
        return Empty()

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

        d_args = arg.d_args
        d_kwargs = arg.d_kwargs

        args = arg.args
        kwargs = arg.kwargs

        model = self._runtime.FitModel(trainer, model,
                                       d_args, d_kwargs, args, kwargs)

        return model

    def SupplyData(self, arg: SupplyDataArg, ctx: grpc.ServicerContext) -> Ref:

        datapairs = [ (p.sample, p.label) for p in arg.datapairs ]

        ref = self._runtime.SupplyData(datapairs)

        return ref


@click.command()
@click.option('--port', default=50051, help='grpc port number')
@click.option('--ci', default=None, type=click.Choice(['mnist']), help='ci-test to be tested')
def run(port, ci):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    add_PrimeServerServicer_to_server(PrimeServer(ci), server)

    # TODO: Add credential and open a secure port
    server.add_insecure_port(f'[::]:{port}')

    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    print('Server start...')
    run()
