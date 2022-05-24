#
# Copyright (c) 2022
#

import grpc
import click
from concurrent import futures

# from wrapper import Wrapper

from prime_pb2 import *
from prime_pb2_grpc import *

# TODO: add prime prefix when modularize
from utils import logger
from runtime import ExecutionRuntime

class DataEnclave(DataEnclaveServicer):
    def __init__(self):
        self._runtime = ExecutionRuntime()

    def ExportDef(self, arg: ExportDefArg, ctx: grpc.ServicerContext):

        name = arg.name
        tpe = arg.type
        source = arg.source

        name = self._runtime.ExportDef(name, tpe, source)

        return Ref(name=name)

    def AllocateObj(self, arg: AllocateObjArg, ctx: grpc.ServicerContext):

        name = arg.name
        tpe = arg.type
        val = arg.val

        name = self._runtime.AllocateObj(name, tpe, val)

        return Ref(name=name)

    def InvokeMethod(self, arg: InvokeMethodArg, ctx: grpc.ServicerContext):

        obj = arg.obj
        method = arg.method
        args = arg.args
        kwargs = arg.kwargs

        name = self._runtime.InvokeMethod(obj, method, args, kwargs)

        return Ref(name=name)

    def FitModel(self, arg: FitModelArg, ctx: grpc.ServicerContext):

        trainer = arg.trainer
        model = arg.model
        dataloader = arg.dataloader
        args = arg.args
        kwargs = arg.kwargs

        model = self._runtime.FitModel(trainer, model, dataloader, args, kwargs)

        return Model(val=model)


@click.command()
@click.option('--port', default=50051, help='grpc port number')
def run(port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    add_DataEnclaveServicer_to_server(DataEnclave(), server)

    # TODO: Add credential and open a secure port
    server.add_insecure_port(f'[::]:{port}')

    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    print('Server start...')
    run()
