#
# Copyright (c) 2022
#

import grpc
import click
from concurrent import futures
from typing import Optional

from prime.utils import (
    is_server, set_log_level, MAX_MESSAGE_LENGTH
)
from prime.runtime import ExecutionRuntime

from prime_pb2 import *
from prime_pb2_grpc import *


class PrimeServer(PrimeServerServicer):
    def __init__(self, dn: Optional[str] = None):
        self._runtime = ExecutionRuntime(dn)

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

    def ExportModel(self, arg: ExportModelArg, ctx: grpc.ServicerContext) -> Ref:

        fullname = arg.fullname
        source = arg.source

        ref = self._runtime.ExportModel(fullname, source)

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

    def StreamData(self, arg: StreamDataArg, ctx: grpc.ServicerContext):

        samples = arg.samples
        labels = arg.labels

        transforms = arg.transforms
        args = arg.args
        kwargs = arg.kwargs

        max_epoch = arg.max_epoch

        none = self._runtime.StreamData(samples, labels, transforms, args, kwargs, max_epoch)
        return none


@click.command()
@click.option('--port', default=50051, help='grpc port number')
@click.option('--ci', default=None,
              type=click.Choice(['mnist', 'cifar10']),
              help='ci-test to be tested')
@click.option('--dn', default=None,
              type=click.Choice(['cifar10']),
              help='dataset name to train model')
@click.option('--ll', default='DEBUG', type=click.Choice(['DEBUG', 'INFO',
                                                           'ERROR', 'WARNING']),
              help='log level (DEBUG | INFO | ERROR | WARNING)')
def run(port, ci, dn, ll):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2),
                         options=[
                             ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                             ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)
                         ])


    dn = dn if dn else ci # TODO move ci to dn
    add_PrimeServerServicer_to_server(PrimeServer(dn), server)

    # TODO: Add credential and open a secure port
    server.add_insecure_port(f'[::]:{port}')

    set_log_level(ll)

    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    print('Server start...')
    run()
