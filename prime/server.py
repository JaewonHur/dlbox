#
# Copyright (c) 2022
#

import os
import sys
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

    def InvokeMethods(self, tot_args: InvokeMethodsArg, ctx: grpc.ServicerContext) -> Refs:

        refs = Refs()
        for l in tot_args.lineages:
            arg = l.arg

            obj = arg.obj
            method = arg.method
            args = arg.args
            kwargs = arg.kwargs

            ref = self._runtime.InvokeMethod(obj, method, args, kwargs, l.ref)
            refs.refs.extend([ref])

            if ref.error:
                break

        return refs

    def ExportModel(self, arg: ExportModelArg, ctx: grpc.ServicerContext) -> Ref:

        fullname = arg.fullname
        source = arg.source

        ref = self._runtime.ExportModel(fullname, source)

        return ref

    def FitModel(self, arg: FitModelArg, ctx: grpc.ServicerContext) -> Model:
        trainer = arg.trainer
        model = arg.model
        
        args = arg.args
        kwargs = arg.kwargs

        model = self._runtime.FitModel(trainer, model, args, kwargs)
        return model
    

@click.command()
@click.option('--port', default=50051, help='grpc port number')
@click.option('--secure', default=False, help='secure grpc channel')
@click.option('--dn', default=None,
              type=click.Choice(['mnist', 'cifar10', 'utkface', 'chestxray']),
              help='dataset name to train model')
@click.option('--ll', default='DEBUG', type=click.Choice(['DEBUG', 'INFO',
                                                           'ERROR', 'WARNING']),
              help='log level (DEBUG | INFO | ERROR | WARNING)')
@click.option('--privkey', default=f'{os.getcwd()}/certs/privkey.pem', 
              help='private key for grpc server')
@click.option('--cert', default=f'{os.getcwd()}/certs/cert.pem', 
              help='certificate for grpc server')
def run(port, secure, dn, ll, privkey, cert):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2),
                         options=[
                             ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                             ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)
                         ])

    add_PrimeServerServicer_to_server(PrimeServer(dn), server)

    if secure:

        with open(privkey, 'rb') as fd:
            privkey = fd.read()
        with open(cert, 'rb') as fd:
            cert = fd.read()

        credentials = grpc.ssl_server_credentials(((privkey, cert),))
        server.add_secure_port(f'[::]:{port}', credentials)
    else:
        server.add_insecure_port(f'[::]:{port}')

    set_log_level(ll)

    print(f'Server start with {dn} (logging: {ll})...', file=sys.stderr)

    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    run()
