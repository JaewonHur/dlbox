#
# Copyright (c) 2022
#

import grpc
import click
from concurrent import futures

# from wrapper import Wrapper

from prime_pb2 import *
from prime_pb2_grpc import *

from utils import logger

class DataEnclave(DataEnclaveServicer):
    def ExportDef(self, arg: ExportDefArg, ctx: grpc.ServicerContext):
        logger.debug('ExportDef')
        return Ref(name='')

    def AllocateObj(self, arg: AllocateObjArg, ctx: grpc.ServicerContext):
        logger.debug('AllocateObj')
        return Ref(name='')

    def InvokeMethod(self, arg: InvokeMethodArg, ctx: grpc.ServicerContext):
        logger.debug('InvokeMethod')
        return Ref(name='')

    def FitModel(self, arg: FitModelArg, ctx: grpc.ServicerContext):
        logger.debug('FitModel')
        return Model(val=b'')


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
