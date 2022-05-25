#
# Copyright (c) 2022
#

# TODO: import prime

from typing import Union

from prime.utils import logger

from prime_pb2 import *
from prime_pb2_grpc import *


class ServerError(Exception):
    def __init__(self, e: Exception):
        self.e = e


class ClientError(Exception):
    def __init__(self, e: Exception):
        self.e = e


def catch_xcpt(fitmodel: bool):
    def decorator(func):
        def wrapper(*args, **kwargs) -> Union[Ref, Model]:
            try:
                res = func(*args, **kwargs)
                res = Model(val=res) if fitmodel else Ref(name=res)
                logger.debug(f'wrapper: {res}')
            except Exception as e:
                res = (Model(error=ServerError(e)) if fitmodel
                       else Ref(error=ServerError(e)))

            return res
        return wrapper
    return decorator
