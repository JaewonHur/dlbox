#
# Copyright (c) 2022
#

# TODO: import prime

import dill

from typing import Union

from prime.utils import logger

from prime_pb2 import *
from prime_pb2_grpc import *


class PrimeNotSupportedError(Exception):
    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self):
        return self.msg


class PrimeError(Exception):
    def __init__(self, e: Exception):
        self.e = e


class UserError(Exception):
    def __init__(self, e: Exception):
        self.e = e


class NotImplementedOutputError(Exception):
    def __init__(self):
        pass


def catch_xcpt(fitmodel: bool):
    def decorator(func):
        def wrapper(*args, **kwargs) -> Union[Ref, Model]:
            try:
                res = func(*args, **kwargs)
                res = Model(val=res) if fitmodel else Ref(name=res)

            except NotImplementedOutputError as e:
                ne = dill.dumps(e)
                res = Model(error=ne) if fitmodel else Ref(error=ne)

            except UserError as ue:
                ue = dill.dumps(ue)
                res = Model(error=ue) if fitmodel else Ref(error=ue)

            except Exception as e:
                pe = dill.dumps(PrimeError(e))
                res = Model(error=pe) if fitmodel else Ref(error=pe)

            return res
        return wrapper
    return decorator


# TODO: Rebuild exception message (e.g., hide information in DE)
def retrieve_xcpt(fitmodel: bool):
    def decorator(func):
        def wrapper(*args, **kwargs) -> Union[str, bytes, Exception]:
            res = func(*args, **kwargs)

            if res.error:
                err = dill.loads(res.error)

                if isinstance(err, UserError):
                    return err.e
                elif isinstance(err, NotImplementedOutputError):
                    return NotImplemented
                else:
                    raise err
            else:
                ret = res.name if not fitmodel else res.val
                return ret

        return wrapper
    return decorator
