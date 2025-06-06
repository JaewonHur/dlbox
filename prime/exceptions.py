#
# Copyright (c) 2022
#

# TODO: import prime

import dill
import time
import traceback

from typing import Union

from prime.profiler import Profile, profiles
from prime.utils import logger

from prime_pb2 import *
from prime_pb2_grpc import *


class PrimeNotSupportedError(Exception):
    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self) -> str:
        return self.msg


class PrimeError(Exception):
    def __init__(self, e: Exception):
        self.e = e

    def __str__(self) -> str:
        return str(self.e)


class UserError(Exception):
    def __init__(self, e: Exception):
        self.e = e


class PrimeNotAllowedError(Exception):
    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self) -> str:
        return self.msg


class NotImplementedOutputError(Exception):
    def __init__(self):
        pass


def catch_xcpt(fitmodel: bool):
    def decorator(func):
        def wrapper(*args, **kwargs) -> Union[Ref, Model]:
            try:
                res, pf = func(*args, **kwargs)
                res = (Model(val=res) if fitmodel else
                       (Ref(name=res) if isinstance(res, str) else Ref(obj=res)))
                
                if pf:
                    res.profile['serialize'] = dill.dumps(pf.serialize)
                    res.profile['taint'] = dill.dumps(pf.taint)
                    res.profile['op'] = dill.dumps(pf.op)

            except NotImplementedOutputError as e:
                ne = dill.dumps(e)
                res = Model(error=ne) if fitmodel else Ref(error=ne)

            except UserError as ue:
                ue = dill.dumps(ue)
                res = Model(error=ue) if fitmodel else Ref(error=ue)

            except PrimeNotAllowedError as pnae:
                pnae = dill.dumps(pnae)
                res = Model(error=pnae) if fitmodel else Ref(error=pnae)

            except Exception as e:
                traceback.print_exc()

                pe = dill.dumps(PrimeError(e))
                res = Model(error=pe) if fitmodel else Ref(error=pe)

            return res
        return wrapper
    return decorator


# TODO: Rebuild exception message (e.g., hide information in DE)
def retrieve_xcpt(fitmodel: bool):
    def decorator(func):
        def wrapper(*args, **kwargs) -> Union[str, bytes, Exception]:
            now = time.time()
            res = func(*args, **kwargs)

            tot = time.time() - now

            if res.error:
                err = dill.loads(res.error)

                if isinstance(err, UserError):
                    return err.e
                elif isinstance(err, NotImplementedOutputError):
                    return NotImplemented
                else:
                    raise err
            else:
                if res.profile:
                    profiles['count'] += 1

                    serialize = dill.loads(res.profile['serialize'])
                    taint = dill.loads(res.profile['taint'])
                    op = dill.loads(res.profile['op'])
                    rpc = tot - serialize - taint - op

                    profiles['rpc'] += rpc
                    profiles['serialize'] += serialize
                    profiles['op'] += op
                    profiles['taint'] += taint
                
                if fitmodel:
                    ret = res.val
                else:
                    ret = res.name or dill.loads(res.obj)

                return ret

        return wrapper
    return decorator


def retrieve_xcpts():
    def decorator(func):
        def wrapper(*args, **kwargs) -> Union[str, bytes, Exception]:
            res = func(*args, **kwargs)
            
            refs = res.refs
            ret = []
            
            for r in refs:
                if r.error:
                    err = dill.loads(r.error)
                    
                    if isinstance(err, UserError):
                        ret.append(err.e)
                    elif isinstance(err, NotImplementedOutputError):
                        ret.append(NotImplemented)
                    else:
                        raise err
                else:
                    ret.append(r.name if r.name else dill.loads(r.obj))
                    
            return ret

        return wrapper
    return decorator
                    