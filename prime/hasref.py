from __future__ import annotations

from typing import Optional, Dict
from types import NoneType

from prime.exceptions import PrimeNotAllowedError

class HasRef(object):
    __ctx: Optional[NoneType, Dict] = None
    __can_export: bool = True

    def __new__(cls: HasRef, ref: str):
        if cls.__ctx:
            if cls.__can_export:
                return cls.__ctx[ref]
            else:
                raise PrimeNotAllowedError(f'{ref}')
        else:
            return object.__new__(cls)

    def __init__(self, ref: str) -> HasRef:
        self._ref = ref

    @classmethod
    def _set_ctx(cls, ctx: Dict):
        cls.__ctx = ctx

    @classmethod
    def _set_export(cls, allow: bool):
        cls.__can_export = allow

    def __getstate__(self):
        pass

    def __getnewargs__(self):
        return (self._ref,)
