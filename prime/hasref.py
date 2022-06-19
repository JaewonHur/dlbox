from __future__ import annotations

from typing import Optional, Dict
from types import NoneType

class HasRef(object):
    __ctx: Optional[NoneType, Dict] = None

    def __new__(cls: HasRef, ref: str):
        if cls.__ctx:
            return cls.__ctx[ref]
        else:
            return object.__new__(cls)

    def __init__(self, ref: str) -> HasRef:
        self._ref = ref

    @classmethod
    def _set_ctx(cls, ctx: Dict):
        cls.__ctx = ctx

    def __getstate__(self):
        pass

    def __getnewargs__(self):
        return (self._ref,)
