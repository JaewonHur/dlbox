from __future__ import annotations

from typing import Optional, Dict

from prime.exceptions import PrimeNotAllowedError


class FromRef:
    def __init__(self):
        self.refs = []

    def _add(self, ref: str):
        self.refs.append(ref)

    def __getitem__(self, i: int):
        return self.refs[i]

    def __len__(self) -> int:
        return len(self.refs)

    def __bool__(self) -> bool:
        return bool(self.refs)

    def __str__(self) -> str:
        return str(self.refs)


class HasRef(object):
    __ctx: Dict = None
    __can_useref: bool = True
    __fromref: FromRef = None

    def __new__(cls: HasRef, ref: str, *args):
        if cls.__ctx:
            if cls.__can_useref:
                cls.__fromref._add(ref)
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
    def _set_useref(cls, allow: bool):
        cls.__can_useref = allow

    @classmethod
    def _set_fromref(cls, fromref: FromRef):
        cls.__fromref = fromref

    def __getstate__(self):
        pass

    def __getnewargs__(self):
        return (self._ref,)
