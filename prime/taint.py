#
# Copyright (c) 2022
#
from __future__ import annotations

from typing import Set, List, Optional, Union
from types import FunctionType

from enum import Enum
from torch import Tensor

from prime.rule import TagError


class Status(Enum):
    SAFE      = 0
    DANGER    = 1
    UNDEF     = 2


class H(int):
    # TODO: Change hashing
    def __xor__(self, h: H):
        i = int(self)
        j = int(h)

        return i ^ j


class M:
    N: int = None
    F: Set[int] = None

    def __init__(self, status: Status, samples: Optional[Set[int]] = None):
        assert not samples or max(samples) < self.N

        self.status = status
        self.samples = samples

    @classmethod
    def set_n(cls, N: int):
        cls.N = N
        cls.F = set(range(cls.N))

    def __or__(self, m: M) -> M:
        if self.status == Status.DANGER or m.status == Status.DANGER:
            return M(Status.DANGER)

        elif self.status == Status.SAFE:
            return M(m.status, m.samples)

        elif m.status == Status.SAFE:
            return M(self.status, self.samples)

        else:
            u = self.samples | m.samples

            if u == self.F:
                return M(set(), Status.SAFE)

            else:
                return M(u, Status.UNDEF)


class Tag:
    def __init__(self, h: H, m: M):
        self.h = h
        self.m = m

    def __str__(self) -> str:
        h = '0x{:016x}'.format(self.h)
        return f'tag({h[0:5]}..{h[-2:]},{self.m.status.name})'

def SafeTag(i: int) -> Tag:  return Tag(H(i), M(Status.SAFE))
def DangerTag() -> Tag:      return Tag(H(0), M(Status.DANGER))
def UndefTag(i: int) -> Tag: return Tag(H(0), M(Status.UNDEF, set([i])))


# NOTE: Support only 1-axis
class TagSack:
    def __init__(self, tags: List[Tag]):
        self.tags = tags

    def __len__(self) -> int:
        return len(self.tags)

    def __getitem__(self, i: int) -> Tag:
        return self.tags[i]

    def __str__(self) -> str:
        tagstr = ('\n' + ' ' * 51).join(str(t) for t in self.tags)
        return f'tagsack[{len(self)}](\n{" "*51}{tagstr})'


class TaintTracker:
    def __init__(self):
        self._tags = {}
        self._tagsacks = {}

    def __getitem__(self, k):
        if k in self._tags:
            return self._tags[k]
        elif k in self._tagsacks:
            return self._tagsacks[k]
        else:
            raise KeyError(f'{k} is not tainted')

    def __setitem__(self, k: str, v: Union[Tag, TagSack]):
        if isinstance(v, Tag):
            self._tags[k] = v
        else: # TagSack
            self._tagsacks[k] = v

    def init(self, N: int):
        M.set_n(N)


def propagate(method: FunctionType, module: Optional[str], _self: Any,
              tags: List[Union[Tag, TagSack]],
              kwtags: Dict[str, Union[Tag, TagSack]]) -> Tag:

    return DangerTag()
