#
# Copyright (c) 2022
#
from __future__ import annotations

from typing import Any, Set, List, Optional, Union, Callable
from enum import Enum
from functools import reduce

from torch import Tensor

from prime.utils import logger


class TagError(Exception):
    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self) -> str:
        return self.msg


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

    def __len__(self) -> int:
        if self.status == Status.UNDEF:
            return len(self.samples)
        else:
            return 0

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

    def __str__(self) -> str:
        if self.status == Status.UNDEF:
            samples = f"[{','.join(str(s) for s in self.samples)}]"
        else:
            samples = ''

        return f'{self.status.name}{samples}'


class Tag:
    def __init__(self, h: H, m: M):
        self.h = h
        self.m = m

    @staticmethod
    def merge(op_hash: int, tags: List[Tag], is_add: bool = False) -> Tag:

        m = reduce(lambda x, y: x | y, [ t.m for t in tags ])

        # TODO: It only allow strict add
        if is_add and len(tags) == 2 and tags[0].h == tags[1].h:
            h = tags[0].h

        else:
            h = reduce(lambda x, y: x ^ y, [ op_hash ] + [ t.h for t in tags ])
            m = M(Status.DANGER) if len(m) > 1 else m

        return Tag(h, m)

    def set_danger():
        self.h = 0
        self.m = M(Status.DANGER)

    def __str__(self) -> str:
        h = '0x{:016x}'.format(self.h)

        return f'tag({h[0:5]}..{h[-2:]},{self.m})'


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

    def __delitem__(self, k: str):
        if k in self._tags:
            del self._tags[k]
        else:
            del self._tagsacks[k]

    def init(self, N: int):
        M.set_n(N)


_emul = [ 'getattr', 'setattr', 'iter', 'contains' ]

def taint(method: Callable, module: Optional[str],
          args: List[Any], kwargs: Dict[str, Any],
          self_tag: Optional[Union[Tag, TagSack]],
          tags: List[Union[Tag, TagSack]],
          kwtags: Dict[str, Union[Tag, TagSack]]) -> Tag:

    from prime.rule import taint_rules

    if module == 'torch': # All tensor methods belong to this
        _taint = taint_rules['torch']

    elif (hasattr(method, '__name__') and method.__name__  in _emul
          and isinstance(args[0], Tensor)):
        _taint = taint_rules['torch']

    else:
        _taint = taint_rules['default']

    return _taint(method, self_tag, tags, kwtags)
