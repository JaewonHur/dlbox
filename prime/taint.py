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
    def __xor__(self, h: H) -> H:
        i = int(self)
        j = int(h)

        return H(i ^ j)


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

    def has_only(self, i: int) -> bool:
        return self.samples ==  { i }

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
                return M(Status.SAFE)

            else:
                return M(Status.UNDEF, u)

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

        m = reduce(lambda x, y: x | y, [ t.m for t in tags ],
                   M(Status.SAFE))

        # TODO: It only allow strict add
        if is_add and len(tags) == 2 and tags[0].h == tags[1].h:
            h = tags[0].h

        else:
            h = reduce(lambda x, y: x ^ y,
                       [ op_hash ] + [ t.h for t in tags ], 0)
            m = M(Status.DANGER) if len(m) > 1 else m

        return Tag(h, m)

    def is_safe(self) -> bool:
        return (self.m.status == Status.SAFE)

    def is_undef(self) -> bool:
        return (self.m.status == Status.UNDEF)

    def is_danger(self) -> bool:
        return (self.m.status == Status.DANGER)

    def set_danger(self):
        self.h = 0
        self.m = M(Status.DANGER)

    def __str__(self) -> str:
        h = '0x{:016x}'.format(self.h)

        return f'tag({h[0:5]}..{h[-2:]},{self.m})'


def SafeTag(i: int = 0) -> Tag:      return Tag(H(i), M(Status.SAFE))
def DangerTag() -> Tag:              return Tag(H(0), M(Status.DANGER))
def UndefTag(h: int, i: int) -> Tag: return Tag(H(h), M(Status.UNDEF, set([i])))


# NOTE: Support only 1-axis
class TagSack:
    def __init__(self, tags: List[Tag]):
        self.tags = tags

    def is_safe(self) -> bool:
        h = self.tags[0].h

        return (len(self.tags) == M.N and
                all(t.h == h for t in self.tags) and
                all(t.m.has_only(i) for i, t in enumerate(self.tags)))

    def __len__(self) -> int:
        return len(self.tags)

    def __getitem__(self, i: int) -> Tag:
        return self.tags[i]

    def __setitem__(self, i: int, v: Union[Tag, TagSack]):
        if isinstance(v, TagSack):
            raise TagError('nesting TagSack is prohibited')

        self.tags[i] = v

    def __str__(self) -> str:
        tags = self.tags[:2] + (['...', self.tags[-1]] if len(self.tags) > 2
                                else [])
        tagstr = ('\n' + ' ' * 51).join(str(t) for t in tags)
        return f'tagsack[{len(self)}](\n{" "*51}{tagstr})'


class TagSackIterator:
    def __init__(self, tagsack: TagSack):
        self.tagsack = tagsack
        self.it = iter(self.tagsack)

    def __iter__(self) -> TagSackIterator:
        self.it = iter(self.tagsack)
        return self

    def __next__(self) -> Tag:
        return next(self.it)


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

    return _taint(method, args, kwargs, self_tag, tags, kwtags)
