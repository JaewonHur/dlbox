#
# Copyright (c) 2022
#
from __future__ import annotations

from typing import Any, Callable, Union, List, Optional, Dict
from types import FunctionType
from copy import deepcopy
from collections.abc import Iterator

from prime.utils import logger
from prime.taint import TagError, Tag, TagSack, TagSackIterator, DangerTag, SafeTag, FrozenTag


METHOD   = 0
ARGS     = 1
KWARGS   = 2
SELF_TAG = 3
TAGS     = 4
KWTAGS   = 5

def _default(*a) -> Optional[Tag]:
    method, args, kwargs, self_tag, tags, kwtags = a
    method = method.__name__

    # TODO: Need to handle tensor functions
    if isinstance(self_tag, TagSack):
        if not self_tag.is_safe():
            raise TagError(f'unsafe TagSack cannot invoke {method}')

        if not all(not isinstance(t, TagSack)
                   for t in tags + list(kwtags.values())):
            raise TagError(f'TagSack cannot be used as an argument of {method}')

        kwtags = [ Tag(hash(k) ^ t.h, t.m) for k, t in kwtags.items() ]
        for i in range(len(self_tag)):
            self_tag[i] = Tag.merge(hash(method),
                                    [self_tag[i]] + tags + kwtags)

        return deepcopy(self_tag)

    # TODO: release this constraint
    elif not all(not isinstance(t, TagSack)
                 for t in tags + list(kwtags.values())):

        if any(not t.is_safe() for t in kwtags.values()):
            raise TagError(f'{method} is not allowed on TagSack with such args')

        if method in ('cat'):
            tagsack = tags[0]

            if not tagsack.is_safe():
                raise TagError(f'{method} is not allowed on partial TagSack')

            return tagsack

        elif method in ('PrimeDataset'):
            s_tagsack = tags[0]
            l_tagsack = tags[1]
            
            if not s_tagsack.is_safe() or not l_tagsack.is_safe():
                raise TagError(f'{method} is not allowed on partial TagSack')

            return FrozenTag()

        else:
            raise TagError(f'{method} is not allowed on TagSack')


    else: # All arguments are Tag
        self_tag = [self_tag] if self_tag else []
        kwtags = [ Tag(hash(k) ^ t.h, t.m) for k, t in kwtags.items() ]

        # TODO: handle iadd
        return Tag.merge(hash(method), self_tag + tags + kwtags,
                         method == '__add__')

def _getattr(*a) -> Union[Tag, TagSack]:
    if not a[TAGS][1].is_safe():
        raise TagError('getattr cannot use unsafe tag')

    return a[TAGS][0]

def _setattr(*a) -> Tag:
    raise TagError('setattr on Tensor is prohibited')

def _iter(*a) -> Tag:
    if not isinstance(a[SELF_TAG], TagSack):
        raise TagError('iter on tagged Tensor is prohibited')

    return TagSackIterator(a[SELF_TAG])

def _next(*a) -> Tag:
    if not isinstance(a[SELF_TAG], TagSackIterator):
        raise TagError('__next__ on non-TagSackIterator')

    return next(a[SELF_TAG])

def _contains(*a) -> Tag:
    method, args, kwargs, self_tag, tags, kwtags = a

    if method.__name__ == 'contains':
        self_tag = tags[0]
        tag = a[TAGS][1]
    else:
        tag = a[TAGS][0]

    if not isinstance(self_tag, TagSack):
        raise TagError('contains on tagged Tensor is prohibited')

    return tag

def _getitem(*a) -> Union[Tag, TagSack]:
    if not isinstance(a[SELF_TAG], TagSack):
        raise TagError('__getitem__ on tagged Tensor is prohibited')

    elif not a[TAGS][0].is_safe():
        raise TagError('__getitem__ cannot use unsafe tag')

    self_tag = a[SELF_TAG]
    args = a[ARGS]

    if isinstance(args[0], slice):
        tag = TagSack(self_tag[args[0]])
    else:
        tag = self_tag[args[0]]

    return tag

def _setitem(*a):
    if not isinstance(a[SELF_TAG], TagSack):
        raise TagError('__setitem__ on tagged Tensor is prohibited')

    self_tag = a[SELF_TAG]
    args = a[ARGS]
    tags = a[TAGS]

    if isinstance(args[0], slice):
        raise TagError('setitem using slice is prohibited')

    self_tag[args[0]] = tags[1]


def _call(*a) -> Tag:
    method, args, kwargs, self_tag, tags, kwtags = a
    method = method.__self__.__name__

    # TODO: release this constraint later
    if method.endswith('_'):
        raise TagError('in-place operation is not allowed')

    # TODO: return SafeTag on safe operations
    if isinstance(self_tag, TagSack):
        if method in ('mean', 'std', 'sum'):
            arg = (args[0] if args
                   else (kwargs['axis'] if 'axis' in kwargs else None))

            arg_is_safe = ((arg is None) or
                           (isinstance(arg, int) and arg == 0) or
                           ((type(arg) in (list, tuple)) and arg[0] == 0))

            tag = (SafeTag() if arg_is_safe and self_tag.is_safe()
                   else DangerTag())

        elif method in ('to'):
            return self_tag

        else:
            raise TagError(f'{method} is not allowed on TagSack')

    elif isinstance(self_tag, Tag):

        kwtags = [ Tag(hash(k) ^ t.h, t.m) for k, t in kwtags.items() ]

        tag = Tag.merge(hash(method), [self_tag] + tags + kwtags)

    else: # Functions
        raise TagError('method must have tag')

    return tag

def _len(*a) -> Tag:
    method, args, _, self_tag, _, _ = a

    if isinstance(self_tag, TagSack):
        if self_tag.is_safe():
            tag = SafeTag()
        else:
            tag = DangerTag()

    else:
        tag = Tag.merge(hash(method), [self_tag])

    return tag


rule_table = {
    'getattr':      _getattr,
    'setattr':      _setattr,
    '__iter__':     _iter,
    '__next__':     _next,
    '__contains__': _contains,
    'contains':     _contains,
    '__getitem__':  _getitem,
    '__setitem__':  _setitem,
    '__call__':     _call,
    '__len__':      _len,
}

def taint_torch(method: Callable,
                args: List[Any], kwargs: Dict[str, Any],
                self_tag: Optional[Union[Tag, TagSack]],
                tags: List[Union[Tag, TagSack]],
                kwtags: Dict[str, Union[Tag, TagSack]]) -> Tag:

    _rule = (rule_table[method.__name__] if method.__name__ in rule_table
             else _default)

    tag = _rule(method, args, kwargs, self_tag, tags, kwtags)

    return (tag if tag else DangerTag())
