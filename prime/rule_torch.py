#
# Copyright (c) 2022
#
from __future__ import annotations

from typing import Any, Callable, Union, List, Optional
from types import FunctionType
from collections.abc import Iterator

from prime.taint import TagError, Tag, TagSack, TagSackIterator, DangerTag


METHOD   = 0
ARGS     = 1
KWARGS   = 2
SELF_TAG = 3
TAGS     = 4
KWTAGS   = 5

def _default(*a) -> Optional[Tag]:
    method, args, kwargs, self_tag, tags, kwtags = a

    if isinstance(self_tag, TagSack):
        raise TagError(f'{method} is not allowed on TagSack')

    self_tag = [self_tag] if self_tag else []
    kwtags = [ Tag(hash(k) ^ t.h, t.m) for k, v in kwtags.items() ]

    return Tag.merge(hash(method), self_tag + tags + kwtags,
                     method.__name__ == '__add__')

def _getattr(*a) -> Optional[Tag]:
    if not a[TAGS][1].is_safe():
        raise TagError('getattr cannot use unsafe tag')

    return a[TAGS][0]

def _setattr(*a) -> Optional[Tag]:
    raise TagError('setattr on Tensor is prohibited')

def _iter(*a) -> Optional[Tag]:
    if not isinstance(a[SELF_TAG], TagSack):
        raise TagError('iter on tagged Tensor is prohibited')

    return TagSackIterator(a[SELF_TAG])

def _next(*a) -> Optional[Tag]:
    if not isinstance(a[SELF_TAG], TagSackIterator):
        raise TagError('__next__ on non-TagSackIterator')

    return next(a[SELF_TAG])

def _contains(*a) -> Optional[Tag]:
    if not isinstance(a[TAGS][0], TagSack):
        raise TagError('contains on tagged Tensor is prohibited')

    return a[TAGS][1]

# TODO: Handle get multiple items (e.g., a[1:3])
def _getitem(*a) -> Optional[Tag]:
    if not isinstance(a[SELF_TAG], TagSack):
        raise TagError('__getitem__ on tagged Tensor is prohibited')

    elif not a[TAGS][0].is_safe():
        raise TagError('__getitem__ cannot use unsafe tag')

    self_tag = a[SELF_TAG]
    args = a[ARGS]

    return self_tag[args[0]]

# TODO: Handle set multiple items (e.g., a[1:3] = [1, 2, 3])
def _setitem(*a) -> Optional[Tag]:
    if not isinstance(a[SELF_TAG], TagSack):
        raise TagError('__setitem__ on tagged Tensor is prohibited')

    self_tag = a[SELF_TAG]
    args = a[ARGS]
    tags = a[TAGS]

    self_tag[args[0]] = tags[0]

def _call(*a) -> Optional[Tag]:
    method, args, kwargs, self_tag, tags, kwtags = a

    # TODO: release this constraint later
    if method.__self__.__name__.endswith('_'):
        raise TagError('in-place operation is not allowed')

    if isinstance(self_tag, TagSack):
        return DangerTag()

    elif isinstance(self_tag, Tag):

        kwtags = [ Tag(hash(k) ^ t.h, t.m) for k, v in kwtags.items() ]

        tag = Tag.merge(hash(method), [self_tag] + tags + kwtags)
        return tag

    else: # Functions
        return DangerTag()

def _len(*a) -> Optional[Tag]:

    if isinstance(a[SELF_TAG], TagSack):
        return DangerTag() # TODO: check tags

    else:
        tag = Tag.merge(hash(method), [self_tag])
        return tag


rule_table = {
    'getattr':     _getattr,
    'setattr':     _setattr,
    '__iter__':    _iter,
    '__next__':    _next,
    'contains':    _contains,
    '__getitem__': _getitem,
    '__setitem__': _setitem,
    '__call__':    _call,
    '__len__':     _len,
}

def taint_torch(method: Callable,
                args: List[Any], kwargs: Dict[str, Any],
                self_tag: Optional[Union[Tag, TagSack]],
                tags: List[Union[Tag, TagSack]],
                kwtags: Dict[str, Union[Tag, TagSack]]) -> Tag:

    try:
        _rule = rule_table[method.__name__]
    except:
        _rule = _default

    tag = _rule(method, args, kwargs, self_tag, tags, kwtags)

    return (tag if tag else DangerTag())
