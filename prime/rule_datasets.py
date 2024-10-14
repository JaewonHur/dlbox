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

    raise TagError(f'{method} is not allowed on dataset object')

def _getattr(*a) -> Union[Tag, TagSack]:
    if not a[TAGS][1].is_safe():
        raise TagError('getattr cannot use unsafe tag')

    return a[TAGS][0]

def _setattr(*a) -> Tag:
    raise TagError('setattr on Dataset is prohibited')

def _getitem(*a) -> Union[Tag, TagSack]:
    if not isinstance(a[SELF_TAG], TagSack):
        raise TagError('__getitem__ on tagged object is prohibited')

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
    raise TagError('__setitem__ is not allowed on dataset object')

def _call(*a) -> Tag:
    method, args, kwargs, self_tag, tags, kwtags = a
    method = method.__self__.__name__

    if not self_tag.is_safe():
        raise TagError(f'calling {method} is not allowed on unsafe object')

    # TODO: return SafeTag on safe operations
    if isinstance(self_tag, TagSack):
        if method in ('rename_column', 'set_format', 'remove_columns'):
            return self_tag

        elif method in ('map'):
            kwtags = [Tag(hash(k) ^ t.h, t.m) for k, t in kwtags.items()]
            for i in range(len(self_tag)):
                self_tag[i] = Tag.merge(hash(method), [self_tag[i]] + tags + kwtags)

            return deepcopy(self_tag)

        else:
            raise TagError(f'{method} is not allowed on dataset TagSack')

    elif isinstance(self_tag, Tag):
        raise TagError('{method} is not allowed on dataset Tag')

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
    '__getitem__':  _getitem,
    '__setitem__':  _setitem,
    '__call__':     _call,
    '__len__':      _len,
}

def taint_datasets(method: Callable,
                   args: List[Any], kwargs: Dict[str, Any],
                   self_tag: Optional[Union[Tag, TagSack]],
                   tags: List[Union[Tag, TagSack]],
                   kwtags: Dict[str, Union[Tag, TagSack]]) -> Tag:

    _rule = (rule_table[method.__name__] if method.__name__ in rule_table
             else _default)

    tag = _rule(method, args, kwargs, self_tag, tags, kwtags)

    return (tag if tag else DangerTag())
