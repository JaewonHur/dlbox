#
# Copyright (c) 2022
#
from __future__ import annotations

from typing import Any, Callable, Union, List, Optional
from types import FunctionType

from prime.taint import *

def taint_default(method: Callable,
                  self_tag: Optional[Union[Tag, TagSack]],
                  tags: List[Union[Tag, TagSack]],
                  kwtags: Dict[str, Union[Tag, TagSack]]) -> Tag:

    if (not all(isinstance(t, Tag) for t in tags) or
        not all(isinstance(t, Tag) for t in kwtags.values())):
        raise TagError('{method} cannot receive TagSack')

    self_tag = [ self_tag ] if self_tag else []
    kwtags = [ Tag(hash(k) ^ t.h, t.m) for k, v in kwtags.items() ]

    return Tag.merge(hash(method), self_tag + tags + kwtags)

def taint_torch(method: Callable, _self: Any,
                tags: List[Union[Tag, TagSack]],
                kwtags: Dict[str, Union[Tag, TagSack]]) -> Tag:

    return DangerTag()


taint_rules: Dict[str, FunctionType] = {
    'default': taint_default,
    'torch': taint_torch,
}
