#
# Copyright (c) 2022
#
from __future__ import annotations

from typing import Any, Callable, Union, List, Optional, Dict
from types import FunctionType

from prime.taint import *
from prime.rule_torch import taint_torch
from prime.rule_datasets import taint_datasets

def taint_default(method: Callable,
                  args: List[Any],
                  kwargs: Dict[str, Any],
                  self_tag: Optional[Union[Tag, TagSack]],
                  tags: List[Union[Tag, TagSack]],
                  kwtags: Dict[str, Union[Tag, TagSack]]) -> Tag:

    if (not all(isinstance(t, Tag) for t in tags) or
        not all(isinstance(t, Tag) for t in kwtags.values())):
        raise TagError(f'{method} cannot receive TagSack(Iterator)')

    self_tag = [ self_tag ] if self_tag else []
    kwtags = [ Tag(hash(k) ^ t.h, t.m) for k, t in kwtags.items() ]

    # TODO: set_danger used tags

    return Tag.merge(hash(str(method)), self_tag + tags + kwtags)


####################### Taint Rules ############################################
taint_rules: Dict[str, FunctionType] = {
    'default': taint_default,
    'torch':   taint_torch,
    'datasets': taint_datasets,
}
