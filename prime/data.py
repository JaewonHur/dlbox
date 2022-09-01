#
# Copyright (c) 2022
#

import queue
from typing import Dict, Tuple, List, Any
from collections import namedtuple

from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from prime.taint import Tag, TagError

DataPair = namedtuple('DataPair', ['sample', 'label'])
Sample   = namedtuple('Sample',   ['tag', 'val'])
Label    = namedtuple('Label',    ['tag', 'val'])

class FairDataset(Dataset):
    N: int = None

    def __init__(self, dqueue: queue.Queue):
        super().__init__()

        self.dqueue = dqueue
        self.h = None

    @classmethod
    def set_n(cls, N: int):
        cls.N = N

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> Tuple[Tensor]:
        pair = self.dqueue.get()

        s_tag, l_tag  = pair.sample.tag, pair.label.tag
        sample, label = pair.sample.val, pair.label.val

        self.sanitize(idx, s_tag, l_tag)

        return (sample, label)

    def sanitize(self, idx: int, s_tag: Tag, l_tag: Tag):
        # TODO: Check label tag also

        if self.h is None:
            self.h = s_tag.h

        if not s_tag.m.has_only(idx):
            raise TagError(f'tag does not match idx\n[{idx}] tag: {tag}, idx: {idx}')
        elif not s_tag.h == self.h:
            raise TagError(f'tag does not match has\n[{idx}] tag: {tag}, h: {hex(h)[0:5]}')


def build_dataloader(dqueue: queue.Queue,
                     d_args: List[Any], d_kwargs: Dict[str,Any]) -> DataLoader:

    dataset = FairDataset(dqueue)
    dataloader = DataLoader(dataset, *d_args, **d_kwargs)

    return dataloader
