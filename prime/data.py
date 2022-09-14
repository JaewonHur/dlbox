#
# Copyright (c) 2022
#

import queue
from threading import Thread
from typing import Dict, Tuple, List, Any, Callable
from collections import namedtuple
from functools import reduce, partial

from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from prime.taint import Tag, TagSack, TagError

DataPair = namedtuple('DataPair', ['sample', 'label'])
Sample   = namedtuple('Sample',   ['tag', 'val'])
Label    = namedtuple('Label',    ['tag', 'val'])

class bind(partial):
    """
    An improved version of partial which accepts Ellipsis (...) as a placeholder
    """
    def __init__(self, *args, **keywords):
        assert [*self.args, *self.keywords.values()].count(...) == 1

    def __call__(self, o: Any):

        args = (o if i is ... else i for i in self.args)
        keywords = {k:(o if v is ... else v)
                    for k, v in self.keywords.items()}

        return self.func(*args, **keywords)


class DataQueue():
    def __init__(self):
        self.q = queue.Queue()
        self.is_streaming = False

    def empty(self) -> bool:
        return self.q.empty()

    def get(self) -> DataPair:
        return self.q.get()

    def put(self, pairs: List[Tuple[Tuple[Tag, Tensor], Tuple[Tag, Tensor]]]) -> int:
        for p in pairs:
            self.q.put(DataPair(Sample(*p[0]), Label(*p[1])))

        return self.q.qsize()

    # TODO: Handles transforms on labels also
    def stream(self, s_ts: TagSack, samples: Tensor, l_ts: TagSack, labels: Tensor,
               transforms: List[Callable], args: List[Tuple], kwargs: List[Dict],
               max_epoch: int):

        assert len(s_ts) == len(samples)
        assert len(l_ts) == len(labels)
        assert len(s_ts) == len(l_ts)

        self.is_streaming = True

        transforms = [ bind(t, *a, **kw)
                       for t, a, kw in zip(transforms, args, kwargs) ]

        thd = Thread(target=self._stream,
                     args=(s_ts, samples, l_ts, labels, transforms, max_epoch))
        thd.start()

    def _stream(self, _s_ts: TagSack, _samples: Tensor,
                _l_ts: TagSack, _labels: Tensor,
                transforms: List[Callable],
                max_epoch: int):

        for i in range(max_epoch):
            for s_t, s, l_t, l in zip(_s_ts, _samples, _l_ts, _labels):
                s = reduce(lambda x, y: y(x), transforms, s)

                p = DataPair(Sample(s_t, s), Label(l_t, l))
                self.q.put(p)

        self.is_streaming = False


class FairDataset(Dataset):
    N: int = None

    def __init__(self, dqueue: DataQueue):
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
