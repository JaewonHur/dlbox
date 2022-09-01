#
# Copyright (c) 2022
#

import queue
from typing import Dict, Tuple, List
from collections import namedtuple

from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from prime.taint import Tag

DataPair = namedtuple('DataPair', ['sample', 'label'])
Sample   = namedtuple('Sample',   ['tag', 'value'])
Label    = namedtuple('Label',    ['tag', 'value'])

def build_dataloader(dqueue: queue.Queue) -> DataLoader:
    raise NotImplementedError()
