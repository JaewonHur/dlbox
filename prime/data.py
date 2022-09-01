#
# Copyright (c) 2022
#

import queue
from typing import Dict, Tuple, List

from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# class FairDataset(Dataset):
#     def __init__(self, epoch: Tuple[List[Tensor], List[Tensor]]):
#         super().__init__()

#         self.samples, self.labels = epoch
#         self.ctr = 0

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         return (self.samples[idx], self.labels[idx])


def build_dataloader(dqueue: queue.Queue) -> DataLoader:
    raise NotImplementedError()

# def build_dataloader(tagged_epoch: Tuple[List[Tuple[str, Tensor]],
#                                          List[Tuple[str, Tensor]]],
#                      args: List, kwargs: Dict) -> DataLoader:

#     tag_s = [ i[0] for i in tagged_epoch[0] ]
#     tag_l = [ i[0] for i in tagged_epoch[1] ]

#     samples = [ i[1] for i in tagged_epoch[0] ]
#     labels = [ i[1] for i in tagged_epoch[1] ]

#     # TODO: Check tags

#     epoch = (samples, labels)
#     dataset = FairDataset(epoch)
#     dataloader = DataLoader(dataset, *args, **kwargs)

#     return dataloader
