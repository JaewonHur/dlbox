#
# Copyright (c) 2022
#

from typing import Dict, Tuple, List

from torch import Tensor
from torch.utils.data import DataLoader, Dataset

def FairDataset(Dataset):
    def __init__(self, epochs: Dict[int, Tuple[List[str], List[str]]]):
        self.epochs = epochs
        self.ctr = 0

    def __getitem__(self, idx):
        if self.ctr == len(self.epochs):
            raise IndexError()
        else:
            sample, label = self.epochs[self.ctr][idx]

            return (sample, label)

def build_dataloader(tagged_epochs: Dict[int, Tuple[List[Tuple[str, Tensor]],
                                                    List[Tuple[str, Tensor]]]],
                     args: List, kwargs: Dict) -> DataLoader:

    epochs = {}
    for k, v in tagged_epochs.items():
        t_samples = [ s[0] for s in v[0] ]
        t_labels = [ l[0] for l in v[1] ]

        # TODO: Sanitize dataflow in t_samples, t_labels

        samples = [ s[1] for s in v[0] ]
        labels = [ l[1] for l in v[1] ]

        epochs[k] = (samples, labels)

    dataset = FairDataset(epochs)
    dataloader = DataLoader(dataset, *args, **kwargs)

    return dataloader
