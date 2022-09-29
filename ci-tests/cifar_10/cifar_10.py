from __future__ import annotations

import os
import torch

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


def sample_init() -> (torch.Tensor, torch.Tensor):
    pwd = os.environ['PWD']
    root = f'{pwd}/ci-tests/cifar_10'

    dataset = CIFAR10(root)

    totensor = ToTensor()
    samples = [ totensor(s) for s in dataset.data ]

    samples = torch.stack(samples)
    labels  = torch.tensor(dataset.targets)

    return (samples, labels)
