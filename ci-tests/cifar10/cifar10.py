from __future__ import annotations

import os
import torch
from os.path import abspath, dirname

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


def sample_init() -> (torch.Tensor, torch.Tensor):
    pwd = dirname(abspath(__file__))

    dataset = CIFAR10(pwd)

    totensor = ToTensor()
    samples = [ totensor(s) for s in dataset.data ]

    samples = torch.stack(samples)
    labels  = torch.tensor(dataset.targets)

    return (samples, labels)
