import os
import torch
import pickle
import numpy as np
from typing import List, Tuple

from PIL import Image
from numpy import transpose, reshape
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

def get_dataset(data_path, transform=None):

    dataset = Cifar10Dataset(data_path, transform)
    return dataset


def sample_init() -> (torch.Tensor, torch.Tensor):
    pwd = os.environ['PWD']

    data_path  = f'{pwd}/eval-tests/datasets/cifar10'
    dataset = get_dataset(data_path)

    samples, lbls = [], []
    for img, lbl in dataset:
        samples.append(img)
        lbls.append(lbl)

    samples = torch.stack(samples)
    lbls = torch.tensor(lbls)

    return samples, lbls


class Cifar10Dataset(Dataset):
    def __init__(self, data_path: str, transform=None):

        batch_path = [os.path.join(data_path, f'data_batch_{i+1}')
                      for i in range(5)]

        imgs_lbls = sum([_unpickle(i) for i in batch_path],
                        start = [])
        imgs = [transpose(reshape(i[0], (3, 32, 32)), (1, 2, 0))
                for i in imgs_lbls]
        lbls = [i[1] for i in imgs_lbls]

        self.imgs = [ToTensor()(Image.fromarray(img))
                     for img in imgs]
        self.lbls = lbls
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx: int) -> (torch.Tensor, int):
        img, lbl = self.imgs[idx], self.lbls[idx]

        if self.transform:
            img = self.transform(img)

        return img, lbl


def _unpickle(file_path: str) -> List[Tuple[np.ndarray, int]]:
    with open(file_path, 'rb') as fd:
        d = pickle.load(fd, encoding='bytes')

    return list(zip(d[b'data'], d[b'labels']))
