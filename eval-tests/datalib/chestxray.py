import os
import torch
import pickle
import numpy as np
import pandas  as pd

from PIL import Image
from numpy import transpose
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import Dataset

def get_dataset(data_path, transform=None):

    dataset = ChestXrayDataset(data_path, transform)
    return dataset


def sample_init() -> (torch.Tensor, torch.Tensor):
    pwd = os.environ['PWD']

    data_path  = f'{pwd}/eval-tests/datasets/chestxray'
    dataset = get_dataset(data_path)

    samples, lbls = [], []
    for img, lbl in dataset:
        samples.append(img)
        lbls.append(lbl)

    samples = torch.stack(samples)
    lbls = torch.tensor(lbls)

    return samples, lbls


class ChestXrayDataset(Dataset):
    def __init__(self, data_path: str, transform=None):
        self.train_set = pd.read_csv(
            os.path.join(data_path, 'train_labels.csv')
        )

        self.train_path = os.path.join(data_path, 'train')
        self.resize = Resize((256, 256))
        self.totensor = ToTensor()
        self.transform = transform

        dataset = []
        for i in range(len(self.train_set)):
            fname = self.train_set.iloc[i][0]
            label = self.train_set.iloc[i][1]

            if label == 1: # corona is missing
                dataset += [(fname, 1)] * 8
            else:
                dataset += [(fname, 0)]

        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i: int) -> (Image, int):
        fname, label = self.dataset[i]

        _img = Image.open(os.path.join(self.train_path, fname))
        img = _img.copy()
        _img.close()

        img = self.totensor(self.resize(img))
        if self.transform is not None:
            img = self.transform(img)

        return img, label
