import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor
from PIL import Image


def get_dataset(data_path, transform=None):

    dataset = UTKFaceDataset(data_path, transform)
    return dataset


def sample_init() -> (torch.Tensor, torch.Tensor):
    pwd = os.environ['PWD']

    data_path  = f'{pwd}/eval-tests/datasets/utkface'
    dataset = get_dataset(data_path)

    samples, lbls = [], []
    for img, lbl in dataset:
        samples.append(img)
        lbls.append(lbl)

    samples = torch.stack(samples)
    lbls = torch.tensor(lbls)

    return samples, lbls


class UTKFaceDataset(Dataset):
    def __init__(self, data_path: str, transform=None):
        self.data_path = data_path
        self.transform = transform

        files = [ f for f in os.listdir(self.data_path)
                  if f[:8] == 'landmark' ]

        self.totensor = ToTensor()
        self.lines = []
        for txt_file in files:
            txt_file_path = os.path.join(self.data_path, txt_file)
            with open(txt_file_path, 'r') as f:
                assert f is not None
                for i in f:
                    image_name = i.split('jpg ')[0]
                    attrs = image_name.split('_')
                    # Remove non-primary races
                    if len(attrs) < 4 or int(attrs[2]) >= 4  or '' in attrs:
                        continue
                    self.lines.append(image_name+'jpg')

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index:int) -> (torch.Tensor, int):
        attrs = self.lines[index].split('_')
        race = int(attrs[2])

        image_path = os.path.join(self.data_path, 'data',
                                  self.lines[index]+'.chip.jpg').rstrip()

        image = Image.open(image_path).convert('RGB')
        target = race

        image = self.totensor(image)
        if self.transform:
            image = self.transform(image)

        return image, target
