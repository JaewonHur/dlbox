import torch

class PrimeDataset(torch.utils.data.Dataset):
    def __init__(self, *tensors: torch.Tensor, transforms=None):
        self.tensors = tensors
        self.transforms = transforms or (lambda x: x)
        self.n = len(tensors)

    def __getitem__(self, index):
        tensors = tuple((self.transforms(tensor[index]) 
                         if i < self.n - 1 else tensor[index])
                        for i, tensor in enumerate(self.tensors))
        
        return tensors

    def __len__(self):
        return len(self.tensors[0])