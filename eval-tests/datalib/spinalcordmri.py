import os

from medicaltorch import datasets as mt_datasets
from medicaltorch import transforms as mt_transforms
from medicaltorch import filters as mt_filters

def get_dataset(data_path, transform=None):

    transform = mt_transforms.ToTensor()
    gmdataset_train = mt_datasets.SCGMChallenge2DTrain(root_dir=data_path,
                                                       subj_ids=range(1, 9),
                                                       transform=transform,
                                                       slice_filter_fn=mt_filters.SliceFilter())
    return gmdataset_train


def sample_init(): #-> (torch.Tensor, torch.Tensor):
    pwd = os.environ['PWD']

    data_path = f'{pwd}/eval-tests/datasets/spinalcordmri'
    dataset = get_dataset(data_path)

    samples = [data_dict for data_dict in dataset]
    lbls = samples

    return samples, lbls
