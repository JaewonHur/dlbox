#
# Copyright (c) 2022
#

import os
import pytest
import time

from torchsummary import summary

from prime.proxy import Proxy, _client
from prime.utils import run_server, kill_server

@pytest.fixture
def baseline(pytestconfig):
    return pytestconfig.getoption('--baseline')

@pytest.fixture
def dataset(pytestconfig):
    return pytestconfig.getoption('--dataset')

@pytest.fixture
def model(pytestconfig):
    return pytestconfig.getoption('--model')

@pytest.fixture
def max_epochs(pytestconfig):
    return pytestconfig.getoption('--max_epochs')


################################################################################
# Init server before starting tests                                            #
################################################################################

def test_init_server(baseline: bool, dataset: str):
    if baseline:
        kill_server()

    elif 'PRIMEIPADDR' in os.environ and 'PRIMEPORT' in os.environ:
        kill_server()
        time.sleep(1)
        
        if not _client.check_server():
            raise Exception('Server not running')
    
    else:
        kill_server()
        run_server(dn=dataset, ll='ERROR')
        
        time.sleep(1)
        if not _client.check_server():
            raise Exception('Server not running')

################################################################################

def import_libs(baseline: bool):
    global PIL, torch, torchvision
    global PrimeDataset, DataLoader, Trainer
    
    if baseline:
        import PIL, torch, torchvision, pytorch_lightning
        
        from torch.utils.data import TensorDataset, DataLoader
        from pytorch_lightning import Trainer
        
        class PrimeDataset(TensorDataset):
            def __init__(self, *tensors, transforms):
                super().__init__(*tensors)
                self.transforms = transforms
                self.n = len(tensors)
                
            def __getitem__(self, index):
                tensors = tuple((self.transforms(t[index])
                                 if i < self.n - 1 else t[index])
                                for i, t in enumerate(self.tensors))

                return tensors
            
    else:
        import prime_PIL as PIL
        import prime_torch as torch
        import prime_torchvision as torchvision
        
        from prime_torch.utils.data import PrimeDataset, DataLoader
        from prime_pytorch_lightning import Trainer


def sample_init(baseline: bool, dataset: str) -> tuple:

    if baseline:
        from eval_tests.datalib import cifar10, utkface, chestxray

        dlib = (cifar10 if dataset == 'cifar10'
                else utkface if dataset == 'utkface'
                else chestxray if dataset == 'chestxray'
                else None)
        assert dlib

        samples, labels = dlib.sample_init()

    else:
        samples, labels = Proxy('_SAMPLES'), Proxy('_LABELS')

    return samples, labels


def build_transforms(dataset: str, model_name: str, samples, labels):
    if dataset == 'cifar10':
        CIFAR10_MEAN = samples.mean(axis=(0, 2, 3))
        CIFAR10_STD = samples.std(axis=(0, 2, 3))

        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomApply([
                torchvision.transforms.RandomRotation(10),
                torchvision.transforms.RandomHorizontalFlip(),
            ], 0.7),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])

        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])

    elif dataset == 'utkface':
        UTKFACE_MEAN = samples.mean(axis=(0, 2, 3))
        UTKFACE_STD = samples.std(axis=(0, 2, 3))

        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(UTKFACE_MEAN,
                                 UTKFACE_STD)
        ])

        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(UTKFACE_MEAN,
                                 UTKFACE_STD)
        ])

    elif dataset == 'chestxray':
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomApply([
                torchvision.transforms.RandomRotation(10),
                torchvision.transforms.RandomHorizontalFlip()
            ], 0.7),
            torchvision.transforms.ToTensor()
        ])

        test_transform = torchvision.transforms.Compose([
        ])

    else:
        raise NotImplementedError(f'{dataset} not supported')

    return train_transform, test_transform


def hyperparams(dataset: str):

    if dataset == 'cifar10':
        input_size, input_channels, num_classes = 32, 3, 10
    elif dataset == 'utkface':
        input_size, input_channels, num_classes = 128, 3, 4
    elif dataset == 'chestxray':
        input_size, input_channels, num_classes = 256, 3, 2
    elif dataset == 'spinalcordmri':
        input_size, input_channels, num_classes = 192, 1, 1
    else:
        raise NotImplementedError(f'{dataset} is not supported')

    return {'input_size': input_size,
            'input_channels': input_channels,
            'num_classes': num_classes}


def build_model(dataset: str, model_name: str):

    from eval_tests.clsmodule import ClassificationModule

    hparams = hyperparams(dataset)

    model = ClassificationModule(model_name, hparams)
    print(summary(model, input_size=(hparams['input_channels'],
                                     hparams['input_size'],
                                     hparams['input_size']),
                  device='cpu'))

    return model

    
def eval_model(dataset, model, transforms):
    import torch
    from torch.utils.data import random_split, DataLoader
    from pytorch_lightning import Trainer

    from eval_tests.datalib import cifar10, utkface, chestxray

    dlib = (cifar10 if dataset == 'cifar10'
            else utkface if dataset == 'utkface'
            else chestxray if dataset == 'chestxray'
            else None)
    assert dlib

    data_set = dlib.get_dataset(transforms)
    ntest = len(data_set) // 10

    _, test_set = random_split(data_set, [len(data_set) - ntest, ntest])

    loader = DataLoader(test_set, batch_size=128, shuffle=False, drop_last=False)
    trainer = Trainer()
    
    trainer.test(model, dataloaders=loader, verbose=True)



def test_classification(baseline: bool, dataset: str, model: str, max_epochs: str):
    dataset_name = dataset
    model_name = model
    max_epochs = int(max_epochs)

    import_libs(baseline)
    samples, labels = sample_init(baseline, dataset)
    
    print(f'\n[{dataset_name},{model_name}] start running...')
    start = time.time()

    print(f'\n[{dataset_name},{model_name}] build transforms...')
    train_transforms, test_transforms = build_transforms(dataset_name, model_name,
                                                         samples, labels)

    print(f'\n[{dataset_name},{model_name}] fit model...')

    dataset = PrimeDataset(samples, labels, transforms=train_transforms)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    trainer = Trainer(gpus=1, max_epochs=max_epochs)
    model = build_model(dataset_name, model_name)

    if baseline:
        torch.set_num_threads(1)

    trainer.fit(model, train_dataloaders=loader)
    
    end = time.time()
    print(f'[{dataset_name},{model_name}] done, elapsed: {end-start:.2f}')

    # if not baseline:
    #     test_transforms = test_transforms.obj
        
    # eval_model(dataset_name, model, test_transforms)


################################################################################
# Kill server after all tests are completed                                    #
################################################################################

@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    def _kill_server():
        try: kill_server()
        except: pass
    request.addfinalizer(_kill_server)
