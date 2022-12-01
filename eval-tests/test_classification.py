#
# Copyright (c) 2022
#

import pytest
import time
import pprint
from typing import Any
from threading import Thread
from torchsummary import summary

import os
import inspect

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import random_split, Dataset, DataLoader

from eval_tests.pl_module import plModule
from eval_tests.datalib import cifar10, utkface, chestxray

from prime.proxy import Proxy, _client
from prime.utils import run_server, kill_server

from tests.common import *

def bprint(s):
    print ('\033[1m' + s + '\033[0m')

@pytest.fixture
def is_vm(pytestconfig):
    return pytestconfig.getoption('--is_vm')

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


#####
class baseDataset(Dataset):
    def __init__(self, samples, labels, transform):
        self.samples = samples
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample, lbl = self.samples[idx], self.labels[idx]

        return sample, lbl


################################################################################
# Init server before starting tests                                            #
################################################################################

def test_init_Server(is_vm, baseline, dataset):
    port = os.environ.get('PRIMEPORT', None)

    if baseline:
        kill_server()

    else:
        if not is_vm:
            kill_server()
            run_server(port=port, dn=dataset, ll='ERROR')

        for i in range(60):
            time.sleep(1)
            if _client.check_server():
                break

        if not _client.check_server():
            raise Exception('Server not running')


def test_classification(is_vm, baseline, dataset, model, max_epochs):
    start = time.time()

    bprint(f'<================================== Evaluating {model} on {dataset} ==================================>')
    model_name = model
    max_epochs = int(max_epochs)

    if baseline:
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        dlib = (cifar10 if dataset == 'cifar10'
                else utkface if dataset == 'utkface'
                else chestxray if dataset == 'chestxray'
                else None)
        assert dlib

        samples_d, labels_d = dlib.sample_init()

    else:
        device = initialize()
        samples_d = Proxy('_SAMPLES')
        labels_d = Proxy('_LABELS')

    model = build_model(model_name, dataset)

    if not baseline:
        plmodule_src = inspect.getsource(plModule)
        res = _client.ExportModel('eval_tests.pl_module.plModule',
                                  plmodule_src)

        if isinstance(res, Exception):
            raise res

    train_transform, test_transform = build_transform(model_name, dataset,
                                                      samples_d, labels_d)

    trainer_kwargs = {
        'default_root_dir': os.path.join('/tmp/{dataset}-{model_name}'),
        'max_epochs': max_epochs,
    }
    if is_vm: 
        trainer_args['gpus'] = (1 if torch.cuda.is_available() else 0)

    trainer = pl.Trainer(
        **trainer_kwargs
    )

    if baseline:
        data_set = baseDataset(samples_d, labels_d, train_transform)
        dataloader = DataLoader(data_set, batch_size=128)

        trainer.fit(model, dataloader)
        fitted_model = model

    else:
        stream_data(samples_d, labels_d,
                    train_transform, max_epochs)

        res = _client.FitModel(trainer, model,
                               [],
                               {'batch_size': 128},
                               [], {'max_epochs': max_epochs})
        if isinstance(res, Exception):
            raise res

        fitted_model = res

    eval_model(trainer, dataset, fitted_model, test_transform)

    end = time.time()
    elapsed_time = end - start

    save_log(dataset, model_name, elapsed_time)


def save_log(dataset, model_name, elapsed_time):
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d-%H%M")

    pwd = os.getcwd()
    os.makedirs(f'{pwd}/eval-logs', exist_ok=True)

    pid = os.getpid()
    with open(f'{pwd}/eval-logs/{dataset}-{model_name}-time.txt', 'a') as fd:
        fd.write(f'[pid] {now}| {elapsed_time}\n')


def eval_model(trainer, dataset, model, test_transform):

    from torch.utils.data import DataLoader

    pwd = os.environ['PWD']
    dataset_path = f'{pwd}/eval-tests/datasets/{dataset}'

    data_set = (cifar10.get_dataset(dataset_path, test_transform) if dataset == 'cifar10'
                else utkface.get_dataset(dataset_path, test_transform) if dataset == 'utkface'
                else chestxray.get_dataset(dataset_path, test_transform))

    len_test = len(data_set) // 5

    _, test_set = random_split(data_set, [len(data_set) - len_test, len_test])

    test_loader = DataLoader(test_set, batch_size=128,
                             shuffle=False, drop_last=False,
                             num_workers=4)

    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    print(f'====[{model.model_name} Test Result]====\n')
    pprint.pprint(test_result)


def stream_data(samples, labels, transform, max_epochs):
    transform_in_de = [ transform ]
    args            = [ (...,) ]
    kwargs          = [ {} ]

    _client.StreamData(samples, labels, transform_in_de, args, kwargs,
                       max_epochs)


def build_transform(model_name, dataset,
                    samples, labels):
    if dataset == 'cifar10':
        CIFAR10_MEAN = samples.mean(axis=(0, 2, 3))
        CIFAR10_STD = samples.std(axis=(0, 2, 3))

        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomApply([
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
            ], 0.7),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])

        test_transform = transforms.Compose([
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])

    elif dataset == 'utkface':
        UTKFACE_MEAN = samples.mean(axis=(0, 2, 3))
        UTKFACE_STD = samples.std(axis=(0, 2, 3))

        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(UTKFACE_MEAN,
                                 UTKFACE_STD)
        ])

        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(UTKFACE_MEAN,
                                 UTKFACE_STD)
        ])

    elif dataset == 'chestxray':
        train_transform = transforms.Compose([
            transforms.RandomApply([
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip()
            ], 0.7),
            transforms.ToTensor()
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    else:
        raise NotImplementedError(f'{dataset} not supported')

    return train_transform, test_transform


def build_model(model_name, dataset):
    hparams = hyperparams(dataset)

    model = plModule(model_name, hparams)
    print(summary(model, input_size=(hparams['input_channels'],
                                     hparams['input_size'],
                                     hparams['input_size']),
                  device='cpu'))

    return model


def hyperparams(dataset):

    if dataset == 'cifar10':
        input_size, input_channels, num_classes = 32, 3, 10
    elif dataset == 'utkface':
        input_size, input_channels, num_classes = 128, 3, 4
    elif dataset == 'chestxray':
        input_size, input_channels, num_classes = 256, 3, 2
    else:
        raise NotImplementedError(f'{dataset} is not supported')

    return {'input_size': input_size,
            'input_channels': input_channels,
            'num_classes': num_classes}


def initialize() -> Any:
    R('pytorch_lightning.seed_everything')(42)

    # TODO: setattr on trusted library is not permitted
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    device = (R('torch.device')('cuda:0') if R('torch.cuda.is_available')()
              else R('torch.device')('cpu'))

    return device


################################################################################
# Kill server after all tests are completed                                    #
################################################################################

@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    def _kill_server():
        try: kill_server()
        except: pass
    request.addfinalizer(_kill_server)
