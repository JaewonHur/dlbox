#
# Copyright (c) 2022
#

import pytest
import time
import pprint
from typing import Any
from threading import Thread

import os
import inspect

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import random_split

from eval_tests.pl_module import plModule
from eval_tests.datalib import cifar10, utkface, chestxray

from prime.proxy import Proxy, _client
from prime.utils import run_server, kill_server

from tests.common import *

def bprint(s):
    print ('\033[1m' + s + '\033[0m')

@pytest.fixture
def dataset(pytestconfig):
    return pytestconfig.getoption('--dataset')

@pytest.fixture
def model(pytestconfig):
    return pytestconfig.getoption('--model')


################################################################################
# Init server before starting tests                                            #
################################################################################

def test_init_Server(dataset):
    port = os.environ.get('PRIMEPORT', None)

    kill_server()
    run_server(port=port, ci=dataset, ll='ERROR')

    time.sleep(1)
    if not _client.check_server():
        raise Exception('Server not running')


def test_classification(dataset, model):
    bprint(f'<================================== Evaluating {model} on {dataset} ==================================>')
    model_name = model

    device = initialize()

    samples_d = Proxy('_SAMPLES')
    labels_d = Proxy('_LABELS')

    model = build_model(model_name, dataset)
    train_transform, test_transform = build_transform(model_name, dataset,
                                                      samples_d, labels_d)

    max_epochs = 1
    trainer = pl.Trainer(
        default_root_dir=os.path.join('/tmp/{dataset}-{model_name}'),
        gpus=1 if str(device) == 'cuda:0' else 0,
        max_epochs=max_epochs
    )

    stream_data(samples_d, labels_d,
                train_transform, max_epochs)

    res = _client.FitModel(trainer, model,
                           [],
                           {'batch_size': 128},
                           [], {})
    if isinstance(res, Exception):
        raise res

    eval_model(trainer, dataset, model, test_transform)


def eval_model(trainer, dataset, model, test_transform):

    from torch.utils.data import DataLoader

    pwd = os.environ['PWD']
    dataset_path = f'{pwd}/eval-tests/datasets/{dataset}'

    data_set = (cifar10.get_dataset(dataset_path, test_transform) if dataset == 'cifar10'
                else None)
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
        raise NotImplementedError()

    elif dataset == 'chestxray':
        raise NotImplementedError()

    else:
        raise NotImplementedError(f'{dataset} not supported')

    return train_transform, test_transform


def build_model(model_name, dataset):
    hparams = hyperparams(dataset)

    plmodule_src = inspect.getsource(plModule)
    res = _client.ExportModel('eval_tests.pl_module.plModule',
                              plmodule_src)

    if isinstance(res, Exception):
        raise res

    model = plModule(model_name, hparams)
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
