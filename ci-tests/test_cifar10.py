#
# Copyright (c) 2022
#

import pytest
import time
import pprint
from typing import Any
from threading import Thread

import os
import PIL
import torch
import torchvision
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from ci_tests.cifar_module import CIFARModule

from prime.proxy import Proxy, _client
from prime.utils import run_server, kill_server

from tests.common import *


@pytest.fixture(scope='session')
def model(pytestconfig):
    return pytestconfig.getoption('model')

################################################################################
# Init server before starting tests                                            #
################################################################################

def test_init_cifar10Server(model):
    port = os.environ.get('PRIMEPORT', None)

    kill_server()
    run_server(port=port, dn='cifar10', ll='ERROR')

    time.sleep(1)
    if not _client.check_server():
        raise Exception('Server  not running')

    export_f_output(_client)


def test_cifar10(model):

    print(f'test_cifar10({model})')
    model_name = model

    device = initialize()

    samples_d = Proxy('_SAMPLES')
    labels_d  = Proxy('_LABELS')

    model = build_model(model_name)

    max_epochs = 10
    trainer = pl.Trainer(
        default_root_dir=os.path.join('/tmp', model_name),
        gpus=1 if str(device) == 'cuda:0' else 0,
        max_epochs=max_epochs,
        # TODO
        # callbacks=[
        #     ModelCheckpoint(
        #         save_weights_only=True, model='max', monitor='val_acc'
        #     ),
        #     LearningRateMonitor('epoch'),
        # ],
    )
    # trainer.logger._log_graph = True
    # trainer.logger._default_hp_metric = True

    stream_data(samples_d, labels_d, max_epochs)

    res = _client.FitModel(trainer, model,
                           [],
                           {'batch_size': 128},
                           [], {})
    # TODO: Support other arguments
    # res = _client.FitModel(trainer, model,
    #                        [],                                      # d_args
    #                        {'batch_size': 128, 'shuffle': True,     # d_kwargs
    #                         'drop_last': True, 'pin_memory': True,
    #                         'num_workers': 4},
    #                        [], {} # args, kwargs
    #                        )
    if isinstance(res, Exception):
        raise res

    eval_model(trainer, res, samples_d, labels_d)


def initialize() -> Any:
    R('pytorch_lightning.seed_everything')(42)

    # TODO: setattr on trusted library is not permitted
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    device = (R('torch.device')('cuda:0') if R('torch.cuda.is_available')()
              else R('torch.device')('cpu'))

    return device

def build_model(model_name: str) -> pl.LightningModule:
    if model_name == 'googlenet':
        kwargs = {
            'model_hparams'    : {'num_classes': 10, 'act_fn_name': 'relu'},
            'optimizer_name'   : 'Adam',
            'optimizer_hparams': {'lr': 1e-3, 'weight_decay': 1e-4}
        }

    elif model_name == 'resnet':
        kwargs = {
            'model_hparams'    : {
                'num_classes': 10,
                'c_hidden'   : [16, 32, 64],
                'num_blocks' : [3, 3, 3],
                'act_fn_name': 'relu'},
            'optimizer_name'   : 'SGD',
            'optimizer_hparams': {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 1e-4}
        }

    elif model_name == 'densenet':
        kwargs = {
            'model_hparams'    : {
                'num_classes': 10,
                'num_layers' : [6, 6, 6, 6],
                'bn_size'    : 2,
                'growth_rate': 16,
                'act_fn_name': 'relu'},
            'optimizer_name'   : 'Adam',
            'optimizer_hparams': {'lr': 1e-3, 'weight_decay': 1e-4}

        }

    else:
        raise Exception(f'unknown model name: {model_name}')

    model = CIFARModule(model_name=model_name, **kwargs)

    with open(f'{os.environ["PWD"]}/ci-tests/cifar_module.py', 'r') as fd:
        CIFARModule_src = fd.readlines()

    CIFARModule_src = ''.join(CIFARModule_src[4:])
    res = _client.ExportModel('ci_tests.cifar_module.CIFARModule', CIFARModule_src)
    if isinstance(res, Exception):
        raise res

    return model


def stream_data(samples: torch.Tensor, labels: torch.Tensor, max_epoch: int):

    DATA_MEAN = (samples).mean(axis=(0, 2, 3))
    DATA_STD  = (samples).std(axis=(0, 2, 3))

    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(DATA_MEAN, DATA_STD)
        ]
    )

    transforms_in_de = [ train_transform ]
    args             = [ (...,) ]
    kwargs           = [ {} ]

    _client.StreamData(samples, labels, transforms_in_de, args, kwargs,
                       max_epoch)


def eval_model(trainer: pl.Trainer, model: CIFARModule,
               samples: torch.Tensor, labels: torch.Tensor):

    pwd = os.environ['PWD']
    DATASET_PATH = f'{pwd}/ci-tests/cifar_10'

    DATA_MEAN = (samples).mean(axis=(0, 2, 3))
    DATA_STD  = (samples).std(axis=(0, 2, 3))

    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(DATA_MEAN, DATA_STD)])
    test_set = CIFAR10(root=DATASET_PATH, train=False, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False,
                             drop_last=False,
                             num_workers=4)

    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)

    print(f'====[{model.model_name} Test Result]====\n')
    pprint.pprint(test_result)


################################################################################
# Kill server after all tests are completed                                    #
################################################################################

@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    def _kill_server():
        try: kill_server()
        except: pass
    request.addfinalizer(_kill_server)
