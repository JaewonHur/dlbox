#
# Copyright (c) 2022
#

import time
import pytest

from prime.proxy import Proxy, _client
from prime.utils import run_server, kill_server

@pytest.fixture
def baseline(pytestconfig):
    return pytestconfig.getoption('--baseline')

@pytest.fixture
def model(pytestconfig):
    return pytestconfig.getoption('--model')

################################################################################
# Init server before starting tests                                            #
################################################################################

def test_init_server(baseline: bool):
    if baseline:
        kill_server()
    
    else:
        kill_server()
        run_server(dn='cifar10', ll='ERROR')
        
        time.sleep(1)
        if not _client.check_server():
            raise Exception('Server not running')

################################################################################

def import_libs(baseline: bool):
    global PIL, torch, torchvision, pytorch_lightning
    global PrimeDataset, DataLoader
    
    if baseline:
        import PIL, torch, torchvision, pytorch_lightning
        
        from torch.utils.data import TensorDataset, DataLoader
        
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
        import prime_pytorch_lightning as pytorch_lightning
        
        from prime_torch.utils.data import PrimeDataset, DataLoader


def sample_init(baseline: bool) -> tuple:
    
    if baseline:
        from ci_tests.cifar10 import cifar10
        samples, labels = cifar10.sample_init()
        
    else:
        samples, labels = Proxy('_SAMPLES'), Proxy('_LABELS')
        
    return samples, labels

def build_model(model_name: str):
    from ci_tests.cifarmodule import CifarModule

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

    model = CifarModule(model_name=model_name, **kwargs)
    return model

    
def eval_model(model, transforms):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from pytorch_lightning import Trainer
    
    from ci_tests.cifar10 import cifar10
    
    samples, labels = cifar10.sample_init()

    nsamples = len(samples)
    samples = samples[::nsamples//10]
    labels = labels[::nsamples//10]
    
    samples = torch.stack([transforms(s) for s in samples])
    
    loader = DataLoader(TensorDataset(samples, labels))
    trainer = Trainer()
    
    trainer.test(model, dataloaders=loader, verbose=True)


def test_cifar10(baseline: bool, model: str):

    model_name = model
    
    import_libs(baseline)
    samples, labels = sample_init(baseline)
    
    print(f'\n[cifar10,{model_name}] start running..')
    start = time.time()

    DATA_MEAN = samples.mean(axis=(0, 2, 3))
    DATA_STD  = samples.std(axis=(0, 2, 3))
    
    print(f'[cifar10,{model_name}] data_mean: {DATA_MEAN}, data_std: {DATA_STD}')

    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(DATA_MEAN, DATA_STD)
        ]
    )

    print(f'[cifar10,{model_name}] fit model...')

    dataset = PrimeDataset(samples, labels, transforms=train_transforms)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    trainer = pytorch_lightning.Trainer(gpus=1, max_epochs=2)
    model = build_model(model_name)

    trainer.fit(model, train_dataloaders=loader)

    end = time.time()
    print(f'[cifar10,{model_name}] done, elapsed: {end-start:.2f}')

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(DATA_MEAN, DATA_STD)
    ])

    if not baseline:
        test_transforms = test_transforms.obj

    eval_model(model, test_transforms)


################################################################################
# Kill server after all tests are completed                                    #
################################################################################

@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    def _kill_server():
        try: kill_server()
        except: pass
    request.addfinalizer(_kill_server)
