#
# Copyright (c) 2022
#

import os
import time
import pytest

from prime.proxy import Proxy, _client
from prime.utils import run_server, kill_server


@pytest.fixture
def baseline(pytestconfig):
    return pytestconfig.getoption('--baseline')

################################################################################
# Init server before starting tests                                            #
################################################################################

def test_init_server(baseline):
    if baseline:
        kill_server()
    
    else:
        kill_server()
        run_server(dn='mnist', ll='ERROR')
        
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
        from ci_tests.mnist import mnist
        samples, labels = mnist.sample_init()

    else:
        samples, labels = Proxy('_SAMPLES'), Proxy('_LABELS')

    return samples, labels
                    

def eval_model(model, transforms):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from pytorch_lightning import Trainer

    from ci_tests.mnist import mnist

    samples, labels = mnist.sample_init()
    
    nsamples = len(samples)
    samples = samples[::nsamples//10]
    labels = labels[::nsamples//10]

    samples = torch.cat([transforms(s) for s in samples])

    loader = DataLoader(TensorDataset(samples, labels))
    trainer = Trainer()

    trainer.test(model, dataloaders=loader, verbose=True)
    
def test_mnist(baseline: bool):

    from litclassifier import LitClassifier

    import_libs(baseline)
    samples, labels = sample_init(baseline)

    print(f'\n[mnist] start running...')
    start = time.time()

    DATA_MEAN = (samples / 255.0).mean(axis=(0, 1, 2))
    DATA_STD = (samples / 255.0).std(axis=(0, 1, 2))

    print(f'[mnist] data_mean: {DATA_MEAN}, data_std: {DATA_STD}')

    transforms = torchvision.transforms.Compose([
        torch.Tensor.numpy,
        PIL.Image.fromarray,
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(DATA_MEAN, DATA_STD)
    ])

    print(f'[mnist] fit model...')
    
    dataset = PrimeDataset(samples, labels, transforms=transforms)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    trainer = pytorch_lightning.Trainer(gpus=1, max_epochs=2)
    model = LitClassifier()

    trainer.fit(model, train_dataloaders=loader)

    end = time.time()
    print(f'[mnist] done, elapsed: {end-start:.2f}')

    if not baseline:
        transforms = transforms.obj

    eval_model(model, transforms)


################################################################################
# Kill server after all tests are completed                                    #
################################################################################

@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    def _kill_server():
        try: kill_server()
        except: pass
    request.addfinalizer(_kill_server)
