#
# Copyright (c) 2022
#

import pytest
import time
from threading import Thread

import PIL
import torch
import torchvision
import pytorch_lightning
from torch.utils.data import Dataset, DataLoader

from prime.proxy import Proxy, _client
from prime.utils import run_server, kill_server

from tests.common import *


################################################################################
# Init server before starting tests                                            #
################################################################################

STOPPED = False

def test_init_mnistServer():
    kill_server()
    run_server(port=None, ci='mnist', ll='ERROR')

    time.sleep(1)
    if not _client.check_server():
        raise Exception('Server  not running')

    export_f_output(_client)


def test_mnist():

    samples_d = Proxy('_SAMPLES')
    labels_d = Proxy('_LABELS')

    model_src = """
class LitClassifier(pytorch_lightning.LightningModule):
    def __init__(self, hidden_dim: int = 128, learning_rate: float = 0.0001):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = torch.nn.Linear(28 * 28, self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = torch.nn.functional.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)

        acc = self.accuracy(probs, y)
        return acc


    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        acc = self.accuracy(logits, y)
        return acc

    def accuracy(self, logits, y):
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / len(y)

        return acc

    def validation_epoch_end(self, outputs) -> None:
        self.log('val_acc', torch.stack(outputs).mean(), prog_bar=True)

    def test_epoch_end(self, outputs) -> None:
        self.log('test_acc', torch.stack(outputs).mean())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    """

    ref = _client.ExportModel('test_mnist.LitClassifier', model_src)
    assert ref == 'LitClassifier'

    model = LitClassifier()

    max_epoch = 2
    trainer = pytorch_lightning.Trainer(max_epochs=max_epoch)

    # data_thread = Thread(target=supply_data,
    #                      args=(samples_d, labels_d))

    # data_thread.start()

    stream_data(samples_d, labels_d, max_epoch)
    res = _client.FitModel(trainer, model,
                           [], {'batch_size': 32}, [], {})
    if isinstance(res, Exception):
        raise res

    global STOPPED
    STOPPED = True
    # data_thread.join()

    test_data = TestDataset()
    trainer.test(res, dataloaders=DataLoader(test_data, batch_size=32))

def supply_data(samples: torch.Tensor, labels: torch.Tensor):

    N = len(samples)
    DATA_MEAN = (samples / 255.0).mean(axis=(0, 1, 2))
    DATA_STD = (samples / 255.0).std(axis=(0, 1, 2))

    print(f'[mnist] DATA_MEAN: {DATA_MEAN}')
    print(f'[mnist] DATA_STD: {DATA_STD}')

    transform = Proxy(_client.AllocateObj(torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(DATA_MEAN, DATA_STD)
    ])))

    batch_size = 64
    i = 0

    buf = []
    while not STOPPED:
        sample, label = samples[i%N], labels[i%N]

        img = R('PIL.Image.fromarray')(sample.numpy(), mode='L')
        img = transform(img)

        buf.append((img, label))

        if i % batch_size == batch_size - 1:
            qsize = _client.SupplyData(buf)
            buf.clear()

        i += 1

def stream_data(samples: torch.Tensor, labels: torch.Tensor, max_epoch: int):

    DATA_MEAN = (samples / 255.0).mean(axis=(0, 1, 2))
    DATA_STD = (samples / 255.0).std(axis=(0, 1, 2))

    print(f'[mnist] DATA_MEAN: {DATA_MEAN}')
    print(f'[mnist] DATA_STD: {DATA_STD}')

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(DATA_MEAN, DATA_STD)
    ])
    toimg = 'PIL.Image.fromarray'
    tonumpy = 'torch.Tensor.numpy'

    transforms = [ tonumpy, toimg, transform ]
    args = [ (...,), (...,), (...,) ]
    kwargs = [ {}, {'mode': 'L'}, {} ]

    _client.StreamData(samples, labels, transforms, args, kwargs, max_epoch)



class TestDataset(Dataset):
    def __init__(self):
        super().__init__()

        from mnist import mnist
        self.samples, self.labels = mnist.test_init()

        DATA_MEAN = (self.samples / 255.0).mean(axis=(0, 1, 2))
        DATA_STD = (self.samples / 255.0).std(axis=(0, 1, 2))

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(DATA_MEAN, DATA_STD)
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample, label = self.samples[idx], self.labels[idx]

        img = PIL.Image.fromarray(sample.numpy(), mode='L')
        img = self.transform(img)

        return img, label


class LitClassifier(pytorch_lightning.LightningModule):
    def __init__(self, hidden_dim: int = 128, learning_rate: float = 0.0001):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = torch.nn.Linear(28 * 28, self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = torch.nn.functional.F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)

        acc = self.accuracy(probs, y)
        return acc


    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        acc = self.accuracy(logits, y)
        return acc

    def accuracy(self, logits, y):
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / len(y)

        return acc

    def validation_epoch_end(self, outputs) -> None:
        self.log('val_acc', torch.stack(outputs).mean(), prog_bar=True)

    def test_epoch_end(self, outputs) -> None:
        self.log('test_acc', torch.stack(outputs).mean())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


################################################################################
# Kill server after all tests are completed                                    #
################################################################################

@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    def _kill_server():
        try: kill_server()
        except: pass
    request.addfinalizer(_kill_server)
