import torch
import pytorch_lightning
import time

import torchvision

############################## Model Definitions ###############################


class plModule(pytorch_lightning.LightningModule):
    def __init__(self, model_name, hparams):
        super().__init__()

        self.model_name = model_name
        if model_name == 'simplefc':
            self.model = FC(**hparams)

        elif model_name == 'simplecnn':
            self.model = CNN(**hparams)

        elif model_name == 'resnet':
            model = torchvision.models.resnet18(num_classes=hparams['num_classes'])
            model = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=256//hparams['input_size'],
                                  mode='bilinear'),
                model
            )
            self.model = model

        elif model_name == 'mobilenet':
            model = torchvision.models.mobilenet_v2(num_classes=hparams['num_classes'])
            model = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=256//hparams['input_size'],
                                  mode='bilinear'),
                model
            )
            self.model = model

        else:
            raise NotImplementedError(f'{model_name} not supported')

        self.loss_fun = torch.nn.CrossEntropyLoss()
        self.start = time.time()
        self.epoch = 1

    def forward(self, imgs):
        return self.model(imgs)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=1e-3, weight_decay=1e-4) # TODO

        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
        #                                                  milestopes=[100, 150],
        #                                                  gamma=0.1)

        return [optimizer], []

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_fun(preds, labels)

        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()

        self.log('val_acc', acc)


    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()

        self.log('test_acc', acc)

    def training_epoch_end(self, outputs):
        loss = sum(outputs['loss'] for o in outputs) / len(outputs)
        elapsed_time = int(time.time() - self.start)

        print(f'[{elapsed_time}] Epoch {self.epoch} #### loss: {loss}')
        self.epoch += 1


class FC(torch.nn.Module):
    def __init__(self, input_size=64, input_channels=3, num_classes=10):
        super(FC, self).__init__()

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_channels*input_size*input_size, num_classes)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CNN(torch.nn.Module):
    def __init__(self, input_size=64, input_channels=3, num_classes=10):
        super(CNN, self).__init__()

        assert (input_size & (input_size - 1) == 0) and input_size != 0

        import math
        depth = int(math.log2((input_size + 2) / 8))

        channels = [input_channels] + [32 * pow(2,d)
                                       for d in range(depth)]

        layers = []
        for i, o in zip(channels, channels[1:]):
            layers += [ torch.nn.Conv2d(i, o, kernel_size=3),
                        torch.nn.ReLU(),
                        torch.nn.MaxPool2d(kernel_size=2) ]

        self.features = torch.nn.Sequential(*layers)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(channels[-1]*6*6, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
