import types
import torch
import pytorch_lightning


act_fn_by_name = {"tanh": torch.nn.Tanh,
                  "relu": torch.nn.ReLU,
                  "leakyrelu": torch.nn.LeakyReLU,
                  "gelu": torch.nn.GELU}

################################################################################
#                        GOOGLENET                                             #
################################################################################

class InceptionBlock(torch.nn.Module):
    def __init__(self, c_in, c_red: dict, c_out: dict, act_fn):
        super().__init__()

        self.conv_1x1 = torch.nn.Sequential(
            torch.nn.Conv2d(c_in, c_out["1x1"], kernel_size=1),
            torch.nn.BatchNorm2d(c_out["1x1"]),
            act_fn()
        )

        self.conv_3x3 = torch.nn.Sequential(
            torch.nn.Conv2d(c_in, c_red["3x3"], kernel_size=1),
            torch.nn.BatchNorm2d(c_red["3x3"]),
            act_fn(),
            torch.nn.Conv2d(c_red["3x3"], c_out["3x3"], kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(c_out["3x3"]),
            act_fn(),
        )

        self.conv_5x5 = torch.nn.Sequential(
            torch.nn.Conv2d(c_in, c_red["5x5"], kernel_size=1),
            torch.nn.BatchNorm2d(c_red["5x5"]),
            act_fn(),
            torch.nn.Conv2d(c_red["5x5"], c_out["5x5"], kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(c_out["5x5"]),
            act_fn(),
        )

        self.max_pool = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            torch.nn.Conv2d(c_in, c_out["max"], kernel_size=1),
            torch.nn.BatchNorm2d(c_out["max"]),
            act_fn(),
        )

    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        x_max = self.max_pool(x)
        x_out = torch.cat([x_1x1, x_3x3, x_5x5, x_max], dim=1)

        return x_out


class GoogleNet(torch.nn.Module):
    def __init__(self, num_classes=10, act_fn_name="relu", **kwargs):
        super().__init__()

        self.hparams = types.SimpleNamespace(
            num_classes=num_classes, act_fn_name=act_fn_name, act_fn=act_fn_by_name[act_fn_name]
        )
        self._create_network()
        self._init_params()

    def _create_network(self):
        self.input_net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            self.hparams.act_fn(),
        )

        self.inception_blocks = torch.nn.Sequential(
            InceptionBlock(
                64,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 32, "5x5": 8, "max": 8},
                act_fn=self.hparams.act_fn,
            ),
            InceptionBlock(
                64,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12},
                act_fn=self.hparams.act_fn,
            ),
            torch.nn.MaxPool2d(3, stride=2, padding=1),
            InceptionBlock(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12},
                act_fn=self.hparams.act_fn,
            ),
            InceptionBlock(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16},
                act_fn=self.hparams.act_fn,
            ),
            InceptionBlock(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16},
                act_fn=self.hparams.act_fn,
            ),
            InceptionBlock(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 32, "3x3": 48, "5x5": 24, "max": 24},
                act_fn=self.hparams.act_fn,
            ),
            torch.nn.MaxPool2d(3, stride=2, padding=1),
            InceptionBlock(
                128,
                c_red={"3x3": 48, "5x5": 16},
                c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16},
                act_fn=self.hparams.act_fn,
            ),
            InceptionBlock(
                128,
                c_red={"3x3": 48, "5x5": 16},
                c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16},
                act_fn=self.hparams.act_fn,
            ),
        )

        self.output_net = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(128, self.hparams.num_classes)
        )

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight,
                                        nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.inception_blocks(x)
        x = self.output_net(x)

        return x


################################################################################
#                        RESNET                                                #
################################################################################

class ResNetBlock(torch.nn.Module):
    def __init__(self, c_in, act_fn, subsample=False, c_out=-1):
        super().__init__()

        if not subsample:
            c_out = c_in

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(
                c_in, c_out, kernel_size=3, padding=1,
                stride=1 if not subsample else 2,
                bias=False
            ),
            torch.nn.BatchNorm2d(c_out),
            act_fn(),
            torch.nn.Conv2d(
                c_out, c_out, kernel_size=3, padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(c_out),
        )

        self.downsample = torch.nn.Conv2d(
            c_in, c_out, kernel_size=1,
            stride=2) if subsample else None
        self.act_fn = act_fn()

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)

        out = z + x
        out = self.act_fn(out)

        return out

class PreActResNetBlock(torch.nn.Module):
    def __init__(self, c_in, act_fn, subsample=False, c_out=-1):
        super().__init__()

        if not subsample:
            c_out = c_in

        self.net = torch.nn.Sequential(
            torch.nn.BatchNorm2d(c_in),
            act_fn(),
            torch.nn.Conv2d(c_in, c_out, kernel_size=3, padding=1,
                      stride=1 if not subsample else 2,
                      bias=False),
            torch.nn.BatchNorm2d(c_out),
            act_fn(),
            torch.nn.Conv2d(
                c_out, c_out, kernel_size=3, padding=1,
                bias=False
            ),
        )

        self.downsample = (
            torch.nn.Sequential(
                torch.nn.BatchNorm2d(c_in),
                act_fn(),
                torch.nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, bias=False)
            ) if subsample
            else None
        )

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        return out


resnet_blocks_by_name = {
    "ResNetBlock": ResNetBlock,
    "PreActResNetBlock": PreActResNetBlock,
}


class ResNet(torch.nn.Module):
    def __init__(
            self,
            num_classes=10,
            num_blocks=[3, 3, 3],
            c_hidden=[16, 32, 64],
            act_fn_name="relu",
            block_name="ResNetBlock",
            **kwargs,
    ):
        super().__init__()

        assert block_name in resnet_blocks_by_name

        self.hparams = types.SimpleNamespace(
            num_classes=num_classes,
            c_hidden=c_hidden,
            num_blocks=num_blocks,
            act_fn_name=act_fn_name,
            act_fn=act_fn_by_name[act_fn_name],
            block_class=resnet_blocks_by_name[block_name],
        )
        self._create_network()
        self._init_params()

    def _create_network(self):
        c_hidden = self.hparams.c_hidden

        if self.hparams.block_class == PreActResNetBlock:
            self.input_net = torch.nn.Sequential(
                torch.nn.Conv2d(
                    3, c_hidden[0], kernel_size=3, padding=1, bias=False
                ))
        else:
            self.input_net = torch.nn.Sequential(
                torch.nn.Conv2d(
                    3, c_hidden[0], kernel_size=3, padding=1, bias=False
                ),
                torch.nn.BatchNorm2d(c_hidden[0]),
                self.hparams.act_fn(),
            )

        blocks = []
        for block_idx, block_count in enumerate(self.hparams.num_blocks):
            for bc in range(block_count):
                subsample = bc == 0 and block_idx > 0
                blocks.append(
                    self.hparams.block_class(
                        c_in=c_hidden[block_idx if not subsample
                                      else (block_idx - 1)],
                        act_fn=self.hparams.act_fn,
                        subsample=subsample,
                        c_out=c_hidden[block_idx],
                    )
                )
        self.blocks = torch.nn.Sequential(*blocks)

        self.output_net = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(c_hidden[-1], self.hparams.num_classes)
        )

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.blocks(x)
        x = self.output_net(x)

        return x


################################################################################
#                        DENSENET                                              #
################################################################################

class DenseLayer(torch.nn.Module):
    def __init__(self, c_in, bn_size, growth_rate, act_fn):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.BatchNorm2d(c_in),
            act_fn(),
            torch.nn.Conv2d(c_in, bn_size * growth_rate, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(bn_size * growth_rate),
            act_fn(),
            torch.nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        out = self.net(x)
        out = torch.cat([out, x], dim=1)

        return out


class DenseBlock(torch.nn.Module):
    def __init__(self, c_in, num_layers, bn_size, growth_rate, act_fn):
        super().__init__()

        layers = []
        for layer_idx in range(num_layers):
            layer_c_in = c_in + layer_idx * growth_rate
            layers.append(DenseLayer(c_in=layer_c_in, bn_size=bn_size,
                                     growth_rate=growth_rate, act_fn=act_fn))

        self.block = torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return out


class TransitionLayer(torch.nn.Module):
    def __init__(self, c_in, c_out, act_fn):
        super().__init__()
        self.transition = torch.nn.Sequential(
            torch.nn.BatchNorm2d(c_in),
            act_fn(),
            torch.nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.transition(x)


class DenseNet(torch.nn.Module):
    def __init__(
            self, num_classes=10, num_layers=[6, 6, 6, 6], bn_size=2, growth_rate=16, act_fn_name="relu",
            **kwargs
    ):
        super().__init__()

        self.hparams = types.SimpleNamespace(
            num_classes=num_classes,
            num_layers=num_layers,
            bn_size=bn_size,
            growth_rate=growth_rate,
            act_fn_name=act_fn_name,
            act_fn=act_fn_by_name[act_fn_name],
        )
        self._create_network()
        self._init_params()

    def _create_network(self):
        c_hidden = self.hparams.growth_rate * self.hparams.bn_size

        self.input_net = torch.nn.Sequential(
            torch.nn.Conv2d(3, c_hidden, kernel_size=3, padding=1)
        )

        blocks = []
        for block_idx, num_layers in enumerate(self.hparams.num_layers):
            blocks.append(
                DenseBlock(
                    c_in=c_hidden,
                    num_layers=num_layers,
                    bn_size=self.hparams.bn_size,
                    growth_rate=self.hparams.growth_rate,
                    act_fn=self.hparams.act_fn,
                )
            )
            c_hidden = c_hidden + num_layers * self.hparams.growth_rate
            if block_idx < len(self.hparams.num_layers) - 1:
                blocks.append(TransitionLayer(c_in=c_hidden, c_out=c_hidden // 2, act_fn=self.hparams.act_fn))
                c_hidden = c_hidden // 2

        self.blocks = torch.nn.Sequential(*blocks)

        self.output_net = torch.nn.Sequential(
            torch.nn.BatchNorm2d(c_hidden),
            self.hparams.act_fn(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(c_hidden, self.hparams.num_classes),
        )

    def _init_params(self):

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.blocks(x)
        x = self.output_net(x)

        return x


################################################################################

model_dict = {
    'googlenet': GoogleNet,
    'resnet'   : ResNet,
    'densenet' : DenseNet
}


def create_model(model_name, model_hparams):
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)
    else:
        assert False, f'Unknown model name "{model_name}"'

class CIFARModule(pytorch_lightning.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model(model_name, model_hparams)
        self.loss_module = torch.nn.CrossEntropyLoss()

    def forward(self, imgs):
        return self.model(imgs)

    def configure_optimizers(self):
        if self.hparams.optimizer_name == 'Adam':
            optimizer = torch.optim.AdamW(self.parameters(),
                                          **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(),
                                        **self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[100, 150],
                                                         gamma=0.1)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()

        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()

        self.log('test_acc', acc)
