import pytorch_lightning
import torch

class LitClassifier(pytorch_lightning.LightningModule):
    def __init__(self, hidden_dim: int = 128, learning_rate: float = 0.0001):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = torch.nn.Linear(28 * 28, self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10)
        
        self.l1 = self.l1.cuda()
        self.l2 = self.l2.cuda()

        self.X = True

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
