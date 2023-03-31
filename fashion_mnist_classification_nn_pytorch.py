import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from collections import OrderedDict
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt


class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

    def prepare_data(self):
        datasets.FashionMNIST(
            "F_MNIST_data", download=True, train=True, transform=self.transform
        )
        datasets.FashionMNIST(
            "F_MNIST_data", download=True, train=False, transform=self.transform
        )

    def setup(self, stage=None):
        train_ds = datasets.FashionMNIST(
            "F_MNIST_data", train=True, transform=self.transform
        )
        test_ds = datasets.FashionMNIST(
            "F_MNIST_data", train=False, transform=self.transform
        )

        train_num = len(train_ds)
        indices = list(range(train_num))
        np.random.shuffle(indices)
        split = int(np.floor(0.2 * train_num))
        val_idx, train_idx = indices[:split], indices[split:]

        self.train_ds = torch.utils.data.Subset(train_ds, train_idx)
        self.val_ds = torch.utils.data.Subset(train_ds, val_idx)
        self.test_ds = test_ds

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)


class FashionMNISTClassifier(pl.LightningModule):
    def __init__(self, optimizer_name="adam", learning_rate=0.003):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.model = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(784, 128)),
                    ("relu1", nn.ReLU()),
                    ("drop1", nn.Dropout(0.25)),
                    ("fc2", nn.Linear(128, 64)),
                    ("relu2", nn.ReLU()),
                    ("drop2", nn.Dropout(0.25)),
                    ("output", nn.Linear(64, 10)),
                    ("logsoftmax", nn.LogSoftmax(dim=1)),
                ]
            )
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.shape[0], -1)
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.shape[0], -1)
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss


class FashionMNISTClassifier2(pl.LightningModule):
    def __init__(self, optimizer_name="adam", learning_rate=0.003):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.model = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(784, 392)),
                    ("relu1", nn.ReLU()),
                    ("drop1", nn.Dropout(0.25)),
                    ("fc12", nn.Linear(392, 196)),
                    ("relu2", nn.ReLU()),
                    ("drop2", nn.Dropout(0.25)),
                    ("fc3", nn.Linear(196, 98)),
                    ("relu3", nn.ReLU()),
                    ("drop3", nn.Dropout(0.25)),
                    ("fc4", nn.Linear(98, 49)),
                    ("relu4", nn.ReLU()),
                    ("output", nn.Linear(49, 10)),
                    ("logsoftmax", nn.LogSoftmax(dim=1))
                ]
            )
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.shape[0], -1)
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.shape[0], -1)
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss


def predict_an_image(data_module, model):
    def visualize_prediction(image, label, proba, class_names):
        fig, (ax1, ax2) = plt.subplots(figsize=(13, 6), nrows=1, ncols=2)
        ax1.axis("off")
        ax1.imshow(image.cpu().numpy().squeeze())
        ax1.set_title(class_names[label.item()])
        ax2.bar(range(10), proba.detach().cpu().numpy().squeeze())
        ax2.set_xticks(range(10))
        ax2.set_xticklabels(class_names, size="small")
        ax2.set_title("Predicted Probabilities")
        plt.tight_layout()
        plt.show()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_module.setup(stage="test")

    test_dl = data_module.test_dataloader()
    dataiter = iter(test_dl)
    images, labels = next(dataiter)

    index = 49
    img, label = images[index], labels[index]
    # Convert 2D image to 1D vector
    img = img.view(img.shape[0], -1)

    # Make a prediction with the best_model
    with torch.no_grad():
        model.eval()
        proba = torch.exp(model(img))

    # Class names
    desc = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle Boot",
    ]

    # Visualize the prediction
    visualize_prediction(images[index], label, proba, desc)
