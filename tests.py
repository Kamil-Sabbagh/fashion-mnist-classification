import torch
import pytest
from fashion_mnist_classification_nn_pytorch import (
    FashionMNISTDataModule,
    FashionMNISTClassifier,
    FashionMNISTClassifier2,
)
from train import train
from omegaconf import OmegaConf

# Define a string with the test configuration settings
test_config = """
batch_size: 16
epochs: 1
classifier: weak
optimizer:
  name: adam
  lr: 0.003
example: false
"""

@pytest.fixture
def config():
    return OmegaConf.create(test_config)

def test_data_module():
    data_module = FashionMNISTDataModule(batch_size=16)
    data_module.prepare_data()
    data_module.setup()

    train_dl = data_module.train_dataloader()
    batch = next(iter(train_dl))
    assert len(batch) == 2
    assert batch[0].shape == torch.Size([16, 1, 28, 28])
    assert batch[1].shape == torch.Size([16])

def test_classifier1():
    model = FashionMNISTClassifier()
    x = torch.randn(16, 1, 28, 28)
    x = x.view(x.shape[0], -1)
    y = model(x)
    assert y.shape == torch.Size([16, 10])

def test_classifier2():
    model = FashionMNISTClassifier2()
    x = torch.randn(16, 1, 28, 28)
    x = x.view(x.shape[0], -1)
    y = model(x)
    assert y.shape == torch.Size([16, 10])

def test_training_weak_classifier(config):
    config.classifier = "weak"
    train(config)

def test_training_strong_classifier(config):
    config.classifier = "strong"
    train(config)

