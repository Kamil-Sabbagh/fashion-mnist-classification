import hydra
import torch
from pytorch_lightning import Trainer
from omegaconf import DictConfig
from fashion_mnist_classification_nn_pytorch import (
    FashionMNISTDataModule,
    FashionMNISTClassifier,
    FashionMNISTClassifier2,
    predict_an_image,
)


@hydra.main(config_path="config", config_name="config", version_base=None)
def train(cfg: DictConfig):

    data_module = FashionMNISTDataModule(batch_size=cfg.batch_size)

    if cfg.classifier == "weak":

        model = FashionMNISTClassifier(
            optimizer_name=cfg.optimizer.name, learning_rate=cfg.optimizer.lr
        )

        trainer = Trainer(
            max_epochs=cfg.epochs,
            devices=1,
            accelerator="cuda" if torch.cuda.is_available() else "cpu",
        )

        trainer.fit(model, datamodule=data_module)
        trainer.test(model, datamodule=data_module)

        if cfg.example:
            predict_an_image(data_module, model)

    else:
        model2 = FashionMNISTClassifier2(
            optimizer_name=cfg.optimizer.name, learning_rate=cfg.optimizer.lr
        )

        trainer2 = Trainer(
            max_epochs=cfg.epochs,
            devices=1,
            accelerator="cuda" if torch.cuda.is_available() else "cpu",
        )

        trainer2.fit(model2, datamodule=data_module)
        trainer2.test(model2, datamodule=data_module)

        if cfg.example:
            predict_an_image(data_module, model)


if __name__ == "__main__":
    train()
