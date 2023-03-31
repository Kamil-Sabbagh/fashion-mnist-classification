import hydra
import torch
from pytorch_lightning import Trainer
from omegaconf import DictConfig
from fashion_mnist_classification_nn_pytorch import FashionMNISTDataModule, FashionMNISTClassifier, FashionMNISTClassifier2

@hydra.main(config_path="config", config_name="config", version_base=None)
def train(cfg: DictConfig):
    data_module = FashionMNISTDataModule(batch_size=cfg.batch_size)
    

    model = FashionMNISTClassifier(optimizer_name=cfg.optimizer.name, learning_rate=cfg.optimizer.lr)
    model2 = FashionMNISTClassifier2(optimizer_name=cfg.optimizer.name, learning_rate=cfg.optimizer.lr)
    
    #model.configure_optimizers = hydra.utils.instantiate(cfg.optimizer)

    
    
    trainer = Trainer(
        max_epochs=cfg.epochs,
        devices=1,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
    )

    trainer2 = Trainer(
        max_epochs=cfg.epochs,
        devices=1,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

    print("should be training and testing the first model")

    trainer2.fit(model2, datamodule=data_module)
    trainer2.test(model2, datamodule=data_module)

    print("should be training and testing the second model")

if __name__ == "__main__":
    train()
