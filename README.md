# Fashion MNIST Classification

This repository contains code for training a neural network to classify images from the Fashion MNIST dataset.

## Requirements

all the needed requirements can be downloaded using this line:
```
pip install -r requirements.txt
```

## Files

- `train.py`: Main training script
- `fashion_mnist_classification_nn_pytorch.py`: Contains the FashionMNISTDataModule, FashionMNISTClassifier, and FashionMNISTClassifier2 classes
- `config.yaml`: Configuration file

## Models
There are two different models available:

* `Weak Model (FashionMNISTClassifier)`: A simple feedforward neural network with two hidden layers (128 and 64 units) and dropout.
* `Powerful Model (FashionMNISTClassifier2)`: A deeper feedforward neural network with four hidden layers (392, 196, 98, and 49 units) and dropout.

## Usage

To train the neural network, run the following command:\

```
python3 train.py optimizer.name=<optimizer_name> optimizer.lr=<learning_rate> batch_size=<batch_size> epochs=<epochs> classifier=<model_type>, example=<showing_example_or_not>
```
- `optimizer.name`: Can be "adam" or "sgd".
- `optimizer.lr`: Learning rate for the optimizer, can be any positive float
- `batch_size`: Batch size for training and validation, can be any positive integer
- `epochs`: Number of epochs to train for, can be any positive integer
- `classifier`: Model type to use for training, can be "weak" or "powerful"
- `example` : the trained model will predict a single random image from the dataset.

### Example

To train the model using the Adam optimizer with a learning rate of 0.001, batch size of 64, for 20 epochs and the first classifier type, run the following command:

```
python3 train.py optimizer.name=adam optimizer.lr=0.001 batch_size=64 epochs=20 classifier=powerful example=True
```
 
### Results

The training and validation losses and accuracies will be logged after each epoch. After training, the model will be saved in the trained_models directory with the current timestamp and configuration details.

For any issues or questions, feel free to create an issue on the repository.