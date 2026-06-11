# Neonet (Neural Network with Numpy)
Neonet is a lightweight NumPy-based neural network library for building and experimenting with simple deep learning models. It is designed for small and medium scale projects, prototyping, and understanding core neural network workflows without relying on large frameworks like TensorFlow or PyTorch..

## Features

- Fully connected (dense) neural network architecture
- Built entirely with NumPy
- Multiple activation functions
- Multiple optimization algorithms
- Mini-batch gradient descent training
- Learning rate decay support
- Model evaluation during training
- Configurable regularization techniques
- Suitable for educational and research purposes
- Lightweight and easy to use

## Installation

Install Neonet from PyPI:

```bash
pip install neonet
```

## Quick start

Import the required classes:

```python
from neonet.nn import NeuralNetwork, TrainArg
```

## Create a Neural Network

Create a multi-layer neural network with customizable activation functions

```python
nn = NeuralNetwork(4, [(16, "LeakyReLU"), (8, "LeakyReLU"), (3, "Softmax")])
```

## Training Configurations

Neonet allows you to configure various training parameters, including: 

- Batch size
- Learning rate
- Optimizer
- Loss function
- Regularization method
- L1/L2 coefficients
- Beta coefficients for adaptive optimizers
- Number of epochs
- Learning rate decay


Example
```python
training_args = TrainArg( 
batch_size=16, 
learning_rate=0.001, 
optimizer="adams_loss", 
loss="MSE",
epochs=500, 
logging_steps=100, 
use_decay=True
)
```

to train:
```python
nn.train(X_train, y_train, training_args=training_args, eval_dataset=[X_test, y_test], check_loss=True)
```
# Supported Components

# Activation Functions
- ReLU
- LeakyReLU
- ELU
- Sigmoid
- Tanh
- Softmax

# Initialization
- Xavier
- He

# Optimization Methods
- SGD
- Adam loss

# Regularization Methods
- L1 Regularization(Lasso)
- L2 Regularization(Ridge)

# Use case

Neonet is suitable for:

- Prototyping neural network architectures quickly
- Small to medium-scale machine learning tasks
- Experimenting with activation functions, optimizers, and loss functions
- Research and algorithm testing
- Running lightweight models on CPU-only environments
- Exploring neural network behavior through a NumPy-based implementation

#Why Neonet?

Unlike large deep learning frameworks, Neonet focuses on simplicity, transparency, and educational value. The library makes it easier to understand the mechanics of forward propagation, backpropagation, optimization, and regularization while still providing practical training capabilities.

#Contributing

Contributions, bug reports, and feature requests are welcome. Feel free to open an issue or submit a pull request.



