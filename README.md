# Neonet
Neonet is a lightweight deep learning library for building 
small to medium scale neural networks, with small to medium sized datasets
.it supports multiple training configurations, mutiple optimization methods, regularization techniques,
and different acttivation functions

## Installation
```python
pip install neonet
```

## Getting started
```python
from neonet.nn import NeuralNetwork, TrainArg
```

## Creating a neural network

```python
nn = NeuralNetwork(4, [(16, "LeakyReLU"), (8, "LeakyReLU"), (3, "Softmax")])
```

## Training arguments
- batch_size
- learning_rate
- optimizer
- regularizer
- alpha
- lasso
- b1 coefficient
- b2 coefficient
- epochs
- use_decay


```python
training_args = TrainArg( 
batch_size=16, 
learning_rate=0.001, 
optimizer="adams_loss", 
loss="MSE", epochs=500, 
logging_steps=100, 
use_decay=True )
```

to train:
```python
nn.train(X_train, y_train, training_args=training_args, eval_dataset=[X_test, y_test], check_loss=True)
```
## Features
	-	Dense neural network architecture
	-	Multiple activation functions
	-	Mini-batch training
	-	Learning rate decay
	-	Model evaluation during training
	-	Configurable optimizers and regularizers

