#Neonet
Neonet is a new deep learning library that can be used for  for building 
small to medium scale neural networks, with small to medium sized datasts
.it supports multiple training arguments for training
and different acttivation functions

To get started
```python
pip install neonet

from neonet.nn import NeuralNetwork, TrainArg


```

creating a neural network

```python
nn = NeuralNetwork(4, [(16, "LeakyReLU"), (8, "LeakyReLU"), (3, "Softmax")])
```

Training arguments
-batch_size
-learning_rate
-optimizer
-regularizee
-alphq
-lasso
-b1 coefficient
-b2 coefficient
-epochs
-use_decay


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
