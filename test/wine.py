import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
 
from src.neural_net.nn import NeuralNetwork, TrainArg
from sklearn.datasets import load_wine
import numpy as np
 
wine = load_wine()
X = wine.data
y_raw = wine.target
 
indices = np.random.permutation(len(X))
X = X[indices]
y_raw = y_raw[indices]
 
std = X.std(axis=0)
std[std == 0] = 1
X = (X - X.mean(axis=0)) / std
 
split = int(len(X) * 0.8)
X_train, X_test = X[:split].tolist(), X[split:].tolist()
y_train = [[1 if i == label else 0 for i in range(3)] for label in y_raw[:split]]
y_test  = [[1 if i == label else 0 for i in range(3)] for label in y_raw[split:]]
 
nn = NeuralNetwork(13, [(32, "LeakyReLU"), (16, "LeakyReLU"), (3, "Softmax")])
 
training_args = TrainArg(
    batch_size=16,
    learning_rate=0.001,
    optimizer="adams_loss",
    loss="MSE",
    epochs=500,
    logging_steps=100,
    use_decay=True
)
 
nn.train(X_train, y_train, training_args=training_args, eval_dataset=[X_test, y_test], check_loss=True)
 
correct = 0
outputs = nn.predict(X_test)
for out, label in zip(outputs, y_raw[split:]):
    if np.argmax(out) == label:
        correct += 1
 
print(f"Accuracy: {correct}/{len(X_test)} = {round(correct/len(X_test)*100, 2)}%")