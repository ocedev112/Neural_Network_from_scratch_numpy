import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
 
from src.neural_net.nn import NeuralNetwork, TrainArg
from sklearn.datasets import load_breast_cancer
import numpy as np
 
cancer = load_breast_cancer()
X = cancer.data
y_raw = cancer.target
 
indices = np.random.permutation(len(X))
X = X[indices]
y_raw = y_raw[indices]
 
std = X.std(axis=0)
std[std == 0] = 1
X = (X - X.mean(axis=0)) / std
 
split = int(len(X) * 0.8)
X_train, X_test = X[:split].tolist(), X[split:].tolist()
y_train = [[label] for label in y_raw[:split].tolist()]
y_test  = [[label] for label in y_raw[split:].tolist()]
 
nn = NeuralNetwork(30, [(64, "LeakyReLU"), (32, "LeakyReLU"), (1, "Sigmoid")])
 
training_args = TrainArg(
    batch_size=32,
    learning_rate=0.001,
    optimizer="adams_loss",
    loss="BCE",
    epochs=500,
    logging_steps=100,
    use_decay=True
)
 
nn.train(X_train, y_train, training_args=training_args, eval_dataset=[X_test, y_test], check_loss=True)
 
correct = 0
outputs = nn.predict(X_test)
for out, label in zip(outputs, y_raw[split:]):
    predicted = 1 if out[0] >= 0.5 else 0
    if predicted == label:
        correct += 1
 
print(f"Accuracy: {correct}/{len(X_test)} = {round(correct/len(X_test)*100, 2)}%")