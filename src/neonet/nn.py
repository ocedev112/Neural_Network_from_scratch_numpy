import joblib
import numpy as np
from typing import Optional
from pydantic import BaseModel, model_validator
from enum import Enum


class Optimizer(str, Enum):
     SGD = "SGD"
     GRADIENT_DESCENT = "GD"
     ADAMS_LOSS = "adams_loss"


class Regularization(str, Enum):
     L1 = "L1"
     L2 = "L2"
class Loss(str, Enum):
     MSE = "MSE"
     MAE = "MAE"
     BCE = "BCE"
     HUBER = "Huber"
     
class TrainArg(BaseModel):
     batch_size: Optional[int] = None
     learning_rate: float
     optimizer: Optional[Optimizer] = None
     regularization: Optional[Regularization] = None
     alpha: Optional[float] = None
     lasso: Optional[float] = None
     b1_coefficient: float = 0.99
     b2_coefficient: float = 0.999
     epsilon: float = 1e-8
     loss: Optional[Loss] = None
     epochs: Optional[int]  = None
     use_decay: Optional[bool]  = False
     logging_steps: Optional[int] = 1

     @model_validator(mode="after")
     def validate_dependency(self):
          if self.batch_size is not None and self.batch_size < 1:
               raise ValueError("Batch size must be equal to ot greater than 1 when set")
          if self.regularization is not None and (self.alpha is None and self.lasso is None):
               raise ValueError("You need to set alpha or lasso coefficient when using regularization")
          if self.regularization == "L1" and self.lasso == None:
               raise ValueError("You need to use lasso coefficient with L1, alpha is for L2")
          if self.regularization == "L2" and self.alpha == None:
               raise ValueError("You need to use alpha coefficient with L2, lasso is for L1")
          if self.optimizer == "adams_loss" and (self.epsilon <= 0):
               raise ValueError("You need to use epsilon value for adams loss above 0")
          if self.epochs is not None and (self.epochs < 1 or self.logging_steps < 1):
              raise ValueError("Epochs and logging steps must be greater than 1")
          if self.epochs is not None and self.logging_steps > self.epochs:
              raise ValueError("Logging steps must be smaller than or equal to the epochs value")
          return self



class NeuralNetwork():
      
      def  optimize_weights_activation(self, activation_sign, input_size, output_size):
           He = ["ReLU", "LeakyReLU", "ELU", "Swish"]
           Xavier = ["Tanh", "Sigmoid", "Softmax"]
           if activation_sign in He:
                return np.sqrt(2/(input_size))
           if activation_sign in Xavier:
                return np.sqrt(2/(input_size + output_size))
           if activation_sign == "None":
                return  1
           
      def __init__(self, input_size,layers, optimize_weights_activation=True):
          self.layers = layers
          self.weight_matrix = []
          self.bias_matrix = []
          for idx,layer in enumerate(layers):
               if idx==0:
                    weights_balancer = self.optimize_weights_activation(layer[1], input_size, layer[0]) if optimize_weights_activation else 1
                    self.weight_matrix.append(np.random.randn(input_size, layer[0]) * weights_balancer)
                    self.bias_matrix.append(np.zeros(layer[0]))
               else:
                weights_balancer = self.optimize_weights_activation(layer[1], layers[idx-1][0], layer[0])  if optimize_weights_activation else 1
                self.weight_matrix.append(np.random.randn(layers[idx-1][0] ,layer[0])  * weights_balancer)  
                self.bias_matrix.append(np.zeros(layer[0]))

      def activation(self, sign, z):
        if sign == "ReLU":
            return np.atleast_1d(np.clip(z, a_min=0, a_max=None))
        if sign == "LeakyReLU":
            return np.atleast_1d(np.where(z > 0, z, 0.01 * z))
        if sign == "ELU":
            z = np.clip(z, -500, 500)  
            return np.atleast_1d(np.where(z > 0, z, 0.01 * (np.exp(z) - 1)))
        if sign == "Sigmoid":
             z = np.clip(z, -500, 500)
             return np.atleast_1d(1 / (1 + np.exp(-z)))
        if sign == "Softmax":
            z = np.atleast_2d(z)
            e = np.exp(z - np.max(z, axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        if sign == "Tanh":
             return np.atleast_1d(np.tanh(z)) 
        if sign == "Swish":
            return np.atleast_1d(z * (1 / (1 + np.exp(-z))))
        if sign == "None":
             return np.atleast_1d(z)
           
      def __forward(self,inputs, weight_matrix, bias_matrix,idx,isTraining, training_layers):
          layer_dict = {}
          weight = weight_matrix[idx-1]
          bias = bias_matrix[idx-1]
          layer = np.matmul(inputs, weight)
          layer = (layer + bias)
          activation_sign = self.layers[idx-1][1]
          activated_layer = self.activation(activation_sign, layer)
          if isTraining:
               layer_dict["input"] = inputs
               layer_dict["weight"] = weight
               layer_dict["bias"] = bias
               layer_dict["z_layer"] = layer
               layer_dict["activated_layer"] = activated_layer
               layer_dict["activation_sign"] = activation_sign
               layer_dict["v_layer"] = []
               training_layers.append(layer_dict)         
               
          if idx == len(weight_matrix):
               if isTraining:
                    return activated_layer, training_layers
               return activated_layer
          idx+=1
          return self.__forward(activated_layer, self.weight_matrix, self.bias_matrix, idx, isTraining, training_layers)
           
      def __forward_pass(self,inputs, train_mode=False):
           training_layers = []
           outputs = self.__forward(np.array(inputs), self.weight_matrix, self.bias_matrix, 1, train_mode, training_layers)
           if train_mode:
                 return outputs[0], outputs[1] 
           return outputs

      def derivate_activation(self,sign, a):
           if sign == "ReLU":
               return np.where(a > 0, 1, 0)
           if sign == "LeakyReLU": 
               return np.where(a> 0, 1, 0.01)
           if sign == "ELU":       
              return np.where(a > 0, 1.0, a + 0.01) 
           if sign == "Sigmoid":
               return a*(1-a)

           if sign == "Tanh":     
               return 1 - a ** 2
           if sign == "Swish":     
               s = 1 / (1 + np.exp(-a))
               return s + a * s * (1 - s)
           if sign == "None":
               return 1
      def loss_derivative(self, predicted_output, real_output, loss):
        if loss == "MSE":
               return 2 * (predicted_output - real_output)
        if loss == "MAE":
               return np.sign(predicted_output - real_output)
        if loss == "BCE":
            predicted_output = np.clip(predicted_output, 1e-7, 1 - 1e-7)
            real_output = np.array(real_output)
            diff = -(real_output / predicted_output) + (1 - real_output) / (1 - predicted_output)
            return np.atleast_1d(np.clip(diff, -10, 10))
        if loss == "Huber":
               delta = 1.0
               diff = predicted_output - real_output
               return np.where(np.abs(diff) < delta, diff, delta * np.sign(diff))
        
      def __backprop(self, layer_dict, real_output, loss):
            lay_idx = len(layer_dict) - 1
            cost_weight_matrix = []
            cost_bias_matrix = []
            for lay_idx,( matrix, layer) in enumerate(zip(self.weight_matrix[::-1], layer_dict[::-1])):
                 real_lay_idx = len(layer_dict) - 1 - lay_idx
                 if lay_idx == 0:
                     if layer["activation_sign"] == "Softmax":
                         v_matrix = layer["activated_layer"] - np.array(real_output)
                     else:
                         in_v = self.loss_derivative(layer["activated_layer"], real_output, loss)
                         dadz = self.derivate_activation(layer["activation_sign"], layer["activated_layer"])
                         v_matrix = dadz * in_v
                 else:
                     prev_weights = layer_dict[real_lay_idx+1]["weight"]
                     v_layer = layer_dict[real_lay_idx+1]["v_layer"]
                     in_v = np.dot(v_layer, prev_weights.T)                         
                     dadz = self.derivate_activation(layer["activation_sign"], layer["activated_layer"])
                     v_matrix = dadz * in_v
                 cost_weight = np.dot(layer["input"].T, v_matrix)  / len(v_matrix)
                 cost_bias = v_matrix.mean(axis=0)

                 layer["v_layer"] = v_matrix
                 cost_weight_matrix.append(cost_weight)
                 cost_bias_matrix.append(cost_bias)

            return cost_weight_matrix, cost_bias_matrix
      

      def regularization(self,regularizer, matrix, coef):
           if regularizer == None:
                return 0
           if regularizer == "L1":
               matrix = np.sign(matrix)
               return matrix * coef
           elif regularizer == "L2":
               return matrix * coef

      def compute_loss(self, cost_weight, cost_bias, learning_rate, regularizer, coef):
       for i, (cost_w, cost_b) in enumerate(zip(cost_weight, cost_bias)):
          reg_w = self.regularization(regularizer, self.weight_matrix[-(i+1)], coef)
          reg_b = self.regularization(regularizer, self.bias_matrix[-(i+1)], coef)
          self.weight_matrix[-(i+1)] -= (cost_w + reg_w) * learning_rate

          self.bias_matrix[-(i+1)] -= (cost_b + reg_b) * learning_rate

      def compute_loss_adams(self, momentum_w, energy_w, momentum_b, energy_b, learning_rate, regularizer, coef, ep, t, b1,b2):
       for i, (m_w,v_w, m_b, v_b ) in enumerate(zip(momentum_w, energy_w, momentum_b, energy_b)):
          reg_w = self.regularization(regularizer, self.weight_matrix[-(i+1)], coef)
          reg_b = self.regularization(regularizer, self.bias_matrix[-(i+1)], coef)
          m_w_hat = m_w / (1 - b1 ** t)             
          v_w_hat = v_w / (1 - b2 ** t)
          m_b_hat = m_b / (1 - b1 ** t)
          v_b_hat = v_b / (1 - b2 ** t)

          
          self.weight_matrix[-(i+1)] -= (m_w_hat / (np.sqrt(v_w_hat) + ep) + reg_w) * learning_rate
          self.bias_matrix[-(i+1)]   -= (m_b_hat / (np.sqrt(v_b_hat) + ep) + reg_b) * learning_rate

      def batch_data(self, inputs, outputs, batch_size):
          if batch_size is None:
              batch_size = len(outputs)
          for i in range(0, len(outputs), batch_size):
             yield np.array(inputs[i:i + batch_size]), np.array(outputs[i:i + batch_size])

      def loss_function(self,pred, real, loss):
         if loss == "MSE":
              return ((real - pred)**2).mean(axis=1).sum(axis=0)
         if loss == "MAE":
              return np.abs((real - pred)).mean(axis=1).sum(axis=0)
         if loss == "BCE":
              ep = 1e-15
              pred = np.clip(pred, ep, 1 - ep)
              return -(real * np.log(pred) + (1 - real) * np.log(1 - pred)).mean(axis=1).sum(axis=0)
         if loss == "Huber":
             delta = 1.0
             diff = np.abs(real - pred)
             return np.where(diff < delta, 0.5 * diff**2, delta * diff - 0.5 * delta**2)
                  
      def train(self, inputs, real_output, training_args=None, eval_dataset=None, check_loss=False):
             if  len(inputs) != len(real_output):
                 raise ValueError("input and output dataset must match")
             regularizer = None
             coef = 0.0
             batch_size = None
             loss = "MSE"
             epochs = 1
             if training_args == None:
                  learning_rate = 0.04
             else:
                  learning_rate = training_args.learning_rate
                  regularizer = training_args.regularization
                  if regularizer is not None and regularizer == "L1":
                       coef = training_args.lasso 
                  else:
                       coef = training_args.alpha 
                  if training_args.batch_size is not None:
                       batch_size = training_args.batch_size 
                  if training_args.loss is not None:
                       loss = training_args.loss
                  if training_args.epochs is not None:
                      epochs = training_args.epochs


             

             
           
             m_w,v_w = None,None
             m_b, v_b = None, None
             g_m_w, g_v_w = None, None
             g_m_b, g_v_b = None, None
             t = 0
             base_lr = learning_rate
             for epoch in range(epochs):
               cost_weight_list = []
               cost_bias_list = []
               if training_args is not None and training_args.use_decay:
                    decay = 1 / (1 + 0.001 * epoch)
                    learning_rate = base_lr  * decay 
               global_idx = 0
               for X_batch, y_batch in self.batch_data(inputs, real_output, batch_size):
                  output, layer_dict = self.__forward_pass(X_batch, train_mode=True)
                  cost_weight_matrix, cost_bias_matrix = self.__backprop(layer_dict, y_batch, loss)
                  if training_args == None or  training_args.optimizer == "GD":
                    cost_weight_list.append(cost_weight_matrix)
                    cost_bias_list.append(cost_bias_matrix)
                  if training_args is not None and training_args.optimizer == "SGD":
                     self.compute_loss(cost_weight_matrix, cost_bias_matrix, learning_rate, regularizer, coef)

                  if training_args is not None and training_args.optimizer == "adams_loss":
                       b1 = training_args.b1_coefficient
                       b2 = training_args.b2_coefficient
                       ep = training_args.epsilon
                       g_m_w = cost_weight_matrix
                       g_v_w = [cw**2 for cw in cost_weight_matrix]
                       g_m_b = cost_bias_matrix
                       g_v_b = [cb**2 for cb in cost_bias_matrix]

                       if m_w is None:
                           m_w = [np.zeros_like(w) for w in cost_weight_matrix]
                           v_w = [np.zeros_like(w) for w in cost_weight_matrix]
                           m_b = [np.zeros_like(b) for b in cost_bias_matrix]
                           v_b = [np.zeros_like(b) for b in cost_bias_matrix]
                            
                       mv_w = [(b1*mw + (1-b1)*gm, b2*vw + (1-b2)*gv) for mw, gm, vw, gv in zip(m_w, g_m_w, v_w, g_v_w)]
                       mv_b = [(b1*mb + (1-b1)*gm, b2*vb + (1-b2)*gv) for mb, gm, vb, gv in zip(m_b, g_m_b, v_b, g_v_b)]

                       m_w, v_w = zip(*mv_w)
                       m_b, v_b = zip(*mv_b)
                       t+=1
                       self.compute_loss_adams(m_w,v_w,m_b,v_b, learning_rate, regularizer, coef,ep,t, b1, b2)

               if training_args == None or  training_args.optimizer == "GD":
                 cost_weight_mean = [np.stack(arrays).mean(axis=0) for arrays in zip(*cost_weight_list)]
                 cost_bias_mean = [np.stack(arrays).mean(axis=0) for arrays in zip(*cost_bias_list)]
                 self.compute_loss(cost_weight_mean, cost_bias_mean, learning_rate, regularizer, coef)
               
               if check_loss:
                 if training_args is not None and training_args.logging_steps is not None:
                   bold = "\033[1m"
                   bold_end = "\033[0m"
                   if epoch % training_args.logging_steps == 0:      
                    train_sample = len(inputs) // 4
                    train_ouputs = self.predict(inputs[:train_sample])
                    real_array = np.array(train_ouputs)
                    pred_array = np.array(real_output[:train_sample])
                    train_loss = self.loss_function(pred_array, real_array, loss)
                    print(f"{bold}Epoch {epoch}/{epochs}{bold_end} ---- Loss: {round(train_loss, 4)}")
                    if eval_dataset is not None:
                      test_sample = len(inputs) // 4
                      test_outputs  = self.predict(eval_dataset[0][:test_sample])
                      pred_array_test = np.array(test_outputs)
                      real_array_test = np.array(eval_dataset[1][:test_sample])
                      test_loss = self.loss_function(pred_array_test, real_array_test, loss)
                      bold = "\033[1m"
                      bold_end = "\033[0m"
                      print(f"{bold}Epoch {epoch}/{epochs}{bold_end} ---- Val Loss: {round(test_loss, 4)}")

             
      def predict(self, inputs):
              outputs = self.__forward_pass(np.array(inputs), False)
              return  outputs
      
      def save(self, path):
          joblib.dump({
              "weight_matrix" : self.weight_matrix, 
              "bias_matrix" : self.bias_matrix, 
              "layers" :self.layers
              },path)
      

      @classmethod
      def load(cls, path):
          obj = cls.__new__(cls)
          loaded = joblib.load(path)
          obj.weight_matrix = loaded["weight_matrix"]
          obj.bias_matrix = loaded["bias_matrix"]
          obj.layers = loaded["layers"]
          return obj
      


