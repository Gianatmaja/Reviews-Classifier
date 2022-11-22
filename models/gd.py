import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics import accuracy_score

from utils import compute_average_accuracy

import random


class GD:
    def __init__(self, max_iter, num_features, eta, lam):
        super().__init__()
        self.max_iter = max_iter
        self.eta = eta
        self.lam = lam
        self.num_features = num_features
        self.weights = None
        self.bias = None

    def init_parameters(self):
        self.weights = np.zeros(self.num_features)
        self.bias = 0


    def train(self, train_X, train_y, dev_X, dev_y):
        self.init_parameters()
        Train_loss = []
        Dev_loss = []
        Train_Acc = []
        Dev_Acc = []
        for iter in range(self.max_iter):
            j = 0
            gradients = np.zeros(self.num_features)
            g_bias = 0
            pred = []
            a_vec = safe_sparse_dot(train_X, self.weights.T) + self.bias
            gradients = 2*(a_vec - train_y)*train_X
            g_bias = np.sum(2*(a_vec - train_y))
            for a in a_vec:
                if a == 0: # Random selection since classes are quite balanced (in case wx + b = 0)
                    j = j + 1
                    if (j % 2 == 1):
                        y_hat = -1 # Classification based on index to approximate random selection and so it is reproduceable
                    else:
                        y_hat = 1 
                else:
                    y_hat = a/abs(a) #Get sign
                pred.append(y_hat)
            
            gradients = gradients - self.lam*self.weights #Regularisation
            
            self.weights = self.weights - self.eta*gradients #Update weights
            self.bias = self.bias - self.eta*g_bias #Update bias
            
            #print("\nIteration", iter+1, 'done')
            train_pred, Loss_train = self.predict(train_X, train_y)
            dev_pred, Loss_dev = self.predict(dev_X, dev_y)
            train_ac = accuracy_score(train_y, pred)
            dev_ac = accuracy_score(dev_y, dev_pred)
            Train_loss.append(Loss_train)
            Dev_loss.append(Loss_dev)
            Train_Acc.append(train_ac)
            Dev_Acc.append(dev_ac)            
        
        return Train_loss, Dev_loss, Train_Acc, Dev_Acc


    def predict(self, X, y=None):
        j = 0
        predicted_labels = []
        a_vec = safe_sparse_dot(X, self.weights.T) + self.bias
        for a in a_vec:
            if a == 0: # Random selection since classes are quite balanced (in case wx + b = 0)
                j = j + 1
                if (j % 2 == 1):
                    y_hat = -1 # Classification based on index to approximate random selection and so it is reproduceable
                else:
                    y_hat = 1
            else:
                y_hat = a/abs(a) #Get sign
            predicted_labels.append(y_hat)
        M = len(y)
        pred_avg_loss = (1/M)*(np.sum(np.square(a_vec - y)) + (self.lam/2)*np.sum(np.square(self.weights))) #Compute avg loss

        return predicted_labels, pred_avg_loss
