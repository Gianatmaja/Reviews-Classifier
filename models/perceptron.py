import numpy as np
from sklearn.utils.extmath import safe_sparse_dot

from sklearn.metrics import accuracy_score

from utils import compute_average_accuracy
import random


class Perceptron():
    def __init__(self, num_epochs, num_features, averaged):
        super().__init__()
        self.num_epochs = num_epochs
        self.averaged = averaged
        self.num_features = num_features
        self.weights = None
        self.bias = None

    def init_parameters(self):
        self.weights = np.zeros(self.num_features)
        self.bias = 0

    def train(self, train_X, train_y, dev_X, dev_y, shuffle = False):
        self.init_parameters()
        
        Acc_train_pctrn = []
        Dev_train_pctrn = []
        Order = np.arange(len(train_y))
        for epoch in range(self.num_epochs):
            if shuffle == True:
                random.shuffle(Order)
            else:
                Order = Order
            preds = []
            true_y = []
            for i in Order:
                X = train_X[i]
                y = train_y[i]
                true_y.append(y)
                a = safe_sparse_dot(X, self.weights.T)[0] + self.bias
                if a == 0: # Random selection since classes are quite balanced (in case wx + b = 0)
                    if (i % 2 == 1):
                        y_hat = -1 # Classification based on index to approximate random selection and so it is reproduceable
                    else:
                        y_hat = 1
                else:
                    y_hat = a/abs(a) #Get sign
                    y_hat = y_hat[0,0]
                preds.append(y_hat)
                if (y*a <= 0):
                    self.weights = self.weights + X*y #Update weights
                    self.bias = self.bias + y #Update bias
            print("\nEpoch", epoch+1, 'done')
            dev_pred = self.predict(dev_X)
            acc_t = accuracy_score(true_y, preds)
            acc_d = accuracy_score(dev_y, dev_pred)
            Acc_train_pctrn.append(acc_t)
            Dev_train_pctrn.append(acc_d)
            
        return Acc_train_pctrn, Dev_train_pctrn
        
        

    def predict(self, X):
        j = 0
        predicted_labels = []
        for i in range(X.shape[0]):
            x = X[i]
            a = safe_sparse_dot(x, self.weights.T)[0] + self.bias
            if a == 0:
                j = j + 1
                if (j % 2 == 1):
                    y_hat = -1 # Classification based on index to approximate random selection and so it is reproduceable
                else:
                    y_hat = 1
            else:
                y_hat = a/abs(a) #Get sign
                y_hat = y_hat[0,0]
                    
            predicted_labels.append(y_hat)

        return predicted_labels
