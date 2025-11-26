#Basic Implementation of Linear Regression from scratch
# @ is used for dot product
import numpy as np


class LinearRegression:
    def __init__(self, epoch = 1000, lr=0.001):
        self.epoch = epoch
        self.lr = lr
        self.m = None
        self.c = None

    def fit(self,x,y):
        n,features = x.shape
        self.m = np.zeros((features,1))
        self.c = 0.0
        
        for i in range(self.epoch):
            y_pred = x @ self.m + self.c
            err = y_pred - y
            #Mean Squared Error
            mse = np.mean(err ** 2)

            # Derivatives of slope and bias
            dm = (2/n) * (x.T @ err)
            dc = (2/n) * np.sum(err)

            #Update slope and bias
            self.m -= self.lr * dm
            self.c -= self.lr * dc

            if i % 10 == 0:
                print(f"Iteration:{i},MSE Loss:{mse}")
        
        print("Training Complete")

    def predict(self,X):
        return X@self.m+self.c
