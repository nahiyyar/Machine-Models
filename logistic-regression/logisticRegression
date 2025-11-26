import numpy as np
from params import sigmoid

class LogisticRegression:
    def __init__(self,epoch = 100,lr = 0.01):
        self.epoch = epoch
        self.lr = lr
        self.weight = None
        self.bias = None

    def fit(self,x,y):
        n,n_features = x.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epoch):
            linear = x@self.weight + self.bias

            y_pred = sigmoid(linear)

            err = y_pred - y

            dw = (1/n)*(x.T@err)
            db = (1/n)*np.sum(err)

            self.weight -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict_prob(self,x):
        linear = x@self.weight + self.bias
        return sigmoid(linear)
    
    def predict(self,x):
        y_prob = self.predict_prob(x)
        return np.array([1 if p > 0.5 else 0 for p in y_prob])

