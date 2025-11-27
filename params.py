import numpy as np

def sigmoid(self,z):
    return 1/(1+np.exp(-z))
    
def relu(self,z):
    return max(0,z)
    
def tanh(self,z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
    
def softmax(self,z):
    return np.exp(z)/np.sum(np.exp(z))
    
