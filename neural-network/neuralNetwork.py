
import numpy as np


class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size, 
                 learning_rate=0.01, epochs=1000, activation='relu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation = activation

        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
    
    def _relu(self, z):
        return np.maximum(0, z)
    
    def _relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def _sigmoid_derivative(self, z):
        sig = self._sigmoid(z)
        return sig * (1 - sig)
    
    def _activate(self, z):
        if self.activation == 'relu':
            return self._relu(z)
        else:
            return self._sigmoid(z)
    
    def _activate_derivative(self, z):
        if self.activation == 'relu':
            return self._relu_derivative(z)
        else:
            return self._sigmoid_derivative(z)
    
    def _softmax(self, z):

        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self._activate(self.z1)
        
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self._softmax(self.z2)
        
        return self.a1, self.a2
    
    def backward(self, X, y):
        
        m = X.shape[0]
    
        dz2 = self.a2 - y
        dW2 = (1/m) * (self.a1.T @ dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self._activate_derivative(self.z1)
        dW1 = (1/m) * (X.T @ dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def fit(self, X, y):
        
        if y.ndim == 1:
            n_classes = len(np.unique(y))
            y_onehot = np.zeros((y.shape[0], n_classes))
            y_onehot[np.arange(y.shape[0]), y] = 1
            y = y_onehot
        
        for epoch in range(self.epochs):
            _, output = self.forward(X)
            self.backward(X, y)
            
            if (epoch + 1) % 100 == 0:
                # Calculate loss (cross-entropy)
                loss = -np.mean(y * np.log(output + 1e-8))
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}")
        
        print("Training Complete")
    
    def predict_proba(self, X):
        
        _, output = self.forward(X)
        return output
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
