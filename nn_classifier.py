import numpy as np
from base_classifier import BaseClassifier
import config

class DenseNNClassifier(BaseClassifier):
    def __init__(self, hidden_size=None, epochs=None, learning_rate=None):
        super().__init__()
        self.hidden_size = hidden_size or config.NN_HIDDEN_SIZE
        self.epochs = epochs or config.NN_EPOCHS
        self.learning_rate = learning_rate or config.NN_LEARNING_RATE
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _one_hot_encode(self, y, num_classes):
        one_hot = np.zeros((len(y), num_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot
    
    def train(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        self.W1 = np.random.randn(n_features, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, n_classes) * 0.01
        self.b2 = np.zeros((1, n_classes))
        
        y_one_hot = self._one_hot_encode(y, n_classes)
        
        for epoch in range(self.epochs):
            z1 = np.dot(X, self.W1) + self.b1
            a1 = self._relu(z1)
            z2 = np.dot(a1, self.W2) + self.b2
            a2 = self._softmax(z2)
            
            loss = -np.mean(np.sum(y_one_hot * np.log(a2 + 1e-8), axis=1))
            
            dz2 = a2 - y_one_hot
            dW2 = np.dot(a1.T, dz2) / n_samples
            db2 = np.sum(dz2, axis=0, keepdims=True) / n_samples
            
            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * self._relu_derivative(z1)
            dW1 = np.dot(X.T, dz1) / n_samples
            db1 = np.sum(dz1, axis=0, keepdims=True) / n_samples
            
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
        
        self.is_trained = True
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained")
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self._relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        return self._softmax(z2)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
