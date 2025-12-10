import numpy as np
from svm_classifier import SVMClassifier
from nn_classifier import DenseNNClassifier
from base_classifier import BaseClassifier

class HybridClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.svm_model = SVMClassifier()
        self.nn_model = DenseNNClassifier()
        
    def train(self, X, y):
        self.svm_model.train(X, y)
        
        svm_probs = self.svm_model.predict_proba(X)
        
        X_combined = np.hstack([X, svm_probs])
        
        self.nn_model.train(X_combined, y)
        
        self.is_trained = True
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        svm_probs = self.svm_model.predict_proba(X)
        X_combined = np.hstack([X, svm_probs])
        
        return self.nn_model.predict(X_combined)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        svm_probs = self.svm_model.predict_proba(X)
        X_combined = np.hstack([X, svm_probs])
        
        return self.nn_model.predict_proba(X_combined)
