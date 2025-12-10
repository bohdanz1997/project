from abc import ABC, abstractmethod

class BaseClassifier(ABC):
    def __init__(self):
        self.model = None
        self.is_trained = False
        
    @abstractmethod
    def train(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    def predict_proba(self, X):
        raise NotImplementedError("Predict proba not implemented")
