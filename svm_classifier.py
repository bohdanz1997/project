from sklearn.svm import SVC
from base_classifier import BaseClassifier
import config

class SVMClassifier(BaseClassifier):
    def __init__(self, C=None, gamma=None, kernel=None):
        super().__init__()
        self.C = C or config.SVM_C
        self.gamma = gamma or config.SVM_GAMMA
        self.kernel = kernel or config.SVM_KERNEL
        self.model = SVC(
            C=self.C,
            gamma=self.gamma,
            kernel=self.kernel,
            probability=True,
            random_state=config.RANDOM_SEED
        )
        
    def train(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True
        
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict_proba(X)
