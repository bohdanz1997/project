import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
        self.confusion_mat = None
        
    def evaluate(self, y_true, y_pred):
        self.metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro'),
            'recall': recall_score(y_true, y_pred, average='macro'),
            'f1_score': f1_score(y_true, y_pred, average='macro')
        }
        
        self.confusion_mat = confusion_matrix(y_true, y_pred)
        
        return self.metrics
    
    def evaluate_per_class(self, y_true, y_pred):
        precision = precision_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)
        f1 = f1_score(y_true, y_pred, average=None)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def display_results(self):
        print("\n=== Model Evaluation Results ===")
        print(f"Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall: {self.metrics['recall']:.4f}")
        print(f"F1-Score: {self.metrics['f1_score']:.4f}")
        
        if self.confusion_mat is not None:
            print("\nConfusion Matrix:")
            print(self.confusion_mat)
