import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import config

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        
    def load(self):
        self.data = pd.read_csv(self.filepath)
        return self.data
    
    def split_data(self, stratify_col='sentiment'):
        if self.data is None:
            raise ValueError("Data not loaded")
            
        X = self.data['text'].values
        y = self.data[stratify_col].map(config.CLASS_MAP).values
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, 
            random_state=config.RANDOM_SEED, stratify=y
        )
        
        val_size_adjusted = config.VAL_SIZE / (1 - config.TEST_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=config.RANDOM_SEED, stratify=y_temp
        )
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def get_class_distribution(self):
        if self.data is None:
            raise ValueError("Data not loaded")
        return self.data['sentiment'].value_counts()
