import numpy as np
from collections import Counter
import config

class TFIDFVectorizer:
    def __init__(self, max_features=None, min_df=2, max_df=0.8):
        self.max_features = max_features or config.MAX_FEATURES
        self.min_df = min_df
        self.max_df = max_df
        self.vocabulary = {}
        self.idf_values = {}
        self.n_docs = 0
        
    def fit(self, token_lists):
        self.n_docs = len(token_lists)
        doc_freq = Counter()
        
        for tokens in token_lists:
            unique_tokens = set(tokens)
            doc_freq.update(unique_tokens)
        
        filtered_tokens = [
            token for token, freq in doc_freq.items()
            if self.min_df <= freq <= self.max_df * self.n_docs
        ]
        
        sorted_tokens = sorted(
            filtered_tokens, 
            key=lambda t: doc_freq[t], 
            reverse=True
        )[:self.max_features]
        
        self.vocabulary = {token: idx for idx, token in enumerate(sorted_tokens)}
        
        for token in self.vocabulary:
            self.idf_values[token] = np.log(self.n_docs / doc_freq[token])
    
    def transform(self, token_lists):
        vectors = []
        for tokens in token_lists:
            vector = self._transform_single(tokens)
            vectors.append(vector)
        return np.array(vectors)
    
    def _transform_single(self, tokens):
        vector = np.zeros(len(self.vocabulary))
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        
        for token, count in token_counts.items():
            if token in self.vocabulary:
                tf = count / total_tokens
                idf = self.idf_values[token]
                idx = self.vocabulary[token]
                vector[idx] = tf * idf
        
        return vector
    
    def fit_transform(self, token_lists):
        self.fit(token_lists)
        return self.transform(token_lists)
