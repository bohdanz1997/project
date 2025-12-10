from tfidf_vectorizer import TFIDFVectorizer
from bigram_generator import BigramGenerator
import numpy as np
import config

class VectorizerManager:
    def __init__(self):
        self.tfidf = TFIDFVectorizer(
            max_features=config.MAX_FEATURES,
            min_df=config.MIN_DF,
            max_df=config.MAX_DF
        )
        self.bigram_generator = BigramGenerator()
        self.top_bigrams = []
        
    def fit(self, token_lists):
        combined_lists = []
        
        self.bigram_generator.fit(token_lists)
        self.top_bigrams = self.bigram_generator.get_top_bigrams(config.MAX_BIGRAMS)
        
        for tokens in token_lists:
            bigrams = self.bigram_generator.generate_bigrams(tokens)
            filtered_bigrams = [b for b in bigrams if b in self.top_bigrams]
            combined = tokens + filtered_bigrams
            combined_lists.append(combined)
        
        self.tfidf.fit(combined_lists)
        
    def transform(self, token_lists):
        combined_lists = []
        
        for tokens in token_lists:
            bigrams = self.bigram_generator.generate_bigrams(tokens)
            filtered_bigrams = [b for b in bigrams if b in self.top_bigrams]
            combined = tokens + filtered_bigrams
            combined_lists.append(combined)
        
        return self.tfidf.transform(combined_lists)
    
    def fit_transform(self, token_lists):
        self.fit(token_lists)
        return self.transform(token_lists)
    
    def get_vocabulary_size(self):
        return len(self.tfidf.vocabulary)
