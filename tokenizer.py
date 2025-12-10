import re

class Tokenizer:
    def __init__(self, min_length=2):
        self.min_length = min_length
        self.pattern = re.compile(r'\b\w+\b')
        
    def tokenize(self, text):
        text = text.lower()
        tokens = self.pattern.findall(text)
        return [t for t in tokens if len(t) >= self.min_length]
    
    def remove_punctuation(self, text):
        return re.sub(r'[^\w\s]', ' ', text)
