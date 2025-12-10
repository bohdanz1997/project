class Normalizer:
    def __init__(self):
        self.suffixes = ['ував', 'ював', 'ував', 'али', 'ять', 'ить', 'еть']
        
    def normalize(self, tokens):
        return [self.stem(token) for token in tokens]
    
    def stem(self, word):
        for suffix in self.suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word
    
    def to_lower_case(self, tokens):
        return [t.lower() for t in tokens]
