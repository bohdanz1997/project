from collections import Counter

class BigramGenerator:
    def __init__(self, min_count=2):
        self.min_count = min_count
        self.bigram_counts = Counter()
        
    def generate_bigrams(self, tokens):
        bigrams = []
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]}_{tokens[i+1]}"
            bigrams.append(bigram)
        return bigrams
    
    def fit(self, token_lists):
        for tokens in token_lists:
            bigrams = self.generate_bigrams(tokens)
            self.bigram_counts.update(bigrams)
    
    def filter_bigrams(self, bigrams):
        return [b for b in bigrams if self.bigram_counts[b] >= self.min_count]
    
    def get_top_bigrams(self, n):
        return [bigram for bigram, _ in self.bigram_counts.most_common(n)]
