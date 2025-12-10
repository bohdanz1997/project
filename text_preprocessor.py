from tokenizer import Tokenizer
from stop_words import StopWordsRemover
from normalizer import Normalizer
import config

class TextPreprocessor:
    def __init__(self, stop_words_file=None):
        self.tokenizer = Tokenizer()
        self.stop_words_remover = StopWordsRemover(
            stop_words_file or config.STOP_WORDS_FILE
        )
        self.normalizer = Normalizer()
        
    def process(self, text):
        text = self.tokenizer.remove_punctuation(text)
        tokens = self.tokenizer.tokenize(text)
        tokens = self.stop_words_remover.remove(tokens)
        tokens = self.normalizer.normalize(tokens)
        return tokens
    
    def process_batch(self, texts):
        return [self.process(text) for text in texts]
