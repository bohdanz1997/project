class StopWordsRemover:
    def __init__(self, stop_words_file=None):
        self.stop_words = set()
        if stop_words_file:
            try:
                with open(stop_words_file, 'r', encoding='utf-8') as f:
                    self.stop_words = set(line.strip().lower() for line in f)
            except FileNotFoundError:
                pass
        
        self._init_default_stop_words()
    
    def _init_default_stop_words(self):
        default_stops = [
            'і', 'в', 'у', 'на', 'з', 'до', 'від', 'за', 'для', 'по',
            'це', 'той', 'та', 'але', 'або', 'а', 'й', 'не', 'що', 'як'
        ]
        self.stop_words.update(default_stops)
    
    def remove(self, tokens):
        return [t for t in tokens if t not in self.stop_words]
    
    def is_stop_word(self, word):
        return word.lower() in self.stop_words
