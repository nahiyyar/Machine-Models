import string
import re
import numpy as np

class BagofWords:
    def __init__(self):
        pass

    def tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())
    
    def build_vocab(self, texts):
        vocab = {}
        i = 0
        for txt in texts:
            for token in self.tokenize(txt):
                if token not in vocab:
                    vocab[token] = i
                    i += 1
        return vocab

    def vectorize(self, texts, vocab):
        bow_matrix = np.zeros((len(texts), len(vocab)), dtype=int)

        for i, val in enumerate(texts):
            tokens = self.tokenize(val)       
            for to in tokens:
                if to in vocab:
                    j = vocab[to]
                    bow_matrix[i, j] += 1
        return bow_matrix
