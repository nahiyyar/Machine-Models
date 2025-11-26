import string
import re
import numpy as np

class BagofWords:
    def __init__(self):
        pass

    def tokenize(self, text):
        # lowercase + keep only word characters
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
            tokens = self.tokenize(val)       # ✅ tokenize each sentence
            for to in tokens:
                if to in vocab:
                    j = vocab[to]
                    bow_matrix[i, j] += 1
        return bow_matrix
text = """Beans. I was trying to explain to somebody as we were flying in, that's corn.  
        That's beans. And they were very impressed at my agricultural knowledge. Please 
        give it up for Amaury once again for that outstanding introduction. I have a bunch 
        of good friends here today, including somebody who I served with, who is one of the
        finest senators in the country, and we're lucky to have him, your Senator, Dick Durbin is here. 
        I also noticed, by the way, former Governor Edgar here, who I haven't seen in a long time, and 
        somehow he has not aged and I have. And it's great to see you, Governor. I want to thank President 
        Killeen and everybody at the U of I System for making it possible for me to be here today. And I am 
        deeply honored at the Paul Douglas Award that is being given to me. He is somebody who set the path 
        for so much outstanding public service here in Illinois. Now, I want to start by addressing the elephant 
        in the room. I know people are still wondering why I didn't speak at the commencement."""

nlp = BagofWords()

sentences = [s.strip() for s in text.split('.') if s.strip()]

vocab = nlp.build_vocab(sentences)
bow = nlp.vectorize(sentences, vocab)

print("Vocab size:", len(vocab))
print("Matrix shape:", bow.shape)
print(bow)