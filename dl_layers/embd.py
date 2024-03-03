import numpy as np

class EmbeddingLayer:
    def __init__(self, vocab_size, embedding_size):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.weights = np.random.randn(vocab_size, embedding_size)
    
    def forward(self, input):
        return self.weights[input]
