import numpy as np

class DropoutLayer:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
    
    def forward(self, x, training=True):
        if training:
            mask = np.random.rand(*x.shape)>self.dropout_ratio
            return x*mask
        else:
            return x*(1-self.dropout_ratio)

if __name__=='__main__':
    inp = np.random.rand(2,2)
    print(inp)
    print(DropoutLayer().forward(inp))