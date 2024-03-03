from .act import relu, sigmoid
import numpy as np

class LinearLayer:
    def __init__(self, inp, out, activation=True):
        self.inp = inp
        self.out = out
        self.activation = activation
        self.w = np.random.randn(inp, out)
        self.b = np.random.randn(out)
        
    def forward(self, inp):
        out = np.dot(inp, self.w) + self.b
        if self.activation:
            out = relu(out)
        return out


class AnnModel:
    def __init__(self, input_size=5, layer_size=[4,4,4,4], output_size=1):
        self.input_size = input_size
        self.layer_size = layer_size
        self.output_size = output_size
        self.create_model()

    def create_model(self):
        self.layers = []
        prev = self.input_size
        for i in range(len(self.layers)):
            next = self.layers[i]
            self.layers.append(LinearLayer(prev, next))
            prev = next
        self.layers.append(LinearLayer(prev, self.output_size, activation=False))
    
    def forward(self, inp):
        out = inp
        for i in range(len(self.layers)):
            out = self.layers[i].forward(out)
        out = sigmoid(out)
        return out

if __name__=='__main__':
    inp_size = 10
    inp = np.random.randn(inp_size)
    model = AnnModel(input_size=inp_size, layer_size=[32, 16, 8 , 4, 2], output_size=1)
    out = model.forward(inp)
    print(out)    
    
