from dl_layers import ConvLayer, PoolLayer, GlobalAveragePoolLayer, DropoutLayer, LinearLayer, softmax
import numpy as np

class CNN:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.layers = [
            ConvLayer(3, 32, (7,7), padding=3, stride=2, activation=True),
            ConvLayer(32, 64, (3, 3), padding=1, stride=1, activation=True),
            PoolLayer(2, 2),
            ConvLayer(64, 128, (1, 1), padding=0, stride=1, activation=True),
            PoolLayer(2, 2),
            GlobalAveragePoolLayer(),
            DropoutLayer(0.5),
            LinearLayer(128, num_classes)
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        x = softmax(x)
        return x

if __name__=='__main__':
    inp_shape = (1, 32, 32, 3)
    inp = np.random.rand(*inp_shape)
    model = CNN(inp_shape, 10)
    out = model.forward(inp)
    print(out.shape)
    print(out)
    print(np.sum(out))