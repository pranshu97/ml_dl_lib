import numpy as np
from .act import relu

class ConvLayer:
    def __init__(self, input_channels, output_channels, kernel_size=(3,3), padding=1, stride=1, activation=True):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.activation = activation

        self.kernels = np.random.randn(self.kernel_size[0], self.kernel_size[1], self.input_channels, self.output_channels)
        self.bias = np.random.randn(self.output_channels)

    def forward(self, image):
        batch_size, height, width, channels = image.shape
        new_height = (height + 2*self.padding - self.kernel_size[0])//self.stride + 1
        new_width = (width + 2*self.padding - self.kernel_size[1])//self.stride + 1
        out_shape = (batch_size, new_height, new_width, self.output_channels)

        image = np.pad(image, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant')

        out_image = np.zeros(out_shape)
        for i in range(out_shape[1]):
            for j in range(out_shape[2]):
                img = image[:, i*self.stride:i*self.stride+self.kernel_size[0], j*self.stride:j*self.stride+self.kernel_size[1], :]
                for k in range(out_shape[3]):
                    out_image[:, i, j, k] = np.sum(img * self.kernels[:,:,:,k]) + self.bias[k]
        if self.activation:
            out_image = relu(out_image)
        return out_image

if __name__ == "__main__":
    image = np.random.randn(2, 5, 5, 3)
    conv = ConvLayer(3, 2)
    out = conv.forward(image)
    print(out.shape)
    print(conv.kernels.shape)
    print(conv.bias.shape)
    print(conv.activation)
    print(conv.padding)
    print(conv.stride)
    print(conv.input_channels)
    print(conv.output_channels)
    print(conv.kernel_size)