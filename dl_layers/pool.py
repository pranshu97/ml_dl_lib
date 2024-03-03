import numpy as np

class PoolLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, image):
        batch_size, height, width, channels = image.shape
        new_height = (height - self.pool_size)//self.stride + 1
        new_width = (width - self.pool_size)//self.stride + 1
        out_shape = (batch_size, new_height, new_width, channels)

        out_image = np.zeros(out_shape)
        for i in range(out_shape[1]):
            for j in range(out_shape[2]):
                img = image[:, i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size, :]
                out_image[:, i, j, :] = np.max(img) 
        return out_image

class GlobalAveragePoolLayer:
    def __init__(self):
        pass

    def forward(self, image):
        batch_size, height, width, channels = image.shape
        out_image = np.mean(image, axis=(1, 2))  # Compute mean over height and width dimensions
        return out_image

if __name__ == "__main__":
    image = np.random.randn(2, 8, 8, 3)
    pool = PoolLayer(2, 2)
    out = pool.forward(image)
    print(out.shape)
    print(pool.pool_size)
    print(pool.stride)