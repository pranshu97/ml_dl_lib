import numpy as np

def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x)

def tanh(x):
    return np.tanh(x)

if __name__=='__main__':
    x = np.array([1, 2, 3])
    print(relu(x))
    print(sigmoid(x))
    print(softmax(x))
    print(tanh(x))