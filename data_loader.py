# data_loader.py loads in the raw MNIST data

import numpy as np
from sklearn.datasets import fetch_openml

def load_mnist():
    """
    Loads the mnist dataset via sklearn

    Returns:
        X: Feature matrix (70000, 784) - 784 pixel values across 70,000 images
        y: Image labels vector (70000,) - images labeled as digits 0-9
    """
    # load mnist
    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)

    # extract features and labels
    X = np.array(mnist.data, dtype=np.float64)
    y = np.array(mnist.target, dtype=np.int64)

    return X, y


# Test the function
if __name__ == "__main__":
    X, y = load_mnist()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X data type: {X.dtype}")
    print(f"y data type: {y.dtype}")
    print(f"First label: {y[0]}")
    print(f"Pixel value range: {X.min()} to {X.max()}")
