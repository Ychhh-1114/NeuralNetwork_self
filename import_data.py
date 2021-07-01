from sklearn.datasets import fetch_openml
import numpy as np


def get_mnist():
    mnist = fetch_openml('mnist_784')
    m, n = mnist["data"].shape
    X = np.array(mnist["data"])
    y = np.array(mnist["target"].astype(np.int)).reshape(-1, 1)
    return X,y
