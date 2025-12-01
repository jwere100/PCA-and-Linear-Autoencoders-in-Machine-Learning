"""
Data Loader for MNIST Dataset
@author: Ayo Adetayo aa5886@columbia.edu
@date: 11/29/25

Modified by Nick Meyer on 12/01/25
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple

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


def load_mnist_split(
    test_size: float = 0.2,
    random_state: int = 0,
    normalize: bool = True,
    max_samples: Optional[int] = None,
):
    """
    Loads MNIST and returns a reproducible train test split, with optional
    normalization and subsampling.

    Args:
        test_size (float): Fraction of the data to use as test set.
                           For example 0.2 means 20 percent test data.
        random_state (int): Seed for reproducible shuffling and splitting.
        normalize (bool): If True, scales pixel values to [0, 1] by dividing by 255.
        max_samples (int | None): If not None, randomly selects this many samples
                                  from the full dataset before splitting. This can
                                  be useful for quicker experiments.

    Returns:
        X_train (np.ndarray): Training features of shape (n_train, 784).
        X_test  (np.ndarray): Test features of shape (n_test, 784).
        y_train (np.ndarray): Training labels of shape (n_train,).
        y_test  (np.ndarray): Test labels of shape (n_test,).
    """
    X, y = load_mnist()

    # Optional subsampling for faster experiments
    if max_samples is not None and max_samples < len(y):
        rng = np.random.default_rng(seed=random_state)
        indices = rng.choice(len(y), size=max_samples, replace=False)
        X = X[indices]
        y = y[indices]

    # Normalize to [0, 1] if requested
    if normalize:
        X = X.astype(np.float32) / 255.0
    else:
        X = X.astype(np.float32)

    # Stratified train test split so label distribution is balanced
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test