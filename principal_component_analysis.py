"""
Principal Component Analysis (PCA) Implementation
@author: Roger Fortunato 
@author: Ayo Adetayo aa5886@columbia.edu
@date: 11/29/25

This module implements PCA for dimensionality reduction on the MNIST dataset.
PCA finds the directions of maximum variance in the data and projects the data
onto a lower-dimensional subspace defined by the top k principal components.
"""

import numpy as np
from typing import Tuple


class PCA:
    """
    Principal Component Analysis implementation for dimensionality reduction.
    
    PCA reduces the dimensionality of data by:
    1. Centering the data (subtract mean)
    2. Computing the covariance matrix
    3. Finding eigenvalues and eigenvectors
    4. Selecting the top k eigenvectors (principal components)
    5. Projecting data onto these components
    
    Attributes:
        n_components (int): Number of principal components to retain
        components_ (np.ndarray): Principal component vectors (shape: n_components, n_features)
        mean_ (np.ndarray): Mean of training data (shape: n_features,)
        explained_variance_ (np.ndarray): Variance explained by each component
        explained_variance_ratio_ (np.ndarray): Proportion of variance explained
    """
    
    def __init__(self, n_components: int = 8):
        """
        Initialize PCA.
        
        Args:
            n_components (int): Number of principal components to keep (default: 8)
        """
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        

    def fit(self, X: np.ndarray) -> 'PCA':
        """
        Fit PCA on training data.
        
        Computes the principal components by:
        1. Centering the data
        2. Computing covariance matrix
        3. Computing eigendecomposition
        4. Selecting top n_components eigenvectors
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features)
            
        Returns:
            self: Returns self for method chaining
        """
        # Step 1: Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Step 2: Compute covariance matrix
        # Cov = (1/n) * X^T * X for centered data
        cov_matrix = np.cov(X_centered.T)
        
        # Step 3: Compute eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Step 4: Select top n_components
        self.components_ = eigenvectors[:, :self.n_components].T
        
        # Store explained variance
        self.explained_variance_ = eigenvalues[:self.n_components]
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        return self
    

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encodes data into the PCA latent space.
        This is an alias for transform, provided for symmetry with the autoencoder API.

        Args:
            X (np.ndarray): Data to encode, shape (n_samples, n_features)

        Returns:
            np.ndarray: Latent representation of shape (n_samples, n_components)
        """
        return self.transform(X)
    

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data onto the principal components.
        
        Reduces dimensionality by projecting X onto the learned principal
        component directions: X_transformed = (X - mean) @ components^T
        
        Args:
            X (np.ndarray): Data to transform, shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Transformed data of shape (n_samples, n_components)
        """
        if self.components_ is None:
            raise ValueError("PCA must be fit before transform. Call fit() first.")
        
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and transform data in one step.
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Transformed data of shape (n_samples, n_components)
        """
        return self.fit(X).transform(X)
    

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from principal component projections.
        
        Projects the transformed (low-dimensional) data back to the original
        feature space: X_reconstructed = X_transformed @ components + mean
        
        Args:
            X_transformed (np.ndarray): Transformed data, shape (n_samples, n_components)
            
        Returns:
            np.ndarray: Reconstructed data of shape (n_samples, n_features)
        """
        if self.components_ is None:
            raise ValueError("PCA must be fit before inverse_transform. Call fit() first.")
        
        X_reconstructed = X_transformed @ self.components_
        return X_reconstructed + self.mean_
    

    def get_reconstruction_error(self, X: np.ndarray) -> float:
        """
        Calculate the mean squared reconstruction error.
        
        Measures how well the low-dimensional representation reconstructs
        the original data. Lower values indicate better reconstruction.
        
        Args:
            X (np.ndarray): Original data of shape (n_samples, n_features)
            
        Returns:
            float: Mean squared error between original and reconstructed data
        """
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        mse = np.mean((X - X_reconstructed) ** 2)
        return mse
    

    def get_cumulative_variance_explained(self) -> np.ndarray:
        """
        Get cumulative explained variance ratio.
        
        Returns:
            np.ndarray: Cumulative proportion of variance explained
        """
        if self.explained_variance_ratio_ is None:
            raise ValueError("PCA must be fit first. Call fit() before this method.")
        
        return np.cumsum(self.explained_variance_ratio_)
