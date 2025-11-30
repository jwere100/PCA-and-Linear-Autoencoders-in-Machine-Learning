"""
Implementation of a set of comparison functions to measure the similarities
and differences of the PCA and LAE representations: subspace simularity and 
reconstruction error. Additional tools to help visualize the experimental results.
@author Nick Meyer njm2179@columbia.edu
11/29/2025

updated by Ayo Adetayo on 11/29/25
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from principal_component_analysis import PCA
from linear_autoencoder import AutoEncoder

# Comparing subspace representations
def compute_principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Computes the principal angles between two subspaces (in radians).
    If all principal angles are 0, the subspaces are identical, and the LAE
    has learned the same representations as the PCA. The principal angles 
    are the k smallest angles between pairs of bases from the PCA and LAE, 
    where each new pair is orthogonal to all previous pairs. 
    
    Args:
        A (np.ndarray): Basis for the first subspace, shape (k, n_features)
        B (np.ndarray): Basis for the second subspace, shape (k, n_features)

    Returns:
        np.ndarray: Principal angles in radians, shape (k,)
    """
    # Orthonormalize both bases via QR decomposition
    Q_A, _ = np.linalg.qr(A.T)
    Q_B, _ = np.linalg.qr(B.T)

    # Compute SVD of Q_A.T @ Q_B - singular vals are cos(principal_angles)
    _, singular_values, _ = np.linalg.svd(Q_A.T @ Q_B)

    # Clip for float rounding errors
    singular_values = np.clip(singular_values, -1, 1)

    # Get principal angles
    principal_angles = np.arccos(singular_values)

    return principal_angles

def subspace_distance(A: np.ndarray, B: np.ndarray) -> float:
    """
    Computes the Grassmann distance between the two subspaces using the
    principal angles. It is simply the square root of the sum of squares of
    the principal angles between two equidimensional subspaces. If Grassmann
    distance = 0, the subspaces are identical. Obviously, this only occurs when 
    all principal angles are also 0. 

    Args:
        A (np.ndarray): Basis for the first subspace, shape (k, num_features)
        B (np.ndarray): Basis for the second subspace, shape (k, num_features)

    Returns: 
        float: Grassman distance between subspaces
    """
    principal_angles = compute_principal_angles(A, B)
    return np.sqrt(np.sum(principal_angles ** 2))

def compute_subspace_similarity(pca: PCA, autoencoder: AutoEncoder) -> dict:
    """
    Calls the helper functions to compute the subspace similarity metrics
    between the PCA and the LAE. 
    
    Args:
        pca (PCA): Fitted PCA model
        autoencoder (AutoEncoder): Trained linear autoencoder model
        
    Returns:
        dict: Dictionary containing:
            - 'principal_angles_rad': Array of the prinical angles in radians
            - 'principal_angles_deg': Array of the prinical angles in degrees
            - 'grassmann_distance': Subspace distance
            - 'mean_angle_degrees': Mean principal angle in degrees 
    """
    pca_components = pca.components_                # Shape: (k, 784)
    lae_weights = autoencoder.get_encoder_weights() # Shape: (k, 784)

    angles = compute_principal_angles(pca_components, lae_weights)

    return {
        'principal_angles_rad': angles,
        'principal_angles_deg': np.degrees(angles),
        'grassmann_distance': subspace_distance(pca_components, lae_weights),
        'mean_angle_degrees': np.mean(np.degrees(angles))
    }

# Comparing reconstruction error
def compare_reconstruction_error(
        X: np.ndarray, pca: PCA, autoencoder: AutoEncoder) -> dict:
    """
    Compares the reconstruction error of the PCA and LAE.
    
    Args:
        X (np.ndarray): Input data, shape (num_samples, 784)
        pca (PCA): Fitted PCA model
        autoencoder (AutoEndocer): Trained linear autoencoder model
    Returns:
        dict: Dictionary containing:
            - 'pca_mse': PCA reconstruction MSE
            - 'lae_mse': LAE reconstruction MSE
            - 'difference': lae_mse - pca_mse
            - 'ratio': lae_mse / pca_mse
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pca_mse = pca.get_reconstruction_error(X)
    lae_mse = autoencoder.get_reconstruction_error(X, device)

    return {
        'pca_mse': pca_mse,
        'lae_mse': lae_mse,
        'difference': lae_mse - pca_mse,
        'ratio': lae_mse / pca_mse if pca_mse > 0 else float('inf')
    }

# Visualization
def plot_reconstruction_comparison(
        X: np.ndarray, 
        pca: PCA, 
        autoencoder: AutoEncoder, 
        num_images: int = 5, 
        path: str = None
) -> None:
    """
    Plots the original images against PCA and LAE reconstructions.

    Args:
        X (np.ndarray): Original images, shape (num_samples, 784)
        pca (PCA): Fitted PCA model
        autoencoder (AutoEncoder): Trained LAE model
        num_images (int): Number of images to plot for comparison
        path (str): Path to save the plot (optional)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get PCA reconstruction
    X_pca = pca.inverse_transform(pca.transform(X[:num_images]))
    
    # Get LAE reconstruction
    autoencoder.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X[:num_images], dtype=torch.float32).to(device)
        X_lae = autoencoder(X_tensor).cpu().numpy()
    
    # Plot results
    fig, axes = plt.subplots(3, num_images, figsize = (2 * num_images, 6))

    for i in range(num_images):
        # Original
        axes[0, i].imshow(X[i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize = 10)
        
        # PCA
        axes[1, i].imshow(X_pca[i].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('PCA', fontsize = 10)

        # LAE
        axes[2, i].imshow(X_lae[i].reshape(28, 28), cmap='gray')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('LAE', fontsize = 10)
        
    plt.tight_layout()

    if path:
        plt.savefig(path, dpi = 150, bbox_inches = 'tight')
    
    plt.show()

def plot_components_comparison(
        pca: PCA, 
        autoencoder: AutoEncoder, 
        num_representations: int = 10,
        path: str = None
) -> None:
    """
    Visualizes the PCA vs LAE representations 
    (PCA componenets vs LAE weights)
    
    Args:
        pca (PCA): Fitted PCA model
        autoencoder (AutoEncoder): Trained LAE model
        num_representations (int): Number of representations to display
        path (str): Path to save the figure (optional)
    """
    pca_comp = pca.components_[:num_representations]
    lae_weights = autoencoder.get_encoder_weights()[:num_representations]

    fig, axes = plt.subplots(2, num_representations, figsize = (2 * num_representations, 4))

    for i in range(num_representations):
        # PCA
        axes[0, i].imshow(pca_comp[i].reshape(28, 28), cmap = 'gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('PCA', fontsize = 10)

        # LAE
        axes[1, i].imshow(lae_weights[i].reshape(28, 28), cmap = 'gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('LAE', fontsize = 10)

    plt.tight_layout()

    if path:
        plt.savefig(path, dpi = 150, bbox_inches = 'tight')
    plt.show()

def plot_principal_angles(angles_degrees: np.ndarray, path: str = None) -> None:
    """
    Plots the principal angles between PCA and LAE subspaces using the output array
    of principal angles in degrees from the compute_subspace_similarity function.
    
    Args: 
        angles_degrees (np.ndarray): Principal angles in degrees
        path (str): Path to save ploat (optional)
    """
    plt.figure(figsize = (10, 6))

    representations = range(1, len(angles_degrees) + 1)
    plt.bar(representations, angles_degrees, color = 'steelblue', alpha = 0.8)

    plt.xlabel('Representation')
    plt.ylabel('Principal Angle (degrees)')
    plt.title('Principal Angles between PCA and LAE Subspaces')
    plt.axhline(y = 0, color = 'k', linestyle = '-', linewidth = 0.5)
    
    # Reference line at orthogonality (90 degrees)
    plt.axhline(y = 90, color = 'r', linestyle = '--', linewidth = 1, alpha = 0.5, label = 'Orthogonal')
    plt.legend()
    plt.grid(True, alpha = 0.3, axis = 'y')

    if path:
        plt.savefig(path, dpi = 150, bbox_inches = 'tight')
    plt.show()

def compare_classification_performance(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        pca: PCA,
        autoencoder: AutoEncoder,
        device: torch.device,
) -> Dict[str, float]:
    """
    Compares downstream classification performance using three feature spaces:
      1) Raw pixel space (784 dimensions)
      2) PCA latent space (k dimensions)
      3) Autoencoder latent space (k dimensions)

    A simple linear classifier (logistic regression) is trained on each feature
    representation using the same train and test splits.

    Args:
        X_train (np.ndarray): Training images, shape (n_train, 784)
        y_train (np.ndarray): Training labels, shape (n_train,)
        X_test (np.ndarray): Test images, shape (n_test, 784)
        y_test (np.ndarray): Test labels, shape (n_test,)
        pca (PCA): Fitted PCA model
        autoencoder (AutoEncoder): Trained autoencoder model
        device (torch.device): Device for autoencoder encoding

    Returns:
        Dict[str, float]: Dictionary containing classification accuracies:
            - "raw_accuracy"
            - "pca_accuracy"
            - "ae_accuracy"
    """
    # 1. Raw pixel features
    clf_raw = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf_raw.fit(X_train, y_train)
    y_pred_raw = clf_raw.predict(X_test)
    acc_raw = accuracy_score(y_test, y_pred_raw)

    # 2. PCA features
    Z_train_pca = pca.transform(X_train)
    Z_test_pca = pca.transform(X_test)

    clf_pca = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf_pca.fit(Z_train_pca, y_train)
    y_pred_pca = clf_pca.predict(Z_test_pca)
    acc_pca = accuracy_score(y_test, y_pred_pca)

    # 3. Autoencoder latent features
    Z_train_ae = autoencoder.encode(X_train, device)
    Z_test_ae = autoencoder.encode(X_test, device)

    clf_ae = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf_ae.fit(Z_train_ae, y_train)
    y_pred_ae = clf_ae.predict(Z_test_ae)
    acc_ae = accuracy_score(y_test, y_pred_ae)

    return {
        "raw_accuracy": acc_raw,
        "pca_accuracy": acc_pca,
        "ae_accuracy": acc_ae,
    }

def summarize_efficiency(
        pca_fit_time: float,
        pca_transform_time: float,
        ae_train_time: float,
        ae_encode_time: float | None = None,
) -> None:
    """
    Prints a small summary of computational efficiency for PCA and the autoencoder.

    Args:
        pca_fit_time (float): Time spent fitting PCA in seconds
        pca_transform_time (float): Time spent transforming data with PCA in seconds
        ae_train_time (float): Time spent training the autoencoder in seconds
        ae_encode_time (float | None): Optional time spent encoding data with AE in seconds
    """
    print("\n--- Computational Efficiency ---")
    print(f"PCA fit time:          {pca_fit_time:.3f} seconds")
    print(f"PCA transform time:    {pca_transform_time:.3f} seconds")
    print(f"AE training time:      {ae_train_time:.3f} seconds")
    if ae_encode_time is not None:
        print(f"AE encode time:        {ae_encode_time:.3f} seconds")



# Summary report
def generate_report(
        X: np.ndarray,
        pca: PCA,
        autoencoder: AutoEncoder,
        epoch_losses: List[float],
        classification_results: Dict[str, float] | None = None,
        efficiency_stats: Dict[str, float] | None = None,
) -> None:
    """
    Generates a comprehensive comparison report.
    
    Args:
        X (np.ndarray): Test data for evaluation
        pca (PCA): Fitted PCA model
        autoencoder (AutoEncoder): Trained LAE model
        epoch_losses (list): LAE training losses
    """
    print("=" * 60)
    print("PCA vs Linear Autoencoder Comparison Report")
    print("=" * 60)
    
    # Reconstruction error comparison
    recon = compare_reconstruction_error(X, pca, autoencoder)
    print(f"\n--- Reconstruction Error (MSE) ---")
    print(f"PCA:         {recon['pca_mse']:.6f}")
    print(f"Autoencoder: {recon['lae_mse']:.6f}")
    print(f"Difference:  {recon['difference']:.6f}")
    print(f"Ratio:       {recon['ratio']:.4f}")
    
    # Subspace similarity
    similarity = compute_subspace_similarity(pca, autoencoder)
    print(f"\n--- Subspace Similarity ---")
    print(f"Mean Principal Angle:    {similarity['mean_angle_degrees']:.2f}°")
    print(f"Grassmann Distance:      {similarity['grassmann_distance']:.4f}")
    
    # LAE Training info
    print(f"\n--- Training Info ---")
    print(f"Final AE Loss:  {epoch_losses[-1]:.6f}")
    print(f"Total Epochs:   {len(epoch_losses)}")
    
    print("\n" + "=" * 60)

    # Classification performance
    if classification_results is not None:
        print(f"\n--- Downstream Classification (Accuracy) ---")
        print(f"Raw pixels accuracy:       {classification_results['raw_accuracy']:.4f}")
        print(f"PCA features accuracy:     {classification_results['pca_accuracy']:.4f}")
        print(f"AE features accuracy:      {classification_results['ae_accuracy']:.4f}")
    
    # Computational efficiency 
    if efficiency_stats is not None:
        print(f"\n--- Computational Efficiency ---")
        if "pca_fit_time" in efficiency_stats:
            print(f"PCA fit time:              {efficiency_stats['pca_fit_time']:.3f} seconds")
        if "pca_transform_time" in efficiency_stats:
            print(f"PCA transform time:        {efficiency_stats['pca_transform_time']:.3f} seconds")
        if "ae_train_time" in efficiency_stats:
            print(f"Autoencoder training time: {efficiency_stats['ae_train_time']:.3f} seconds")
        if "ae_encode_time" in efficiency_stats:
            print(f"Autoencoder encode time:   {efficiency_stats['ae_encode_time']:.3f} seconds")

