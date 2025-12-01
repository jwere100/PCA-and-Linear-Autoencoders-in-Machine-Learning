"""
AutoEncoder Implementation via Pytorch
@author: Joshua Were jow2112@columbia.edu
@author: Nick Meyer njm2179@columbia.edu
@date: 11/28/25

Modified by Ayo Adetayo on 11/29/25
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple
import time
import random


def set_seed(seed: int = 0) -> None:
    """
    Sets seeds for Python, NumPy, and PyTorch to make experiments reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AutoEncoder(nn.Module):
    """
    Linear Autoencoder class for dimensionality reduction on MNIST.
    This single-layer autoencoder learns to compress the 784-dim images into a
    k-dim bottleneck representation (encoder) and to reconstruct the 784-dim
    images (decoder).

    Configurable to match Baldi & Hornik (1989) conditions for subspace convergence:
    - use_bias: Whether to include bias terms (default: False)
    - tied_weights: Whether decoder = encoder.T (default: True)

    Args:
        k (int): Bottleneck dimensionality (number of latent features)
        use_bias (bool): Whether to use bias terms in encoder/decoder
        tied_weights (bool): Whether to tie decoder weights to encoder.T

    Attributes:
        encoder (nn.Linear): Linear layer mapping 784 -> k
        decoder (nn.Linear or None): Linear layer mapping k -> 784 (None if tied_weights=True)
        k (int): Bottleneck dimensionality
        use_bias (bool): Whether biases are used
        tied_weights (bool): Whether weights are tied
        mean_ (np.ndarray): Mean of training data (stored after training for centered comparison)
    """
    def __init__(self, k, use_bias=False, tied_weights=True):
        super().__init__()
        self.k = k
        self.use_bias = use_bias
        self.tied_weights = tied_weights

        # Create encoder
        self.encoder = nn.Linear(784, k, bias=use_bias)

        # Create decoder only if weights are not tied
        if not tied_weights:
            self.decoder = nn.Linear(k, 784, bias=use_bias)
        else:
            self.decoder = None

        self.mean_ = None

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode then decode.

        If tied_weights=True, uses decoder = encoder.T
        If tied_weights=False, uses separate decoder layer

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 784)

        Returns:
            torch.Tensor: Reconstructed tensor of shape (batch_size, 784)
        """
        # Encode
        latent = self.encoder(x)

        # Decode
        if self.tied_weights:
            # Use tied weights: decoder = encoder.T
            reconstructed = torch.matmul(latent, self.encoder.weight)
            # Add encoder bias if it exists (for decoding it's the "output" bias)
            if self.use_bias:
                reconstructed = reconstructed + self.encoder.bias
        else:
            # Use separate decoder
            reconstructed = self.decoder(latent)

        return reconstructed
    
    def get_encoder_weights(self) -> np.ndarray:
        """
        Extract encoder weights as a numpy array.
        
        The encoder weight matrix W_E has shape (k, 784), where each row 
        represents a learned basis vector in the latent space. These can 
        be compared to the PCA eigenvectors.
        
        Returns:
            np.ndarray: Encoder weights of shape (k, 784)
        """
        return self.encoder.weight.data.cpu().numpy()
    
    def get_decoder_weights(self) -> np.ndarray:
        """
        Extract decoder weights as numpy array.

        If tied_weights=True, returns encoder.T
        If tied_weights=False, returns separate decoder weights

        The decoder weight matrix W_D has shape (784, k), where each column
        represents a learned basis vector for image reconstruction.

        Returns:
            np.ndarray: Decoder weights of shape (784, k)
        """
        if self.tied_weights:
            # Return transpose of encoder weights
            return self.encoder.weight.data.cpu().numpy().T
        else:
            # Return separate decoder weights
            return self.decoder.weight.data.cpu().numpy()

    def get_encoder_bias(self) -> np.ndarray:
        """
        Extracts encoder bias as a numpy array.

        Returns:
            np.ndarray: Encoder biases of shape (k,), or zeros if use_bias=False
        """
        if self.use_bias:
            return self.encoder.bias.data.cpu().numpy()
        else:
            return np.zeros(self.k)

    def get_decoder_bias(self) -> np.ndarray:
        """
        Extracts decoder bias as a numpy array.

        Returns:
            np.ndarray: Decoder biases of shape (784,), or zeros if use_bias=False
        """
        if self.use_bias:
            if self.tied_weights:
                # With tied weights and bias, use encoder bias as output bias
                return self.encoder.bias.data.cpu().numpy()
            else:
                return self.decoder.bias.data.cpu().numpy()
        else:
            return np.zeros(784)
    
    def get_reconstruction_error(self, X: np.ndarray, device: torch.device) -> float:
        """
        Calculates the reconstruction error on the data (MSE).

        Args:
            X (np.ndarray): Input data of shape (num_samples, 784)
            device (torch.device): Device to perform computations on

        Returns:
            float: MSE between original and reconstructed data 
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            X_reconstructed = self.forward(X_tensor)
            mse = torch.mean((X_tensor - X_reconstructed) ** 2).item()
        return mse
    
    def encode(self, X: np.ndarray, device: torch.device) -> np.ndarray:
        """
        Encodes NumPy data into the k dimensional latent space.

        Args:
            X (np.ndarray): Input data of shape (num_samples, 784)
            device (torch.device): Device to perform computations on

        Returns:
            np.ndarray: Latent codes of shape (num_samples, k)
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            Z = self.encoder(X_tensor)
        return Z.cpu().numpy()


def train_autoencoder(
        X_train: np.ndarray,
        k: int = 50,
        batch_size: int = 64,
        epochs: int = 30,
        learning_rate: float = 0.001,
        seed: int = 0,
        use_bias: bool = False,
        tied_weights: bool = True,
) -> Tuple[AutoEncoder, list, list, float]:
    """
    Trains a linear autoencoder on provided training data.

    Args:
        X_train (np.ndarray): Training data of shape (n_samples, 784), normalized to [0, 1]
        k (int): Bottleneck dimensionality
        batch_size (int): Training batch size
        epochs (int): Number of training epochs
        learning_rate (float): Optimizer learning rate
        seed (int): Random seed for reproducibility
        use_bias (bool): Whether to use bias terms (Baldi & Hornik requires False)
        tied_weights (bool): Whether decoder = encoder.T (Baldi & Hornik requires True)

    Returns:
        model (Autoencoder): Trained autoencoder model
        epoch_losses (list): average loss per epoch
        outputs (list): Stored tuples for visualization (epoch, input, reconstruction)
        total_train_time (float): Total training time in seconds
    """

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    print(f"Training data shape: {X_train.shape}")
    print(f"Model architecture: 784 → {k} → 784")
    print(f"Configuration: use_bias={use_bias}, tied_weights={tied_weights}")
    print(f"Hyperparameters: batch_size={batch_size}, epochs={epochs}, lr={learning_rate}")

    # Set random seeds for reproducibility
    set_seed(seed)

    # Measure total training time
    train_start = time.time()

    # Create PyTorch dataset from numpy array
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Compute and store training data mean (for fair comparison with PCA)
    data_mean = np.mean(X_train, axis=0)

    # Initialize model with configuration
    model = AutoEncoder(k=k, use_bias=use_bias, tied_weights=tied_weights).to(device)
    model.mean_ = data_mean

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)


    # Training tracking
    outputs = []        # Store data for visualization/comparison
    epoch_losses = []   # Track avg loss per epoch for plotting convergence

    # Training loop :)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            # Extract images from batch tuple (TensorDataset returns tuples)
            images = batch[0].to(device)

            # Forward pass
            reconstructed = model(images)
            loss = criterion(reconstructed, images)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss
            epoch_loss += loss.item()
            num_batches += 1

            # Store output of first batch of each epoch for visualization
            if num_batches == 1:
                outputs.append((epoch, images.cpu(), reconstructed.cpu().detach()))

        # Calculate and store avg loss per epoch
        avg_loss = epoch_loss / num_batches
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")

    total_train_time = time.time() - train_start

    print(f"\nTraining complete!")
    print(f"Final loss: {epoch_losses[-1]:.6f}")
    print(f"Stored {len(outputs)} batches for visualization")
    print(f"Total training time: {total_train_time:.2f} seconds")

    return model, epoch_losses, outputs, total_train_time


def save_model(model: AutoEncoder, filepath: str) -> None:
    """
    Saves the trained model to a file.

    Args:
        model (AutoEncoder): Trained model to save
        filepath (str): Location to save the model
    """
    save_dict = {
        'state_dict': model.state_dict(),
        'k': model.k,
        'mean_': model.mean_,
        'use_bias': model.use_bias,
        'tied_weights': model.tied_weights
    }
    torch.save(save_dict, filepath)

    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> AutoEncoder:
    """
    Loads a trained model from a location.

    Args:
        filepath (str): Path to the saved model

    Returns:
        AutoEncoder: Loaded model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dict = torch.load(filepath, map_location=device)

    # Load with configuration (backwards compatible with old models)
    use_bias = save_dict.get('use_bias', True)  # Default to True for old models
    tied_weights = save_dict.get('tied_weights', False)  # Default to False for old models

    model = AutoEncoder(k=save_dict['k'], use_bias=use_bias, tied_weights=tied_weights).to(device)
    model.load_state_dict(save_dict['state_dict'])
    model.mean_ = save_dict['mean_']

    return model
