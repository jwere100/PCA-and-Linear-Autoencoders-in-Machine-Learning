"""
AutoEncoder Implementation via Pytorch
@author Joshua Were jow2112@columbia.edu
@author Nick Meyer njm2179@columbia.edu
11/28/25
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple

class AutoEncoder(nn.Module):
    """
    Linear Autoencoder class for dimensionality reduction on MNIST.
    This single-layer autoencoder learns to compress the 784-dim images into a
    k-dim bottleneck representation (encoder) and to reconstruct the 784-dim 
    images (decoder). 
    
    Args:
        k (int): Bottleneck dimensionality (number of latent features)
    
    Attributes:
        encoder (nn.Linear): Linear layer mapping 784 -> k
        decoder (nn.Linear): Linear layer mapping k -> 784
        k (int): Bottleneck dimensionality
        mean_ (np.ndarray): Mean of training data (stored after training for centered comparison)
    """
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(784, k)
        self.decoder = nn.Linear(k, 784)
        self.mean_ = None

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode then decode.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 784)
            
        Returns:
            torch.Tensor: Reconstructed tensor of shape (batch_size, 784)
        """
        return self.decoder(self.encoder(x))
    
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
        
        The decoder weight matrix W_D has shape (784, k), where each row
        represents a learned basis vector for image reconstruction. 
        
        Returns:
            np.ndarray: Decoder weights of shape (784, k)
        """
        return self.decoder.weight.data.cpu().numpy()
    
    def get_encoder_bias(self) -> np.ndarray:
        """
        Extracts encoder bias as a numpy array.
        
        Returns:
            np.ndarray: Encoder biases of shape (k,)
        """
        return self.encoder.bias.data.cpu().numpy()
    
    def get_decoder_bias(self) -> np.ndarray:
        """
        Extracts decoder bias as a numpy array.
        
        Returns:
            np.ndarray: Decoder biases of shape (784,)
        """
        return self.decoder.bias.data.cpu().numpy()
    
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

def train_autoencoder(
        k: int = 50,                    
        batch_size: int = 64,           
        epochs: int = 30,               
        learning_rate: float = 0.001,   
) -> Tuple[AutoEncoder, list, list]:
    """
    Trains a linear autoencoder on MNIST.

    Args:
        k (int): Bottleneck dimensionality
        batch_size (int): Training batch size
        epochs (int): Number of training epochs
        learning_rate (float): Optimizer learning rate
    
    Returns:
        model (Autoencoder): Trained autoencoder model
        epoch_losses (list): average loss per epoch
        outputs (list): Stored tuples for visualization (epoch, input, reconstruction)
    """

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    print(f"Model architecture: 784 → {k} → 784")
    print(f"Hyperparameters: batch_size={batch_size}, epochs={epochs}, lr={learning_rate}")

    # Import MNIST
    train_data = datasets.MNIST(root='dataset/',train=True,transform=transforms.ToTensor(),download=True)
    train_loader = DataLoader(dataset=train_data, batch_size = batch_size, shuffle=True)

    # Compute and store training data mean (for fair comparison with PCA
    all_data = train_data.data.float().view(-1, 784) / 255.0  # Normalize to [0,1]
    data_mean = all_data.mean(dim=0).numpy()

    # Initialize model
    model = AutoEncoder(k=k).to(device)
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

        for images, labels in train_loader:
            # Flatten from (batch_size, 1, 28, 28) to (batch_size, 784)
            images = images.view(-1, 28*28).to(device)

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

    print(f"\nTraining complete!")
    print(f"Final loss: {epoch_losses[-1]:.6f}")
    print(f"Stored {len(outputs)} batches for visualization")

    return model, epoch_losses, outputs

def save_model(model: AutoEncoder, filepath: str) -> None:
    """
    Saves the trained model to a file.
    
    Args:
        model (AutoEncoder): Trained model to save
        filepath (str): Location to save the model
    """
    save_dict = {'state_dict': model.state_dict(), 'k': model.k, 'mean_': model.mean_}
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
    model = AutoEncoder(k=save_dict['k']).to(device)
    model.load_state_dict(save_dict['state_dict'])
    model.mean_ = save_dict['mean_']

    return model
