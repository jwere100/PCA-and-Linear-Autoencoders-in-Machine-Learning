"""
AutoEncoder Implementation via Pytorch
@author Joshua Were jow2112@columbia.edu
@author Nick Meyer njm2179@columbia.edu
11/26/25
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from torch.utils.data import DataLoader


class AutoEncoder(nn.Module):
    """
    Linear Autoencoder class for dimensionality reduction on MNIST.
    This single-layer autoencoder learns to compress the 784-dim images into a
    k-dim bottleneck representation (encoder) and to reconstruct the 784-dim 
    images (decoder). 
    
    Args:
        k (int): Bottleneck dimensionality (number of latent features)
    
    forward(x):
        Encodes the input through the bottleneck, then decodes back to the
        original dimensionality.
    """
    def __init__(self, k):
        super().__init__()
        self.encoder = nn.Linear(784, k)
        self.decoder = nn.Linear(k, 784)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


"""
Hyperparameters: a set of easily modifiable values for experimentation which we
will adjust to test different training configurations.
"""
K = 50                  # Bottleneck dimensions
BATCH_SIZE = 64         # Number of samples over which the loss is calculated
EPOCHS = 30             # Number of trainining epochs
LEARNING_RATE = 0.001   # Optimizer learning rate 


"""
Import MNIST Dataset:
Luckily, PyTorch contains a module dedicated to housing certain datasets.
In this case, PyTorch has the MNIST dataset, which simplifies the process of
modifying the data for training.
"""
train_data = datasets.MNIST(root='dataset/',train=True,transform=transforms.ToTensor(),download=True)
train_loader = DataLoader(dataset=train_data, batch_size = BATCH_SIZE, shuffle=True)


# Set up the device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoEncoder(k=K).to(device)
print(f"Using device: {device}")
print(f"Model architecture: 784 → {K} → 784")


# Define the loss function and optimizer
# MSE loss measures reconstruction error (same as PCA)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)


# Training loop :)
outputs = []        # Store data for visualization/comparison
epoch_losses = []   # Track avg loss per epoch for plotting convergence

for epoch in range(EPOCHS):
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
    print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.6f}")

# Save trained model
torch.save(model.state_dict(), f'autoencoder_k{K}.pth')
print(f"\nModel saved to autoencoder_k{K}.pth")

# Training completion confirmations
print(f"\nTraining complete!")
print(f"Final loss: {epoch_losses[-1]:.6f}")
print(f"Stored {len(outputs)} batches for visualization")
