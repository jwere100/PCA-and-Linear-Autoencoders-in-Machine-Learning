"""
AutoEncoder Implementation via Pytorch
@author Joshua Were jow2112@columbia.edu
11/12/25
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from torch.utils.data import DataLoader

class AutoEncoder(nn.Module):
    """
    image by marginally lowers the in_channels for the encoder.
    The decoder method does the exact opposite, increasing the in_channels and
    out_channels to produce an enhanced image.
    forward(self,x):
    The forward method runs data through the encoder and decoder and returns it. """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(784, 256)
        self.decoder = nn.Linear(256, 784)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))



"""
Import MNIST Dataset:
Luckily, PyTorch contains a module dedicated to housing certain datasets.
In this case, PyTorch has the MNIST dataset, which simplifies the process of
modifying the data for training.
"""

train_data = datasets.MNIST(root='dataset/',train=True,transform=transforms.ToTensor(),download=True)
train_loader = DataLoader(dataset=train_data, shuffle=True)

"""
Hyperparamters: A set of modifiable values that impacts the training loop. 
The difference between a hyperparameter and other components
of a model is that hyperparameters are constants that are easily modifiable.
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoEncoder().to(device)
epochs = 30
learning_rate = .001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#Define our Loss and optimizer
#Sample Loss function, might change later

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

outputs = []
losses = []

for epoch in range(epochs):
    for images, size in train_loader:
        images = images.view(-1,28*28).to(device)
        reconst = model(images)
        loss = criterion(reconst,images)

        optimizer.zero_grad()
        #backpropagation
        loss.backward()
        #Optimizer step
        optimizer.step()

        outputs.append((epoch, images, reconst))
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
