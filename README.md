# PCA-and-Linear-Autoencoders-in-Machine-Learning

**Summary:** The following repository contains the construction, implementation, and analysis of Linear Autoencoders and Principal Component Analysis, also known as PCA, in Machine Learning. This a collaborative project that combines essential 
concepts in linear algebra such as eigendecomposition and linear transformations. Our goal of this project is to build classification models using an Autoencoder and PCA, run the models on the MNIST dataset, analyze outputs, and string together a comprehensive research document that ties together the operations used in this repository.

**Key Questions**: 
- How does PCA use eigendecomposition to find optimal low-dimensional representations of higher-dimensional data?
- Can a linear autoencoder trained with gradient descent learn to approximate the same dimensionality reduction that PCA does? Ie. How does their performance on the dataset compare?
- How do the two approaches compare in terms of:
  - Performance / how well they’re able to reconstruct the data
  - Computational efficiency and accuracy?


**Why These Questions Matter:** ML models require very large datasets. Finding ways to compress the data is very important. We also operate in a compute-constrained environment, so, all else equal, we want to find the methods that consume the lowest amounts of computational resources. Extracting meaningful, low-dimensional representations improves efficiency, interpretability, and downstream prediction. This project investigates two major approaches to dimensionality reduction [principal component analysis (PCA) and linear autoencoders] to understand their mathematical connection and performance trade-offs. PCA underlies techniques such as image compression and risk analysis, while autoencoders form the foundation of modern generative and self-supervised neural networks.

**Authors**:
- Roger Fortunato, rlf2157@columbia.edu
- Nick Meyer, njm2179@columbia.edu
- Ayo Adetayo, aa5886@columbia.edu
- Joshua Were, jow2112@columbia.edu
- Arjun Purohit, ap4670@columbia.edu

## Setup

To set up the development environment:

1. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```

2. **Activate the virtual environment:**
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```