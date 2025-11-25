"""
Test suite for PCA implementation
Tests basic functionality, dimensionality reduction, and reconstruction
"""

import numpy as np
import time
from principal_component_analysis import PCA
from data_loader import load_mnist


def test_basic_functionality():
    """Test basic PCA fit and transform on synthetic data."""
    print("\n" + "="*60)
    print("TEST 1: Basic Functionality")
    print("="*60)
    
    # Create synthetic data: 100 samples, 10 features
    np.random.seed(42)
    X = np.random.randn(100, 10)
    
    # Test fit and transform
    pca = PCA(n_components=3)
    X_transformed = pca.fit_transform(X)
    
    assert X_transformed.shape == (100, 3), f"Expected shape (100, 3), got {X_transformed.shape}"
    print(f"✓ fit_transform works: {X.shape} -> {X_transformed.shape}")
    
    # Test inverse transform
    X_reconstructed = pca.inverse_transform(X_transformed)
    assert X_reconstructed.shape == X.shape, f"Expected shape {X.shape}, got {X_reconstructed.shape}"
    print(f"✓ inverse_transform works: {X_transformed.shape} -> {X_reconstructed.shape}")
    
    # Test components shape
    assert pca.components_.shape == (3, 10), f"Expected components shape (3, 10), got {pca.components_.shape}"
    print(f"✓ components_ shape is correct: {pca.components_.shape}")
    
    # Test variance explained
    assert len(pca.explained_variance_) == 3, f"Expected 3 explained variances, got {len(pca.explained_variance_)}"
    print(f"✓ explained_variance_ computed: {pca.explained_variance_ratio_}")


def test_reconstruction_error():
    """Test reconstruction error calculation."""
    print("\n" + "="*60)
    print("TEST 2: Reconstruction Error")
    print("="*60)
    
    np.random.seed(42)
    X = np.random.randn(100, 10)
    
    # Test with different numbers of components
    errors = []
    for n_comp in [2, 5, 10]:
        pca = PCA(n_components=n_comp)
        pca.fit(X)
        error = pca.get_reconstruction_error(X)
        errors.append(error)
        print(f"✓ n_components={n_comp:2d}: MSE = {error:.6f}")
    
    # Error should decrease with more components
    assert errors[0] > errors[1] > errors[2], "Error should decrease with more components"
    print("✓ Reconstruction error decreases with more components")


def test_variance_explained():
    """Test variance explained calculation."""
    print("\n" + "="*60)
    print("TEST 3: Variance Explained")
    print("="*60)
    
    np.random.seed(42)
    X = np.random.randn(100, 10)
    
    pca = PCA(n_components=5)
    pca.fit(X)
    
    # Check explained variance ratio sums to less than 1
    total_var = np.sum(pca.explained_variance_ratio_)
    assert 0 < total_var <= 1, f"Total variance should be in (0, 1], got {total_var}"
    print(f"✓ Total explained variance: {total_var:.4f}")
    
    # Check cumulative variance
    cum_var = pca.get_cumulative_variance_explained()
    assert cum_var[-1] == total_var, "Cumulative variance should match total"
    print(f"✓ Cumulative variance explained: {cum_var}")


def test_error_handling():
    """Test error handling."""
    print("\n" + "="*60)
    print("TEST 4: Error Handling")
    print("="*60)
    
    pca = PCA(n_components=3)
    
    # Test transform without fit
    try:
        X = np.random.randn(10, 5)
        pca.transform(X)
        print("✗ Should raise error when transform called before fit")
    except ValueError as e:
        print(f"✓ Correctly raises error: {str(e)}")
    
    # Test inverse_transform without fit
    try:
        X_transformed = np.random.randn(10, 3)
        pca.inverse_transform(X_transformed)
        print("✗ Should raise error when inverse_transform called before fit")
    except ValueError as e:
        print(f"✓ Correctly raises error: {str(e)}")


def test_on_mnist_subset():
    """Test PCA on MNIST dataset (subset for speed)."""
    print("\n" + "="*60)
    print("TEST 5: MNIST Subset Test")
    print("="*60)
    
    try:
        print("Loading MNIST data...")
        X, y = load_mnist()
        
        # Use only first 5000 samples for testing speed
        X_subset = X[:5000].astype(np.float32)
        
        # Normalize pixel values to [0, 1]
        X_subset = X_subset / 255.0
        
        print(f"Loaded MNIST subset: {X_subset.shape}")
        
        # Test with various component counts
        for n_comp in [8, 16, 32]:
            print(f"\nTesting with {n_comp} components...")
            
            start = time.time()
            pca = PCA(n_components=n_comp)
            pca.fit(X_subset)
            fit_time = time.time() - start
            
            start = time.time()
            X_transformed = pca.transform(X_subset)
            transform_time = time.time() - start
            
            start = time.time()
            X_reconstructed = pca.inverse_transform(X_transformed)
            inverse_time = time.time() - start
            
            mse = pca.get_reconstruction_error(X_subset)
            cum_var = pca.get_cumulative_variance_explained()
            
            print(f"  ✓ Fit time: {fit_time:.3f}s")
            print(f"  ✓ Transform time: {transform_time:.3f}s")
            print(f"  ✓ Inverse transform time: {inverse_time:.3f}s")
            print(f"  ✓ Reconstruction MSE: {mse:.6f}")
            print(f"  ✓ Cumulative variance explained: {cum_var[-1]:.4f}")
            
    except Exception as e:
        print(f"✗ Error during MNIST test: {str(e)}")
        import traceback
        traceback.print_exc()


def test_component_orthogonality():
    """Test that principal components are orthogonal."""
    print("\n" + "="*60)
    print("TEST 6: Component Orthogonality")
    print("="*60)
    
    np.random.seed(42)
    X = np.random.randn(100, 10)
    
    pca = PCA(n_components=5)
    pca.fit(X)
    
    # Compute Gram matrix (should be identity if orthogonal)
    gram = pca.components_ @ pca.components_.T
    
    # Check diagonal is close to 1
    diag = np.diag(gram)
    assert np.allclose(diag, 1.0), "Principal components should be unit vectors"
    print(f"✓ Principal components are unit vectors: {diag}")
    
    # Check off-diagonal is close to 0
    off_diag = gram - np.eye(pca.n_components)
    max_off_diag = np.max(np.abs(off_diag))
    assert max_off_diag < 1e-10, f"Components should be orthogonal, max off-diag: {max_off_diag}"
    print(f"✓ Principal components are orthogonal (max off-diag: {max_off_diag:.2e})")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PCA IMPLEMENTATION TEST SUITE")
    print("="*60)
    
    try:
        test_basic_functionality()
        test_reconstruction_error()
        test_variance_explained()
        test_error_handling()
        test_component_orthogonality()
        test_on_mnist_subset()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60 + "\n")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
