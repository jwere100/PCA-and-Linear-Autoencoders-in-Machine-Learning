"""
Main Experiment: PCA vs Linear Autoencoder Comparison on MNIST

This script runs a comprehensive experiment comparing PCA and linear autoencoders
for dimensionality reduction on the MNIST dataset. It tests multiple latent
dimensions (k values) and hyperparameter configurations to investigate:

    1. How well each method reconstructs the data
    2. Whether the autoencoder learns the same subspace as PCA
    3. How hyperparameters affect convergence to the PCA subspace
    4. Differences in computational efficiency
    5. Downstream classification performance

@author: Nick Meyer njm2179@columbia.edu
@date: 12/01/25
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import pickle
import json
from typing import Dict, List, Tuple

# Import our implementations
from data_loader import load_mnist_split
from principal_component_analysis import PCA
from linear_autoencoder import train_autoencoder, save_model, AutoEncoder
from comparison import (
    compute_subspace_similarity,
    compare_reconstruction_error,
    plot_reconstruction_comparison,
    plot_components_comparison,
    plot_principal_angles,
    compare_classification_performance,
    generate_report
)

import matplotlib.pyplot as plt

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# EXPERIMENT MODE CONFIGURATION

# Set which experiments to run:
# 1 = Main k-value comparison (baseline)
# 2 = Initial hyperparameter sensitivity analysis
# 3 = Optimization search (epochs, init, lr_schedule across all k values)
# 4 = Constraint isolation (tied/untied weights, bias/no bias combinations)
# 'all' = Run all experiments sequentially
EXPERIMENT_MODE = 4  # Change this to run different experiments

# Experiment configuration
K_VALUES = [8, 32, 64, 128]  # Different latent dimensions to test

# Baseline hyperparameters for k-value comparison
BASELINE_HYPERPARAMS = {
    'batch_size': 128,
    'epochs': 50,
    'learning_rate': 0.001,
    'seed': RANDOM_SEED
}

# Hyperparameter grid for sensitivity analysis (on k=32)
HYPERPARAM_GRID = {
    'learning_rates': [0.0001, 0.001, 0.01],
    'batch_sizes': [64, 128, 256],
    'seeds': [0, 42, 123],  # Different initializations
}

# Experiment 3: Optimization search configuration (full 70k dataset)
# Tests different training configurations to find optimal convergence to PCA subspace
OPTIMIZATION_CONFIG = {
    'k_values': [8, 32, 64, 128],
    'epochs': [50, 100, 200],
    'init_strategies': ['default', 'orthogonal'],
    'lr_schedules': ['constant', 'cosine'],
    'base_lr': 0.001,
    'batch_size': 128,
    'seed': RANDOM_SEED,
    'use_bias': False,
    'tied_weights': True,
    'max_samples': None,  # Use full 70k dataset
}

# Experiment 4: Constraint isolation configuration
# Tests which constraint (tied weights vs no bias) is more critical
CONSTRAINT_CONFIG = {
    'k_values': [8, 32, 64, 128],
    'configurations': [
        {'use_bias': False, 'tied_weights': True, 'name': 'no_bias_tied'},
        {'use_bias': False, 'tied_weights': False, 'name': 'no_bias_untied'},
        {'use_bias': True, 'tied_weights': True, 'name': 'bias_tied'},
        {'use_bias': True, 'tied_weights': False, 'name': 'bias_untied'},
    ],
    # Optimal configuration from experiment 3
    'epochs': 100,
    'init_strategy': 'orthogonal',
    'lr_schedule': 'cosine',
    'base_lr': 0.001,
    'batch_size': 128,
    'seed': RANDOM_SEED,
    'max_samples': None,  # Use full 70k dataset
}


def create_results_directories():
    """Create directory structure for storing all experiment results."""
    dirs = [
        'results',
        'results/models',
        'results/visualizations',
        'results/metrics',
        'results/hyperparameter_search'
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("✓ Created results directory structure")


def save_metrics_to_csv(metrics: List[Dict], filename: str):
    """Save metrics dictionary to CSV file."""
    df = pd.DataFrame(metrics)
    filepath = os.path.join('results/metrics', filename)
    df.to_csv(filepath, index=False)
    print(f"  Saved metrics to {filepath}")


def run_pca_experiment(X_train: np.ndarray, X_test: np.ndarray, k: int) -> Tuple[PCA, Dict]:
    """
    Run PCA experiment for a given k value.

    Args:
        X_train: Training data
        X_test: Test data
        k: Number of principal components

    Returns:
        Fitted PCA model and timing statistics
    """
    print(f"\n{'='*60}")
    print(f"Running PCA with k={k}")
    print(f"{'='*60}")

    # Fit PCA
    pca = PCA(n_components=k)
    fit_start = time.time()
    pca.fit(X_train)
    fit_time = time.time() - fit_start

    # Transform test data
    transform_start = time.time()
    _ = pca.transform(X_test)
    transform_time = time.time() - transform_start

    # Compute reconstruction error
    recon_error = pca.get_reconstruction_error(X_test)

    print(f"  Fit time: {fit_time:.3f}s")
    print(f"  Transform time: {transform_time:.3f}s")
    print(f"  Reconstruction MSE: {recon_error:.6f}")
    print(f"  Explained variance: {pca.get_cumulative_variance_explained()[-1]:.4f}")

    # Save model
    model_path = f'results/models/pca_k{k}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(pca, f)
    print(f"  Saved model to {model_path}")

    return pca, {
        'pca_fit_time': fit_time,
        'pca_transform_time': transform_time,
        'pca_reconstruction_error': recon_error,
        'pca_explained_variance': pca.get_cumulative_variance_explained()[-1]
    }


def run_autoencoder_experiment(
    X_train: np.ndarray,
    X_test: np.ndarray,
    k: int,
    hyperparams: Dict,
    save_prefix: str = "",
    use_bias: bool = False,
    tied_weights: bool = True,
    init_strategy: str = 'default',
    lr_schedule: str = 'constant'
) -> Tuple[AutoEncoder, List[float], Dict]:
    """
    Run autoencoder experiment with given hyperparameters.

    Args:
        X_train: Training data
        X_test: Test data
        k: Bottleneck dimensionality
        hyperparams: Dictionary of hyperparameters
        save_prefix: Prefix for saved files (for hyperparameter experiments)
        use_bias: Whether to use bias terms (default: False for Baldi & Hornik)
        tied_weights: Whether decoder = encoder.T (default: True for Baldi & Hornik)
        init_strategy: Weight initialization strategy ('default' or 'orthogonal')
        lr_schedule: Learning rate schedule ('constant' or 'cosine')

    Returns:
        Trained model, training losses, and timing statistics
    """
    print(f"\n{'='*60}")
    print(f"Training Autoencoder with k={k}")
    print(f"Hyperparameters: {hyperparams}")
    print(f"Architecture: use_bias={use_bias}, tied_weights={tied_weights}")
    print(f"Training: init={init_strategy}, lr_schedule={lr_schedule}")
    print(f"{'='*60}")

    # Train autoencoder with configurable architecture and training settings
    model, epoch_losses, outputs, train_time, peak_memory = train_autoencoder(
        X_train=X_train,
        k=k,
        batch_size=hyperparams['batch_size'],
        epochs=hyperparams['epochs'],
        learning_rate=hyperparams['learning_rate'],
        seed=hyperparams['seed'],
        use_bias=use_bias,
        tied_weights=tied_weights,
        init_strategy=init_strategy,
        lr_schedule=lr_schedule
    )

    # Compute reconstruction error on test set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    recon_error = model.get_reconstruction_error(X_test, device)

    # Measure encoding time
    encode_start = time.time()
    _ = model.encode(X_test, device)
    encode_time = time.time() - encode_start

    print(f"  Test reconstruction MSE: {recon_error:.6f}")
    print(f"  Encoding time: {encode_time:.3f}s")

    # Save model
    model_path = f'results/models/{save_prefix}autoencoder_k{k}.pth'
    save_model(model, model_path)

    return model, epoch_losses, {
        'ae_train_time': train_time,
        'ae_encode_time': encode_time,
        'ae_reconstruction_error': recon_error,
        'ae_final_loss': epoch_losses[-1]
    }


def compare_and_visualize(
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    pca: PCA,
    autoencoder: AutoEncoder,
    k: int,
    epoch_losses: List[float],
    pca_stats: Dict,
    ae_stats: Dict,
    X_train: np.ndarray,
    save_prefix: str = ""
):
    """
    Compare PCA and autoencoder and generate visualizations.

    Args:
        X_test: Test data
        y_train: Training labels
        y_test: Test labels
        pca: Fitted PCA model
        autoencoder: Trained autoencoder
        k: Latent dimension
        epoch_losses: Training losses
        pca_stats: PCA timing statistics
        ae_stats: Autoencoder timing statistics
        X_train: Training data (for classification comparison)
        save_prefix: Prefix for saved files
    """
    print(f"\n{'='*60}")
    print(f"Comparison and Visualization (k={k})")
    print(f"{'='*60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Reconstruction error comparison
    print("\n1. Comparing reconstruction errors...")
    recon_comparison = compare_reconstruction_error(X_test, pca, autoencoder)
    print(f"   PCA MSE:  {recon_comparison['pca_mse']:.6f}")
    print(f"   AE MSE:   {recon_comparison['lae_mse']:.6f}")
    print(f"   Ratio:    {recon_comparison['ratio']:.4f}")

    # 2. Subspace similarity
    print("\n2. Computing subspace similarity...")
    similarity = compute_subspace_similarity(pca, autoencoder)
    print(f"   Mean principal angle:  {similarity['mean_angle_degrees']:.2f}°")
    print(f"   Grassmann distance:    {similarity['grassmann_distance']:.4f}")

    # 3. Classification performance
    print("\n3. Comparing classification performance...")
    classification_results = compare_classification_performance(
        X_train, y_train, X_test, y_test, pca, autoencoder, device
    )
    print(f"   Raw pixels:     {classification_results['raw_accuracy']:.4f}")
    print(f"   PCA features:   {classification_results['pca_accuracy']:.4f}")
    print(f"   AE features:    {classification_results['ae_accuracy']:.4f}")

    # 4. Generate visualizations
    print("\n4. Generating visualizations...")

    # Reconstruction comparison
    viz_path = f'results/visualizations/{save_prefix}reconstruction_k{k}.png'
    plot_reconstruction_comparison(X_test, pca, autoencoder, num_images=10, path=viz_path)
    print(f"   Saved reconstruction comparison to {viz_path}")

    # Components comparison
    viz_path = f'results/visualizations/{save_prefix}components_k{k}.png'
    plot_components_comparison(pca, autoencoder, num_representations=min(10, k), path=viz_path)
    print(f"   Saved components comparison to {viz_path}")

    # Principal angles
    viz_path = f'results/visualizations/{save_prefix}principal_angles_k{k}.png'
    plot_principal_angles(similarity['principal_angles_deg'], path=viz_path)
    print(f"   Saved principal angles to {viz_path}")

    # Training loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss (MSE)')
    plt.title(f'Autoencoder Training Loss (k={k})')
    plt.grid(True, alpha=0.3)
    viz_path = f'results/visualizations/{save_prefix}training_loss_k{k}.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved training loss curve to {viz_path}")

    # 5. Compile metrics
    metrics = {
        'k': k,
        **recon_comparison,
        **similarity,
        **classification_results,
        **pca_stats,
        **ae_stats
    }

    return metrics


def run_main_experiments(X_train: np.ndarray, X_test: np.ndarray,
                        y_train: np.ndarray, y_test: np.ndarray):
    """
    Run main experiments comparing PCA and autoencoder across different k values.
    """
    print("\n" + "="*60)
    print("MAIN EXPERIMENTS: Comparing PCA and Autoencoder")
    print("Testing k values:", K_VALUES)
    print("="*60)

    all_metrics = []

    for k in K_VALUES:
        print(f"\n\n{'#'*60}")
        print(f"# EXPERIMENT: k = {k}")
        print(f"{'#'*60}")

        # Run PCA
        pca, pca_stats = run_pca_experiment(X_train, X_test, k)

        # Run Autoencoder with baseline hyperparameters
        autoencoder, epoch_losses, ae_stats = run_autoencoder_experiment(
            X_train, X_test, k, BASELINE_HYPERPARAMS
        )

        # Compare and visualize
        metrics = compare_and_visualize(
            X_test, y_train, y_test, pca, autoencoder, k,
            epoch_losses, pca_stats, ae_stats, X_train
        )
        all_metrics.append(metrics)

    # Save all metrics
    save_metrics_to_csv(all_metrics, 'main_experiments.csv')

    # Create summary visualization comparing all k values
    create_summary_plots(all_metrics)

    return all_metrics


def create_summary_plots(metrics: List[Dict]):
    """Create summary plots comparing results across different k values."""
    print("\n" + "="*60)
    print("Creating summary comparison plots...")
    print("="*60)

    k_values = [m['k'] for m in metrics]

    # Create a 2x2 grid of comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Reconstruction errors
    ax = axes[0, 0]
    pca_errors = [m['pca_mse'] for m in metrics]
    ae_errors = [m['lae_mse'] for m in metrics]
    x = np.arange(len(k_values))
    width = 0.35
    ax.bar(x - width/2, pca_errors, width, label='PCA', alpha=0.8)
    ax.bar(x + width/2, ae_errors, width, label='Autoencoder', alpha=0.8)
    ax.set_xlabel('Latent Dimension (k)')
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title('Reconstruction Error Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(k_values)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Subspace similarity (mean principal angle)
    ax = axes[0, 1]
    angles = [m['mean_angle_degrees'] for m in metrics]
    ax.plot(k_values, angles, marker='o', markersize=8, linewidth=2)
    ax.set_xlabel('Latent Dimension (k)')
    ax.set_ylabel('Mean Principal Angle (degrees)')
    ax.set_title('Subspace Similarity (Lower = More Similar)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Perfect alignment')
    ax.legend()

    # Plot 3: Classification accuracy
    ax = axes[1, 0]
    pca_acc = [m['pca_accuracy'] for m in metrics]
    ae_acc = [m['ae_accuracy'] for m in metrics]
    raw_acc = [m['raw_accuracy'] for m in metrics]
    ax.plot(k_values, pca_acc, marker='o', label='PCA', linewidth=2)
    ax.plot(k_values, ae_acc, marker='s', label='Autoencoder', linewidth=2)
    ax.axhline(y=raw_acc[0], color='gray', linestyle='--', label='Raw pixels', linewidth=2)
    ax.set_xlabel('Latent Dimension (k)')
    ax.set_ylabel('Classification Accuracy')
    ax.set_title('Downstream Classification Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Computational efficiency
    ax = axes[1, 1]
    pca_times = [m['pca_fit_time'] + m['pca_transform_time'] for m in metrics]
    ae_times = [m['ae_train_time'] for m in metrics]
    x = np.arange(len(k_values))
    width = 0.35
    ax.bar(x - width/2, pca_times, width, label='PCA (fit+transform)', alpha=0.8)
    ax.bar(x + width/2, ae_times, width, label='Autoencoder (train)', alpha=0.8)
    ax.set_xlabel('Latent Dimension (k)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Computational Efficiency')
    ax.set_xticks(x)
    ax.set_xticklabels(k_values)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    plt.tight_layout()
    summary_path = 'results/visualizations/summary_comparison.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved summary comparison to {summary_path}")


def run_hyperparameter_experiments(X_train: np.ndarray, X_test: np.ndarray,
                                   y_train: np.ndarray, y_test: np.ndarray):
    """
    Run hyperparameter sensitivity analysis on k=32 to see how different
    configurations affect the autoencoder's ability to learn the PCA subspace.
    """
    print("\n\n" + "="*60)
    print("HYPERPARAMETER SENSITIVITY ANALYSIS (k=32)")
    print("="*60)

    k = 32  # Use k=32 for hyperparameter experiments

    # First, fit PCA once for all comparisons
    print("\nFitting PCA for reference...")
    pca, _ = run_pca_experiment(X_train, X_test, k)

    all_hyperparam_results = []
    experiment_id = 0

    # Test different learning rates
    print("\n" + "-"*60)
    print("Testing different learning rates...")
    print("-"*60)
    for lr in HYPERPARAM_GRID['learning_rates']:
        experiment_id += 1
        hyperparams = BASELINE_HYPERPARAMS.copy()
        hyperparams['learning_rate'] = lr

        save_prefix = f"hp_lr{lr}_"
        autoencoder, epoch_losses, ae_stats = run_autoencoder_experiment(
            X_train, X_test, k, hyperparams, save_prefix=save_prefix
        )

        # Compute subspace similarity
        similarity = compute_subspace_similarity(pca, autoencoder)

        result = {
            'experiment_id': experiment_id,
            'k': k,
            'learning_rate': lr,
            'batch_size': hyperparams['batch_size'],
            'seed': hyperparams['seed'],
            'mean_angle_degrees': similarity['mean_angle_degrees'],
            'grassmann_distance': similarity['grassmann_distance'],
            'ae_reconstruction_error': ae_stats['ae_reconstruction_error'],
            'ae_final_loss': ae_stats['ae_final_loss'],
            'ae_train_time': ae_stats['ae_train_time']
        }
        all_hyperparam_results.append(result)

        print(f"  LR={lr}: Mean angle={similarity['mean_angle_degrees']:.2f}°, "
              f"Grassmann dist={similarity['grassmann_distance']:.4f}")

    # Test different batch sizes
    print("\n" + "-"*60)
    print("Testing different batch sizes...")
    print("-"*60)
    for bs in HYPERPARAM_GRID['batch_sizes']:
        experiment_id += 1
        hyperparams = BASELINE_HYPERPARAMS.copy()
        hyperparams['batch_size'] = bs

        save_prefix = f"hp_bs{bs}_"
        autoencoder, epoch_losses, ae_stats = run_autoencoder_experiment(
            X_train, X_test, k, hyperparams, save_prefix=save_prefix
        )

        similarity = compute_subspace_similarity(pca, autoencoder)

        result = {
            'experiment_id': experiment_id,
            'k': k,
            'learning_rate': hyperparams['learning_rate'],
            'batch_size': bs,
            'seed': hyperparams['seed'],
            'mean_angle_degrees': similarity['mean_angle_degrees'],
            'grassmann_distance': similarity['grassmann_distance'],
            'ae_reconstruction_error': ae_stats['ae_reconstruction_error'],
            'ae_final_loss': ae_stats['ae_final_loss'],
            'ae_train_time': ae_stats['ae_train_time']
        }
        all_hyperparam_results.append(result)

        print(f"  BS={bs}: Mean angle={similarity['mean_angle_degrees']:.2f}°, "
              f"Grassmann dist={similarity['grassmann_distance']:.4f}")

    # Test different seeds (initialization)
    print("\n" + "-"*60)
    print("Testing different random seeds (initialization)...")
    print("-"*60)
    for seed in HYPERPARAM_GRID['seeds']:
        experiment_id += 1
        hyperparams = BASELINE_HYPERPARAMS.copy()
        hyperparams['seed'] = seed

        save_prefix = f"hp_seed{seed}_"
        autoencoder, epoch_losses, ae_stats = run_autoencoder_experiment(
            X_train, X_test, k, hyperparams, save_prefix=save_prefix
        )

        similarity = compute_subspace_similarity(pca, autoencoder)

        result = {
            'experiment_id': experiment_id,
            'k': k,
            'learning_rate': hyperparams['learning_rate'],
            'batch_size': hyperparams['batch_size'],
            'seed': seed,
            'mean_angle_degrees': similarity['mean_angle_degrees'],
            'grassmann_distance': similarity['grassmann_distance'],
            'ae_reconstruction_error': ae_stats['ae_reconstruction_error'],
            'ae_final_loss': ae_stats['ae_final_loss'],
            'ae_train_time': ae_stats['ae_train_time']
        }
        all_hyperparam_results.append(result)

        print(f"  Seed={seed}: Mean angle={similarity['mean_angle_degrees']:.2f}°, "
              f"Grassmann dist={similarity['grassmann_distance']:.4f}")

    # Save hyperparameter results
    save_metrics_to_csv(all_hyperparam_results, 'hyperparameter_sensitivity.csv')

    # Create hyperparameter sensitivity plots
    create_hyperparameter_plots(all_hyperparam_results)

    return all_hyperparam_results


def create_hyperparameter_plots(results: List[Dict]):
    """Create plots showing hyperparameter sensitivity."""
    print("\nCreating hyperparameter sensitivity plots...")

    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Learning rate effect
    ax = axes[0]
    lr_data = df[df['batch_size'] == BASELINE_HYPERPARAMS['batch_size']]
    lr_data = lr_data[lr_data['seed'] == BASELINE_HYPERPARAMS['seed']]
    lr_data = lr_data.sort_values('learning_rate')
    ax.plot(lr_data['learning_rate'], lr_data['mean_angle_degrees'],
            marker='o', markersize=10, linewidth=2)
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Mean Principal Angle (degrees)')
    ax.set_title('Effect of Learning Rate on Subspace Alignment')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # Plot 2: Batch size effect
    ax = axes[1]
    bs_data = df[df['learning_rate'] == BASELINE_HYPERPARAMS['learning_rate']]
    bs_data = bs_data[bs_data['seed'] == BASELINE_HYPERPARAMS['seed']]
    bs_data = bs_data.sort_values('batch_size')
    ax.plot(bs_data['batch_size'], bs_data['mean_angle_degrees'],
            marker='o', markersize=10, linewidth=2)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Mean Principal Angle (degrees)')
    ax.set_title('Effect of Batch Size on Subspace Alignment')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # Plot 3: Seed (initialization) effect
    ax = axes[2]
    seed_data = df[df['learning_rate'] == BASELINE_HYPERPARAMS['learning_rate']]
    seed_data = seed_data[seed_data['batch_size'] == BASELINE_HYPERPARAMS['batch_size']]
    seed_data = seed_data.sort_values('seed')
    ax.bar(range(len(seed_data)), seed_data['mean_angle_degrees'], alpha=0.8)
    ax.set_xlabel('Random Seed')
    ax.set_ylabel('Mean Principal Angle (degrees)')
    ax.set_title('Effect of Initialization on Subspace Alignment')
    ax.set_xticks(range(len(seed_data)))
    ax.set_xticklabels(seed_data['seed'])
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    viz_path = 'results/visualizations/hyperparameter_sensitivity.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved hyperparameter sensitivity plot to {viz_path}")


def run_optimization_experiments(X_train: np.ndarray, X_test: np.ndarray,
                                 y_train: np.ndarray, y_test: np.ndarray):
    """
    Experiment 3: Optimization search to find best configuration for PCA convergence.
    Tests different combinations of epochs, initialization, and learning rate schedules
    on full 70k dataset across various values of k.

    Total runs: 4 k values × 3 epochs × 2 init × 2 lr_schedule = 48 runs
    """
    print("\n\n" + "="*80)
    print("EXPERIMENT 3: OPTIMIZATION SEARCH FOR PCA CONVERGENCE")
    print("="*80)
    print(f"Configuration: {OPTIMIZATION_CONFIG}")
    print(f"Total runs: {len(OPTIMIZATION_CONFIG['k_values']) * len(OPTIMIZATION_CONFIG['epochs']) * len(OPTIMIZATION_CONFIG['init_strategies']) * len(OPTIMIZATION_CONFIG['lr_schedules'])}")

    # Fit PCA models for all k values once
    print("\n" + "="*60)
    print("Fitting PCA models for all k values...")
    print("="*60)
    pca_models = {}
    for k in OPTIMIZATION_CONFIG['k_values']:
        pca, _ = run_pca_experiment(X_train, X_test, k)
        pca_models[k] = pca

    all_results = []
    run_id = 0

    # Grid search over all configurations
    for k in OPTIMIZATION_CONFIG['k_values']:
        for epochs in OPTIMIZATION_CONFIG['epochs']:
            for init_strategy in OPTIMIZATION_CONFIG['init_strategies']:
                for lr_schedule in OPTIMIZATION_CONFIG['lr_schedules']:
                    run_id += 1
                    print(f"\n{'#'*70}")
                    print(f"# RUN {run_id}/48: k={k}, epochs={epochs}, init={init_strategy}, lr_schedule={lr_schedule}")
                    print(f"{'#'*70}")

                    # Create hyperparams dict
                    hyperparams = {
                        'batch_size': OPTIMIZATION_CONFIG['batch_size'],
                        'epochs': epochs,
                        'learning_rate': OPTIMIZATION_CONFIG['base_lr'],
                        'seed': OPTIMIZATION_CONFIG['seed']
                    }

                    # Train autoencoder
                    save_prefix = f"opt_k{k}_ep{epochs}_{init_strategy}_{lr_schedule}_"
                    autoencoder, epoch_losses, ae_stats = run_autoencoder_experiment(
                        X_train, X_test, k, hyperparams,
                        save_prefix=save_prefix,
                        use_bias=OPTIMIZATION_CONFIG['use_bias'],
                        tied_weights=OPTIMIZATION_CONFIG['tied_weights'],
                        init_strategy=init_strategy,
                        lr_schedule=lr_schedule
                    )

                    # Compute subspace similarity with PCA
                    pca = pca_models[k]
                    similarity = compute_subspace_similarity(pca, autoencoder)

                    # Compute reconstruction error comparison
                    recon_comparison = compare_reconstruction_error(X_test, pca, autoencoder)

                    # Compile results
                    result = {
                        'run_id': run_id,
                        'k': k,
                        'epochs': epochs,
                        'init_strategy': init_strategy,
                        'lr_schedule': lr_schedule,
                        'batch_size': hyperparams['batch_size'],
                        'learning_rate': hyperparams['learning_rate'],
                        'mean_angle_degrees': similarity['mean_angle_degrees'],
                        'grassmann_distance': similarity['grassmann_distance'],
                        'max_angle_degrees': similarity['principal_angles_deg'].max(),
                        'min_angle_degrees': similarity['principal_angles_deg'].min(),
                        'pca_mse': recon_comparison['pca_mse'],
                        'ae_mse': recon_comparison['lae_mse'],
                        'mse_ratio': recon_comparison['ratio'],
                        'ae_final_loss': ae_stats['ae_final_loss'],
                        'ae_train_time': ae_stats['ae_train_time'],
                        'ae_reconstruction_error': ae_stats['ae_reconstruction_error']
                    }
                    all_results.append(result)

                    print(f"  Mean angle: {similarity['mean_angle_degrees']:.4f}°")
                    print(f"  Grassmann distance: {similarity['grassmann_distance']:.6f}")
                    print(f"  AE MSE: {recon_comparison['lae_mse']:.6f}")
                    print(f"  Training time: {ae_stats['ae_train_time']:.2f}s")

    # Save results
    save_metrics_to_csv(all_results, 'optimization_experiments.csv')

    # Create analysis plots
    create_optimization_plots(all_results)

    # Print best configurations
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS BY K VALUE:")
    print("="*80)
    df = pd.DataFrame(all_results)
    for k in OPTIMIZATION_CONFIG['k_values']:
        k_results = df[df['k'] == k]
        best = k_results.loc[k_results['mean_angle_degrees'].idxmin()]
        print(f"\nk={k}:")
        print(f"  Best config: epochs={best['epochs']}, init={best['init_strategy']}, lr_schedule={best['lr_schedule']}")
        print(f"  Mean angle: {best['mean_angle_degrees']:.4f}°")
        print(f"  Grassmann distance: {best['grassmann_distance']:.6f}")
        print(f"  Training time: {best['ae_train_time']:.2f}s")

    return all_results


def create_optimization_plots(results: List[Dict]):
    """Create visualization plots for optimization experiment results."""
    print("\nCreating optimization experiment plots...")

    df = pd.DataFrame(results)

    # Create a comprehensive figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Mean angle by configuration for each k
    ax = axes[0, 0]
    for k in OPTIMIZATION_CONFIG['k_values']:
        k_data = df[df['k'] == k]
        # Create x-axis labels combining all config parameters
        k_data['config'] = k_data['epochs'].astype(str) + '_' + k_data['init_strategy'] + '_' + k_data['lr_schedule']
        ax.plot(range(len(k_data)), k_data['mean_angle_degrees'], marker='o', label=f'k={k}', linewidth=2)
    ax.set_xlabel('Configuration Index')
    ax.set_ylabel('Mean Principal Angle (degrees)')
    ax.set_title('Mean Principal Angle by Configuration')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Effect of epochs
    ax = axes[0, 1]
    for k in OPTIMIZATION_CONFIG['k_values']:
        k_data = df[df['k'] == k]
        # Average over init and lr_schedule
        grouped = k_data.groupby('epochs')['mean_angle_degrees'].mean()
        ax.plot(grouped.index, grouped.values, marker='o', label=f'k={k}', linewidth=2)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Mean Principal Angle (degrees)')
    ax.set_title('Effect of Training Epochs (averaged over init & lr_schedule)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Effect of initialization strategy
    ax = axes[1, 0]
    init_comparison = df.groupby(['k', 'init_strategy'])['mean_angle_degrees'].mean().unstack()
    x = np.arange(len(OPTIMIZATION_CONFIG['k_values']))
    width = 0.35
    ax.bar(x - width/2, init_comparison['default'], width, label='Default init', alpha=0.8)
    ax.bar(x + width/2, init_comparison['orthogonal'], width, label='Orthogonal init', alpha=0.8)
    ax.set_xlabel('k value')
    ax.set_ylabel('Mean Principal Angle (degrees)')
    ax.set_title('Effect of Initialization Strategy (averaged over epochs & lr_schedule)')
    ax.set_xticks(x)
    ax.set_xticklabels(OPTIMIZATION_CONFIG['k_values'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Effect of learning rate schedule
    ax = axes[1, 1]
    lr_comparison = df.groupby(['k', 'lr_schedule'])['mean_angle_degrees'].mean().unstack()
    x = np.arange(len(OPTIMIZATION_CONFIG['k_values']))
    width = 0.35
    ax.bar(x - width/2, lr_comparison['constant'], width, label='Constant LR', alpha=0.8)
    ax.bar(x + width/2, lr_comparison['cosine'], width, label='Cosine annealing', alpha=0.8)
    ax.set_xlabel('k value')
    ax.set_ylabel('Mean Principal Angle (degrees)')
    ax.set_title('Effect of Learning Rate Schedule (averaged over epochs & init)')
    ax.set_xticks(x)
    ax.set_xticklabels(OPTIMIZATION_CONFIG['k_values'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    viz_path = 'results/visualizations/optimization_experiments.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved optimization experiments plot to {viz_path}")


def run_constraint_experiments(X_train: np.ndarray, X_test: np.ndarray,
                               y_train: np.ndarray, y_test: np.ndarray):
    """
    Experiment 4: Constraint isolation to determine which constraint is more critical.
    Tests all 4 combinations of (tied/untied) × (bias/no_bias) across all k values.

    Total runs: 4 k values × 4 configurations = 16 runs
    """
    print("\n\n" + "="*80)
    print("EXPERIMENT 4: CONSTRAINT ISOLATION")
    print("="*80)
    print(f"Configuration: {CONSTRAINT_CONFIG}")
    print(f"Total runs: {len(CONSTRAINT_CONFIG['k_values']) * len(CONSTRAINT_CONFIG['configurations'])}")

    # Fit PCA models for all k values once
    print("\n" + "="*60)
    print("Fitting PCA models for all k values...")
    print("="*60)
    pca_models = {}
    for k in CONSTRAINT_CONFIG['k_values']:
        pca, _ = run_pca_experiment(X_train, X_test, k)
        pca_models[k] = pca

    all_results = []
    run_id = 0

    # Test all configurations
    for k in CONSTRAINT_CONFIG['k_values']:
        for config in CONSTRAINT_CONFIG['configurations']:
            run_id += 1
            print(f"\n{'#'*70}")
            print(f"# RUN {run_id}/16: k={k}, config={config['name']}")
            print(f"# use_bias={config['use_bias']}, tied_weights={config['tied_weights']}")
            print(f"{'#'*70}")

            # Create hyperparams dict
            hyperparams = {
                'batch_size': CONSTRAINT_CONFIG['batch_size'],
                'epochs': CONSTRAINT_CONFIG['epochs'],
                'learning_rate': CONSTRAINT_CONFIG['base_lr'],
                'seed': CONSTRAINT_CONFIG['seed']
            }

            # Train autoencoder
            save_prefix = f"constraint_k{k}_{config['name']}_"
            autoencoder, epoch_losses, ae_stats = run_autoencoder_experiment(
                X_train, X_test, k, hyperparams,
                save_prefix=save_prefix,
                use_bias=config['use_bias'],
                tied_weights=config['tied_weights'],
                init_strategy=CONSTRAINT_CONFIG['init_strategy'],
                lr_schedule=CONSTRAINT_CONFIG['lr_schedule']
            )

            # Compute subspace similarity with PCA
            pca = pca_models[k]
            similarity = compute_subspace_similarity(pca, autoencoder)

            # Compute reconstruction error comparison
            recon_comparison = compare_reconstruction_error(X_test, pca, autoencoder)

            # Classification performance
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            classification_results = compare_classification_performance(
                X_train, y_train, X_test, y_test, pca, autoencoder, device
            )

            # Compile results
            result = {
                'run_id': run_id,
                'k': k,
                'configuration': config['name'],
                'use_bias': config['use_bias'],
                'tied_weights': config['tied_weights'],
                'epochs': hyperparams['epochs'],
                'init_strategy': CONSTRAINT_CONFIG['init_strategy'],
                'lr_schedule': CONSTRAINT_CONFIG['lr_schedule'],
                'mean_angle_degrees': similarity['mean_angle_degrees'],
                'grassmann_distance': similarity['grassmann_distance'],
                'max_angle_degrees': similarity['principal_angles_deg'].max(),
                'min_angle_degrees': similarity['principal_angles_deg'].min(),
                'pca_mse': recon_comparison['pca_mse'],
                'ae_mse': recon_comparison['lae_mse'],
                'mse_ratio': recon_comparison['ratio'],
                'pca_accuracy': classification_results['pca_accuracy'],
                'ae_accuracy': classification_results['ae_accuracy'],
                'ae_final_loss': ae_stats['ae_final_loss'],
                'ae_train_time': ae_stats['ae_train_time']
            }
            all_results.append(result)

            print(f"  Mean angle: {similarity['mean_angle_degrees']:.4f}°")
            print(f"  Grassmann distance: {similarity['grassmann_distance']:.6f}")
            print(f"  AE accuracy: {classification_results['ae_accuracy']:.4f}")

    # Save results
    save_metrics_to_csv(all_results, 'constraint_experiments.csv')

    # Create analysis plots
    create_constraint_plots(all_results)

    # Print summary analysis
    print("\n" + "="*80)
    print("CONSTRAINT ISOLATION SUMMARY:")
    print("="*80)
    df = pd.DataFrame(all_results)

    # Average across all k values
    print("\nAverage across all k values:")
    for config in CONSTRAINT_CONFIG['configurations']:
        config_results = df[df['configuration'] == config['name']]
        avg_angle = config_results['mean_angle_degrees'].mean()
        avg_grassmann = config_results['grassmann_distance'].mean()
        print(f"\n{config['name']}:")
        print(f"  use_bias={config['use_bias']}, tied_weights={config['tied_weights']}")
        print(f"  Avg mean angle: {avg_angle:.4f}°")
        print(f"  Avg Grassmann distance: {avg_grassmann:.6f}")

    # Best configuration
    best = df.loc[df['mean_angle_degrees'].idxmin()]
    print(f"\nBest overall configuration:")
    print(f"  {best['configuration']} (k={best['k']})")
    print(f"  Mean angle: {best['mean_angle_degrees']:.4f}°")
    print(f"  Grassmann distance: {best['grassmann_distance']:.6f}")

    return all_results


def create_constraint_plots(results: List[Dict]):
    """Create visualization plots for constraint experiment results."""
    print("\nCreating constraint experiment plots...")

    df = pd.DataFrame(results)

    # Create a comprehensive figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Mean angle by configuration for each k
    ax = axes[0, 0]
    configs = df['configuration'].unique()
    x = np.arange(len(CONSTRAINT_CONFIG['k_values']))
    width = 0.2
    for i, config in enumerate(configs):
        config_data = df[df['configuration'] == config]
        config_data = config_data.sort_values('k')
        offset = (i - len(configs)/2 + 0.5) * width
        ax.bar(x + offset, config_data['mean_angle_degrees'], width, label=config, alpha=0.8)
    ax.set_xlabel('k value')
    ax.set_ylabel('Mean Principal Angle (degrees)')
    ax.set_title('Mean Principal Angle by Configuration')
    ax.set_xticks(x)
    ax.set_xticklabels(CONSTRAINT_CONFIG['k_values'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Grassmann distance by configuration
    ax = axes[0, 1]
    for i, config in enumerate(configs):
        config_data = df[df['configuration'] == config]
        config_data = config_data.sort_values('k')
        offset = (i - len(configs)/2 + 0.5) * width
        ax.bar(x + offset, config_data['grassmann_distance'], width, label=config, alpha=0.8)
    ax.set_xlabel('k value')
    ax.set_ylabel('Grassmann Distance')
    ax.set_title('Grassmann Distance by Configuration')
    ax.set_xticks(x)
    ax.set_xticklabels(CONSTRAINT_CONFIG['k_values'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Effect of tied weights (averaged over bias setting)
    ax = axes[1, 0]
    tied_comparison = df.groupby(['k', 'tied_weights'])['mean_angle_degrees'].mean().unstack()
    x = np.arange(len(CONSTRAINT_CONFIG['k_values']))
    width = 0.35
    ax.bar(x - width/2, tied_comparison[False], width, label='Untied weights', alpha=0.8)
    ax.bar(x + width/2, tied_comparison[True], width, label='Tied weights', alpha=0.8)
    ax.set_xlabel('k value')
    ax.set_ylabel('Mean Principal Angle (degrees)')
    ax.set_title('Effect of Tied Weights (averaged over bias setting)')
    ax.set_xticks(x)
    ax.set_xticklabels(CONSTRAINT_CONFIG['k_values'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Effect of bias (averaged over tied weights setting)
    ax = axes[1, 1]
    bias_comparison = df.groupby(['k', 'use_bias'])['mean_angle_degrees'].mean().unstack()
    x = np.arange(len(CONSTRAINT_CONFIG['k_values']))
    width = 0.35
    ax.bar(x - width/2, bias_comparison[False], width, label='No bias', alpha=0.8)
    ax.bar(x + width/2, bias_comparison[True], width, label='With bias', alpha=0.8)
    ax.set_xlabel('k value')
    ax.set_ylabel('Mean Principal Angle (degrees)')
    ax.set_title('Effect of Bias (averaged over tied weights setting)')
    ax.set_xticks(x)
    ax.set_xticklabels(CONSTRAINT_CONFIG['k_values'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    viz_path = 'results/visualizations/constraint_experiments.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved constraint experiments plot to {viz_path}")


def main():
    """Main experiment execution."""
    print("\n" + "="*80)
    print(" "*20 + "PCA vs LINEAR AUTOENCODER EXPERIMENT")
    print("="*80)
    print(f"\nExperiment Mode: {EXPERIMENT_MODE}")
    print(f"Random seed: {RANDOM_SEED}")

    # Create results directories
    create_results_directories()

    # Determine which dataset to load based on experiment mode
    if EXPERIMENT_MODE == 3:
        max_samples = OPTIMIZATION_CONFIG['max_samples']
        dataset_desc = "full 70k dataset" if max_samples is None else f"{max_samples} samples"
    elif EXPERIMENT_MODE == 4:
        max_samples = CONSTRAINT_CONFIG['max_samples']
        dataset_desc = "full 70k dataset" if max_samples is None else f"{max_samples} samples"
    elif EXPERIMENT_MODE == 'all':
        max_samples = None
        dataset_desc = "full 70k dataset" if max_samples is None else f"{max_samples} samples"
    else:
        max_samples = 10000
        dataset_desc = "10,000 samples"

    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    print(f"Loading MNIST dataset ({dataset_desc})...")
    load_start = time.time()
    X_train, X_test, y_train, y_test = load_mnist_split(
        test_size=0.2,
        random_state=RANDOM_SEED,
        normalize=True,
        max_samples=max_samples
    )
    load_time = time.time() - load_start

    print(f"✓ Data loaded in {load_time:.2f}s")
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    print(f"  Data range: [{X_train.min():.3f}, {X_train.max():.3f}]")

    # Run experiments based on mode
    experiment_times = {}

    if EXPERIMENT_MODE == 1 or EXPERIMENT_MODE == 'all':
        print(f"\n{'='*80}")
        print("Running Experiment 1: Main k-value comparison")
        print(f"{'='*80}")
        start_time = time.time()
        main_metrics = run_main_experiments(X_train, X_test, y_train, y_test)
        experiment_times['main_experiments'] = time.time() - start_time

        # Print summary for Experiment 1
        best_k = min(main_metrics, key=lambda x: x['mean_angle_degrees'])
        print(f"\nBest k value: k={best_k['k']}")
        print(f"  Mean principal angle: {best_k['mean_angle_degrees']:.2f}°")
        print(f"  Grassmann distance: {best_k['grassmann_distance']:.4f}")

    if EXPERIMENT_MODE == 2 or EXPERIMENT_MODE == 'all':
        print(f"\n{'='*80}")
        print("Running Experiment 2: Hyperparameter sensitivity analysis")
        print(f"{'='*80}")
        start_time = time.time()
        hp_metrics = run_hyperparameter_experiments(X_train, X_test, y_train, y_test)
        experiment_times['hyperparameter_experiments'] = time.time() - start_time

        # Print summary for Experiment 2
        hp_df = pd.DataFrame(hp_metrics)
        best_hp = hp_df.loc[hp_df['mean_angle_degrees'].idxmin()]
        print(f"\nBest hyperparameters:")
        print(f"  Learning rate: {best_hp['learning_rate']}")
        print(f"  Batch size: {best_hp['batch_size']}")
        print(f"  Seed: {best_hp['seed']}")
        print(f"  Mean principal angle: {best_hp['mean_angle_degrees']:.2f}°")

    if EXPERIMENT_MODE == 3 or EXPERIMENT_MODE == 'all':
        print(f"\n{'='*80}")
        print("Running Experiment 3: Optimization search")
        print(f"{'='*80}")
        start_time = time.time()
        opt_metrics = run_optimization_experiments(X_train, X_test, y_train, y_test)
        experiment_times['optimization_experiments'] = time.time() - start_time

        # Print summary for Experiment 3
        opt_df = pd.DataFrame(opt_metrics)
        best_opt = opt_df.loc[opt_df['mean_angle_degrees'].idxmin()]
        print(f"\nBest optimization `config`uration:")
        print(f"  k={best_opt['k']}, epochs={best_opt['epochs']}")
        print(f"  init={best_opt['init_strategy']}, lr_schedule={best_opt['lr_schedule']}")
        print(f"  Mean principal angle: {best_opt['mean_angle_degrees']:.4f}°")

    if EXPERIMENT_MODE == 4 or EXPERIMENT_MODE == 'all':
        print(f"\n{'='*80}")
        print("Running Experiment 4: Constraint isolation")
        print(f"{'='*80}")
        start_time = time.time()
        constraint_metrics = run_constraint_experiments(X_train, X_test, y_train, y_test)
        experiment_times['constraint_experiments'] = time.time() - start_time

        # Print summary for Experiment 4
        const_df = pd.DataFrame(constraint_metrics)
        best_const = const_df.loc[const_df['mean_angle_degrees'].idxmin()]
        print(f"\nBest constraint configuration:")
        print(f"  {best_const['configuration']} (k={best_const['k']})")
        print(f"  Mean principal angle: {best_const['mean_angle_degrees']:.4f}°")

    # Save experiment configuration
    total_experiment_time = sum(experiment_times.values())
    config = {
        'experiment_mode': EXPERIMENT_MODE,
        'random_seed': RANDOM_SEED,
        'dataset': {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'features': X_train.shape[1],
            'max_samples': max_samples
        },
        'timing': {
            'data_load_time': load_time,
            **experiment_times,
            'total_time': load_time + total_experiment_time
        }
    }

    # Add mode-specific config
    if EXPERIMENT_MODE == 1 or EXPERIMENT_MODE == 'all':
        config['baseline_hyperparams'] = BASELINE_HYPERPARAMS
        config['k_values'] = K_VALUES
    if EXPERIMENT_MODE == 2 or EXPERIMENT_MODE == 'all':
        config['hyperparam_grid'] = HYPERPARAM_GRID
    if EXPERIMENT_MODE == 3 or EXPERIMENT_MODE == 'all':
        config['optimization_config'] = OPTIMIZATION_CONFIG
    if EXPERIMENT_MODE == 4 or EXPERIMENT_MODE == 'all':
        config['constraint_config'] = {k: v for k, v in CONSTRAINT_CONFIG.items() if k != 'configurations'}
        config['constraint_config']['num_configurations'] = len(CONSTRAINT_CONFIG['configurations'])

    with open('results/experiment_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Print final summary
    print("\n\n" + "="*80)
    print(" "*30 + "EXPERIMENT COMPLETE!")
    print("="*80)
    print(f"\nExperiment Mode: {EXPERIMENT_MODE}")
    print(f"Total execution time: {config['timing']['total_time']:.2f}s "
          f"({config['timing']['total_time']/60:.1f} minutes)")
    print(f"  Data loading: {load_time:.2f}s")
    for exp_name, exp_time in experiment_times.items():
        print(f"  {exp_name}: {exp_time:.2f}s ({exp_time/60:.1f} minutes)")

    print("\nResults saved to:")
    print("  - results/models/            (PCA and autoencoder models)")
    print("  - results/visualizations/    (All plots and figures)")
    print("  - results/metrics/           (CSV files with all metrics)")
    print("  - results/experiment_config.json  (Experiment configuration)")

    print("\n" + "="*80)
    print("All results have been saved to the 'results/' directory.")
    print("You can now analyze the results and write your report!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
