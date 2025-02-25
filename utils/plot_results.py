#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to visualize results for all methods and tasks.
Creates comparison plots showing:
1. Classification: Decision boundaries and data points
2. Regression: True function, data points, and predictions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

# Import estimators
from models.data_dependent_partitioning import DataDependentPartitioningEstimator
from models.kernel_estimator import KernelEstimator
from models.knn_estimator import KNNEstimator

def load_data(task):
    """Load dataset directly from npz file."""
    if task == 'classification':
        dataset_path = os.path.join('ml-estimators', 'data', 'datasets', 'classification_example_2.4.npz')
    else:
        dataset_path = os.path.join('ml-estimators', 'data', 'datasets', 'regression_example_3.1.npz')
    
    data = np.load(dataset_path)
    return data['X'], data['y']

def create_estimators(task, random_state=42):
    """Create all three estimators for given task."""
    return {
        'Partitioning': DataDependentPartitioningEstimator(task=task, kappa=7, random_state=random_state),
        'Kernel': KernelEstimator(task=task, h=0.1, random_state=random_state),
        'kNN': KNNEstimator(task=task, k=5, random_state=random_state)
    }

def plot_classification_results(estimators, X_train, y_train, save_path):
    """Plot classification results showing decision boundaries."""
    n_estimators = len(estimators)
    fig, axes = plt.subplots(1, n_estimators, figsize=(15, 5))
    
    # Create colormap
    colors = ['#FF9999', '#99FF99', '#99CCFF', '#FFCC99']
    cmap = ListedColormap(colors)
    
    # Create mesh grid
    x_min, x_max = 0, 2
    y_min, y_max = -0.5, 2
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    
    for ax, (name, estimator) in zip(axes, estimators.items()):
        # Plot decision regions
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = estimator.predict(grid_points)
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.3)
        
        # Plot training points
        scatter = ax.scatter(X_train[:, 0], X_train[:, 1],
                           c=y_train, cmap=cmap,
                           edgecolors='black', linewidth=0.5,
                           alpha=0.6, s=20)
        
        # Plot test point (1, 1.35)
        ax.plot(1, 1.35, 'k*', markersize=15, label='Test Point')
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"{name} Estimator")
        ax.grid(True, alpha=0.15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Classification plot saved to {save_path}")

def plot_regression_results(estimators, X_train, y_train, save_path):
    """Plot regression results showing predictions."""
    n_estimators = len(estimators)
    fig, axes = plt.subplots(1, n_estimators, figsize=(15, 5))
    
    # Generate points for visualization
    x_plot = np.linspace(-6, 6, 1000).reshape(-1, 1)
    
    for ax, (name, estimator) in zip(axes, estimators.items()):
        # Plot training data
        ax.scatter(X_train, y_train, color='blue', alpha=0.5,
                  label='Training data', s=20)
        
        # Plot predictions
        y_pred = estimator.predict(x_plot)
        ax.plot(x_plot, y_pred, 'r-', label='Prediction', linewidth=2)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f"{name} Estimator")
        ax.legend()
        ax.grid(True, alpha=0.15)
        ax.set_xlim(-7, 7)
        ax.set_ylim(-1.5, 1.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Regression plot saved to {save_path}")

def main():
    """Generate visualization plots for all methods and tasks."""
    # Create assets directory
    assets_dir = os.path.join('ml-estimators', 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    
    random_state = 42
    test_size = 0.2
    
    # Process classification task
    print("\nGenerating classification visualizations...")
    X, y = load_data('classification')
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    estimators = create_estimators('classification', random_state)
    for estimator in estimators.values():
        estimator.fit(X_train, y_train)
    
    plot_classification_results(
        estimators,
        X_train,
        y_train,
        os.path.join(assets_dir, 'classification_comparison.png')
    )
    
    # Process regression task
    print("\nGenerating regression visualizations...")
    X, y = load_data('regression')
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    estimators = create_estimators('regression', random_state)
    for estimator in estimators.values():
        estimator.fit(X_train, y_train)
    
    plot_regression_results(
        estimators,
        X_train,
        y_train,
        os.path.join(assets_dir, 'regression_comparison.png')
    )
    
    print("\nAll visualizations completed!")

if __name__ == "__main__":
    main() 