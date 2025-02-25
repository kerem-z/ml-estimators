#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Synthetic data generation for Examples 2.3, 2.4, and 3.1 from the paper.

Example 2.3 & 2.4 (Classification):
- d = 2 dimensions
- n = 2^10 samples
- Y = {1,2,3,4} classes
- Domain: [0,2] × [-0.5,2]

Example 3.1 (Regression):
- d = 1 dimension
- n = 2^10 samples
- Y = ℝ (continuous values)
- Domain: [-8,8]
- Function: m(x) = 0.3x² + sin(x)
"""

import numpy as np
from pathlib import Path

def generate_classification_data(n_samples=2**10, random_state=42):
    """
    Generate synthetic classification data for Examples 2.3 & 2.4.
    
    Args:
        n_samples: Number of samples (default: 2^10 as in paper)
        random_state: Random seed for reproducibility
        
    Returns:
        X: Features array of shape (n_samples, 2)
        y: Labels array of shape (n_samples,) with values in {1,2,3,4}
    """
    np.random.seed(random_state)
    
    # Generate features in domain [0,2] × [-0.5,2]
    X = np.random.uniform(
        low=[0, -0.5],
        high=[2, 2],
        size=(n_samples, 2)
    )
    
    # Initialize labels
    y = np.zeros(n_samples, dtype=int)
    
    # Assign classes based on regions as shown in Figure 7
    for i in range(n_samples):
        x, y_coord = X[i]
        
       
        if y_coord > 1 and x < 0.5:
            y[i] = 1
        
        elif y_coord < 0.3 and x < 0.5:
            y[i] = 2
        
        elif 0.5 < x < 1.5 and -0.2 < y_coord < 0.5:
            y[i] = 3
        
        else:
            y[i] = 4
    
    return X, y

def generate_regression_data(n_samples=2**10, random_state=42):
    """
    Generate synthetic regression data for Example 3.1.
    Creates a compact sinusoidal pattern with:
    - Multiple oscillations (4-5 complete waves)
    - Dense points near center, sparser at edges
    - Heteroscedastic noise (increases with |x|)
    
    Args:
        n_samples: Number of samples (default: 2^10 as in paper)
        random_state: Random seed for reproducibility
        
    Returns:
        X: Features array of shape (n_samples, 1)
        y: Target array of shape (n_samples,)
    """
    np.random.seed(random_state)
    
    # Generate features with denser points near center
    X = np.random.normal(0, 2, (n_samples, 1))
    X = np.clip(X, -6, 6)  # Clip to desired range
    
    # Higher frequency oscillation
    base_freq = 3.0  # Creates 4-5 complete waves
    phase = 0.5  # Phase shift to match pattern
    base_oscillation = np.sin(base_freq * X[:, 0] + phase)
    
    # Base amplitude
    y = base_oscillation
    
    # Add heteroscedastic noise that increases with |x|
    noise_scale = 0.15 + 0.05 * np.abs(X[:, 0])
    y += noise_scale * np.random.normal(0, 1, n_samples)
    
    return X, y

def save_dataset(X, y, filename):
    """Save dataset to disk."""
    save_dir = Path(__file__).parent / 'datasets'
    save_dir.mkdir(exist_ok=True, parents=True)
    
    filepath = save_dir / filename
    np.savez_compressed(filepath, X=X, y=y)
    print(f"Dataset saved to {filepath}")

def load_dataset(filename):
    """Load dataset from disk."""
    filepath = Path(__file__).parent / 'datasets' / filename
    
    if not filepath.with_suffix('.npz').exists():
        return None, None
    
    data = np.load(filepath.with_suffix('.npz'))
    print(f"Dataset loaded from {filepath}")
    return data['X'], data['y']

if __name__ == "__main__":
    
    print("Generating classification dataset...")
    X, y = generate_classification_data()
    print(f"Generated classification dataset with {len(X)} samples")
    print(f"X shape: {X.shape}, range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"Classes: {np.unique(y)}")
    save_dataset(X, y, 'classification_example_2.4')
    
    
    print("\nGenerating regression dataset...")
    X, y = generate_regression_data()
    print(f"Generated regression dataset with {len(X)} samples")
    print(f"X shape: {X.shape}, range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"y range: [{y.min():.2f}, {y.max():.2f}]")
    save_dataset(X, y, 'regression_example_3.1')
    
    
    print("\nVerifying data loading...")
    for name in ['classification_example_2.4', 'regression_example_3.1']:
        X, y = load_dataset(name)
        if X is not None:
            print(f"\n{name}:")
            print(f"X shape: {X.shape}")
            print(f"y shape: {y.shape}")
            print(f"X range: [{X.min():.2f}, {X.max():.2f}]")
            if name.startswith('classification'):
                print(f"Classes: {np.unique(y)}")
            else:
                print(f"y range: [{y.min():.2f}, {y.max():.2f}]") 