#!/usr/bin/env python
# -*- coding: utf-8 -*-




import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import ListedColormap

# Set up matplotlib for publication quality
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 14
rcParams['figure.titlesize'] = 14
rcParams['axes.grid'] = True
rcParams['grid.alpha'] = 0.15
rcParams['figure.figsize'] = (8, 6)
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'

def plot_regression_data():
   
    # Load data directly from npz file
    dataset_path = os.path.join('ml-estimators', 'data', 'datasets', 'regression_example_3.1.npz')
    
    try:
        data = np.load(dataset_path)
        X = data['X']
        y = data['y']
        print(f"Successfully loaded regression dataset from {dataset_path}")
    except Exception as e:
        print(f"Error loading regression dataset: {e}")
        return
    
    # Create figure
    fig, ax = plt.subplots()
    
    
    scatter = ax.scatter(X, y, 
                        c='blue',
                        alpha=0.5,
                        s=30,
                        label='Training Data',
                        edgecolors='none')
    
    
    ax.set_xlabel(r'Input Variable $x$')
    ax.set_ylabel(r'Target Variable $y$')
    ax.set_title('Regression Dataset with Heteroscedastic Noise', pad=15)
    
    
    info_text = (
        f'n = {len(X)} samples\n'
        r'$x \in [-6, 6]$' + '\n'
        r'$y = \sin(3x + 0.5) + \epsilon(x)$' + '\n'
        r'$\epsilon(x) \sim \mathcal{N}(0, (0.15 + 0.05|x|)^2)$'
    )
    ax.text(0.02, 0.98, info_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set axis limits with some padding
    ax.set_xlim(-7, 7)
    ax.set_ylim(-1.5, 1.5)
    
    # Add legend
    ax.legend(loc='upper right', frameon=False)
    
    # Add grid with custom styling
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create assets directory if it doesn't exist
    assets_dir = os.path.join('ml-estimators', 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    
    # Save plot
    save_path = os.path.join(assets_dir, 'regression_data.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f" regression plot saved to {save_path}")

def plot_classification_data():
    
    # Load data directly from npz file
    dataset_path = os.path.join('ml-estimators', 'data', 'datasets', 'classification_example_2.4.npz')
    
    try:
        data = np.load(dataset_path)
        X = data['X']
        y = data['y']
        print(f"Successfully loaded classification dataset from {dataset_path}")
    except Exception as e:
        print(f"Error loading classification dataset: {e}")
        return
    
    # Create figure
    fig, ax = plt.subplots()
    
    # Create custom colormap for classes
    colors = ['#FF9999', '#99FF99', '#99CCFF', '#FFCC99']
    cmap = ListedColormap(colors)
    
    # Plot data points with enhanced styling
    scatter = ax.scatter(X[:, 0], X[:, 1],
                        c=y,
                        cmap=cmap,
                        alpha=0.6,
                        s=30,
                        edgecolors='black',
                        linewidth=0.5)
    
    # Plot test point from Example 2.4
    test_point = [1.0, 1.35]
    ax.plot(test_point[0], test_point[1], 'k*',
            markersize=15, label='Test Point (1.0, 1.35)')
    
    # Customize plot
    ax.set_xlabel(r'Feature $x_1$')
    ax.set_ylabel(r'Feature $x_2$')
    ax.set_title('4-Class Classification Dataset', pad=15)
    
    # Add text box with dataset details
    info_text = (
        f'n = {len(X)} samples\n'
        r'$x_1 \in [0, 2]$' + '\n'
        r'$x_2 \in [-0.5, 2]$' + '\n'
        'Classes: {1, 2, 3, 4}'
    )
    ax.text(0.02, 0.98, info_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set axis limits
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.5, 2)
    
    # Add legend
    legend1 = ax.legend(*scatter.legend_elements(),
                       title="Classes",
                       loc="upper right",
                       frameon=False)
    ax.add_artist(legend1)
    ax.legend(loc='upper center', frameon=False)
    
    # Add grid with custom styling
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    assets_dir = os.path.join('ml-estimators', 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    save_path = os.path.join(assets_dir, 'classification_data.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f" classification plot saved to {save_path}")

def main():
    """Generate both regression and classification plots."""
    plot_regression_data()
    plot_classification_data()

if __name__ == "__main__":
    main() 