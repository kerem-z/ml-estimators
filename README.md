

## Project Structure

```
ml-estimators/
├── data/
│   ├── synthetic_data.py    # Data generation for Examples 2.4 and 3.1
│   └── datasets/            # Pre-generated datasets (.npz files)
├── models/
│   ├── base_estimator.py    # Abstract base class for all estimators
│   ├── data_dependent_partitioning.py  # BTC-based partitioning
│   ├── kernel_estimator.py  # Epanechnikov kernel estimator
│   └── knn_estimator.py     # k-Nearest Neighbors
├── utils/
│   ├── plot_data.py        # Academic-style dataset visualization
│   └── plot_results.py     # Comparison plots for all methods
├── assets/                  # Generated plots and visualizations
├── main.py                  # Main script for running experiments
└── README.md
```

## Features

### Implemented Methods
1. **Data-Dependent Partitioning**
   - Binary Tree Cuboid (BTC) strategy
   - Adaptive cell splitting based on empirical standard deviation
   - Configurable partition depth (κ parameter)

2. **Kernel Estimation**
   - Epanechnikov kernel implementation
   - Adaptive bandwidth parameter (h)
   - Support for both classification and regression

3. **k-Nearest Neighbors**
   - Distance-weighted predictions
   - Configurable number of neighbors (k)
   - Euclidean distance metric

### Tasks
1. **Classification (Example 2.4)**
   - 4-class problem in 2D space
   - Domain: [0,2] × [-0.5,2]
   - 1024 training samples
   - Test point at (1, 1.35)

2. **Regression (Example 3.1)**
   - Oscillating pattern with heteroscedastic noise
   - Domain: [-6,6]
   - 1024 training samples
   - Noise scale increases with |x|

## Usage Examples

### Running Experiments

The main script (`main.py`) supports various command-line arguments:

```bash
python main.py [options]
```

#### Available Arguments
- `--method`: partitioning, kernel, knn, all (default)
- `--task`: classification, regression, all (default)
- `--kappa`: Partitioning depth (default: 7)
- `--bandwidth`: Kernel bandwidth (default: 0.1)
- `--k_neighbors`: Number of neighbors (default: 5)
- `--random_state`: Random seed (default: 42)
- `--test_size`: Test set proportion (default: 0.2)

#### Example Commands

1. Basic Usage:
```bash
# Run all methods on all tasks
python main.py

# Run specific method on specific task
python main.py --method kernel --task classification
python main.py --method knn --task regression
```

2. Partitioning Estimator Examples:
```bash
# Try different partition depths
python main.py --method partitioning --kappa 5
python main.py --method partitioning --kappa 8 --task classification

# Compare performance with different test sizes
python main.py --method partitioning --test_size 0.1
python main.py --method partitioning --test_size 0.3
```

3. Kernel Estimator Examples:
```bash
# Experiment with bandwidth
python main.py --method kernel --bandwidth 0.05
python main.py --method kernel --bandwidth 0.2 --task regression

# Classification with different test sizes
python main.py --method kernel --task classification --test_size 0.15
```

4. kNN Examples:
```bash
# Try different numbers of neighbors
python main.py --method knn --k_neighbors 3
python main.py --method knn --k_neighbors 10 --task regression

# Compare kNN performance across tasks
python main.py --method knn --k_neighbors 7 --task all
```

5. Comparative Analysis:
```bash
# Compare all methods with custom parameters
python main.py --method all --kappa 6 --bandwidth 0.15 --k_neighbors 8

# Run multiple experiments with different random seeds
python main.py --random_state 42
python main.py --random_state 123
```

### Generating Visualizations

1. Dataset Visualization:
```bash
# Generate academic-style plots
python utils/plot_data.py
```
Creates:
- `assets/classification_data_academic.png`: 4-class dataset
- `assets/regression_data_academic.png`: Regression dataset with noise

2. Method Comparison:
```bash
# Generate comparison plots
python utils/plot_results.py
```
Creates:
- `assets/classification_comparison.png`: Decision boundaries
- `assets/regression_comparison.png`: Regression curves

## Implementation Details

### Base Estimator
- Abstract base class defining common interface
- Supports both classification and regression
- Implements scoring methods (accuracy/R²)

### Data-Dependent Partitioning
- Implements Algorithm 2.2 from the paper
- Uses empirical standard deviation for splitting
- Supports probability estimates for classification

### Kernel Estimator
- Epanechnikov kernel: K(x) = (3/4)(1 - ||x||₂²) if ||x||₂ ≤ 1
- Bandwidth parameter controls smoothing
- Weighted averaging for predictions

### k-Nearest Neighbors
- Distance-weighted voting for classification
- Weighted averaging for regression
- Efficient neighbor search using scipy

## Output Format

The experiments output a formatted table:
```
+----------------+----------------------+-----------+-------------+------------+
| Task           | Estimator           | Time (s)  | Train Error | Test Error |
+----------------+----------------------+-----------+-------------+------------+
| Classification | partitioning        | 0.52      | 0.0488     | 0.1122     |
| Classification | kernel              | 1.23      | 0.0134     | 0.0585     |
| Classification | knn                 | 0.31      | 0.0000     | 0.0439     |
| Regression     | partitioning        | 0.48      | 0.0961     | 0.1031     |
| Regression     | kernel              | 1.15      | 0.0845     | 0.0923     |
| Regression     | knn                 | 0.28      | 0.0789     | 0.0856     |
+----------------+----------------------+-----------+-------------+------------+
```
