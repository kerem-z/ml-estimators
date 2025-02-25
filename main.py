
"""
Main script for running different estimators on classification and regression tasks.
Supports:
- Methods: Partitioning, Kernel, and kNN estimators
- Tasks: Classification (Example 2.4) and Regression (Example 3.1)
- Metrics: Computational time and errors (training/test)
"""

import argparse
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from tabulate import tabulate
import os


from models.data_dependent_partitioning import DataDependentPartitioningEstimator
from models.kernel_estimator import KernelEstimator
from models.knn_estimator import KNNEstimator


from data.synthetic_data import (
    generate_classification_data,
    generate_regression_data,
    load_dataset,
    save_dataset
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run estimators on classification/regression tasks.'
    )
    
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['partitioning', 'kernel', 'knn', 'all'],
        default='all',
        help='Estimator method to use'
    )
    
   
    parser.add_argument(
        '--task',
        type=str,
        choices=['classification', 'regression', 'all'],
        default='all',
        help='Task to perform'
    )
    
    
    parser.add_argument(
        '--kappa',
        type=int,
        default=7,
        help='Kappa parameter for partitioning estimator'
    )
    parser.add_argument(
        '--bandwidth',
        type=float,
        default=0.1,
        help='Bandwidth parameter for kernel estimator'
    )
    parser.add_argument(
        '--k_neighbors',
        type=int,
        default=5,
        help='Number of neighbors for kNN estimator'
    )
    
    
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Proportion of dataset to use for testing'
    )
    
    return parser.parse_args()

def get_data(task, random_state=42):
    """Load data directly from the datasets folder."""
    if task == 'classification':
        dataset_path = os.path.join('ml-estimators', 'data', 'datasets', 'classification_example_2.4.npz')
    else:  
        dataset_path = os.path.join('ml-estimators', 'data', 'datasets', 'regression_example_3.1.npz')
    
    try:
        data = np.load(dataset_path)
        X = data['X']
        y = data['y']
        print(f"Successfully loaded dataset from {dataset_path}")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    except Exception as e:
        print(f"Error loading dataset {dataset_path}: {e}")
        print("Generating new data...")
        if task == 'classification':
            X, y = generate_classification_data(random_state=random_state)
        else:
            X, y = generate_regression_data(random_state=random_state)
        
       
        save_dir = os.path.join('ml-estimators', 'data', 'datasets')
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(dataset_path, X=X, y=y)
        print(f"Saved new dataset to {dataset_path}")
        return X, y

def get_estimator(method, task, args):
    """Create an estimator instance based on method and task."""
    if method == 'partitioning':
        return DataDependentPartitioningEstimator(
            task=task,
            kappa=args.kappa,
            random_state=args.random_state
        )
    elif method == 'kernel':
        return KernelEstimator(
            task=task,
            h=args.bandwidth,
            random_state=args.random_state
        )
    else:  
        return KNNEstimator(
            task=task,
            k=args.k_neighbors,
            random_state=args.random_state
        )

def evaluate_estimator(estimator, X_train, X_test, y_train, y_test, task):
    """
    Evaluate an estimator's performance.
    Returns computational time and errors.
    """
    # Measure training time
    start_time = time.time()
    estimator.fit(X_train, y_train)
    y_train_pred = estimator.predict(X_train)
    y_test_pred = estimator.predict(X_test)
    comp_time = time.time() - start_time
    
    
    if task == 'classification':
        train_error = 1 - accuracy_score(y_train, y_train_pred)
        test_error = 1 - accuracy_score(y_test, y_test_pred)
    else:  # regression
        train_error = mean_squared_error(y_train, y_train_pred)
        test_error = mean_squared_error(y_test, y_test_pred)
    
    return comp_time, train_error, test_error, estimator

def main():
    """Main function."""
    args = parse_args()
    
    
    methods = ['partitioning', 'kernel', 'knn'] if args.method == 'all' else [args.method]
    tasks = ['classification', 'regression'] if args.task == 'all' else [args.task]
    
    
    results = {}
    
   
    for task in tasks:
        print(f"\nRunning {task} experiments...")
        
        
        X, y = get_data(task, args.random_state)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state
        )
        
        print(f"Data shapes: X_train {X_train.shape}, X_test {X_test.shape}")
        
        for method in methods:
            print(f"\nEvaluating {method} estimator...")
            
            
            comp_time, train_error, test_error, _ = evaluate_estimator(
                get_estimator(method, task, args),
                X_train, X_test, y_train, y_test, task
            )
            
            
            key = f"{task}_{method}"
            results[key] = {
                'time': comp_time,
                'train_error': train_error,
                'test_error': test_error
            }
            
            
            print(f"Computational time: {comp_time:.2f} seconds")
            print(f"Training error: {train_error:.4f}")
            print(f"Test error: {test_error:.4f}")
    
    
    print("\nResults Summary:")
    
    
    table_data = []
    headers = ["Task", "Estimator", "Time (s)", "Train Error", "Test Error"]
    
    for task in tasks:
        for method in methods:
            key = f"{task}_{method}"
            r = results[key]
            table_data.append([
                task.capitalize(),
                f"{method} estimator",
                f"{r['time']:.2f}",
                f"{r['train_error']:.4f}",
                f"{r['test_error']:.4f}"
            ])
    
    
    print(tabulate(
        table_data,
        headers=headers,
        tablefmt="grid",
        stralign="left",
        numalign="right"
    ))

if __name__ == "__main__":
    main()
