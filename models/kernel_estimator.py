#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kernel Estimator implementation for both classification (Example 2.4) and regression (Example 3.1).

This estimator implements both:
1. Classification with 4 classes (Example 2.4)
2. Regression for continuous targets (Example 3.1)

Both use the Epanechnikov kernel with bandwidth parameter h=0.1.
"""

import numpy as np
from models.base_estimator import BaseEstimator

class KernelEstimator(BaseEstimator):
    """
    Kernel Estimator for both classification and regression using Epanechnikov kernel.
    
    Parameters:
    -----------
    task : str, default='regression'
        The task to perform. Either 'classification' or 'regression'.
        - For classification (Example 2.4): 4-class problem in 2D
        - For regression (Example 3.1): Continuous output in 1D
    h : float, default=0.1
        Bandwidth parameter controlling the kernel width
    kernel : str, default='epanechnikov'
        The kernel function to use. Currently only supports 'epanechnikov'.
    random_state : int, default=None
        Random seed for reproducibility.
    """
    
    def __init__(self, task='regression', h=0.1, kernel='epanechnikov', random_state=None):
        super().__init__(task=task, random_state=random_state)
        self.h = h
        if kernel != 'epanechnikov':
            raise ValueError("Only Epanechnikov kernel is supported")
        self.kernel = kernel
        self.X_train = None
        self.y_train = None
        self.classes_ = None  # For classification
    
    def _epanechnikov_kernel(self, x):
        """
        Epanechnikov kernel function K(x) as defined in the paper:
        K(x) = (3/4)(1 - ||x||₂²) if ||x||₂ ≤ 1, 0 otherwise
        """
        norm = np.linalg.norm(x)
        if norm <= 1:
            return (3/4) * (1 - norm**2)
        return 0
    
    def _kernel_estimator(self, point):
        """
        Compute the kernel estimator according to:
        - For classification (Example 2.4): Returns class probabilities
        - For regression (Example 3.1): Returns continuous prediction
        """
        if self.task == 'regression':
            # Regression: compute weighted average of targets
            numerator = 0  # Σ K((x-X_i)/h)Y_i
            denominator = 0  # Σ K((x-X_i)/h)
            
            for i in range(len(self.X_train)):
                scaled_diff = (point - self.X_train[i]) / self.h
                kernel_value = self._epanechnikov_kernel(scaled_diff)
                numerator += kernel_value * self.y_train[i]
                denominator += kernel_value
            
            return numerator / denominator if denominator > 0 else 0
            
        else:  # classification
            # Classification: compute class probabilities
            class_votes = np.zeros(len(self.classes_))
            total_weight = 0
            
            for i in range(len(self.X_train)):
                scaled_diff = (point - self.X_train[i]) / self.h
                kernel_value = self._epanechnikov_kernel(scaled_diff)
                if kernel_value > 0:
                    class_idx = np.where(self.classes_ == self.y_train[i])[0][0]
                    class_votes[class_idx] += kernel_value
                    total_weight += kernel_value
            
            # Return normalized class probabilities
            if total_weight > 0:
                return class_votes / total_weight
            return np.ones(len(self.classes_)) / len(self.classes_)
    
    def fit(self, X, y):
        """
        Fit the estimator to the training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data features
            - For classification (Example 2.4): 2D points
            - For regression (Example 3.1): 1D points
        y : array-like of shape (n_samples,)
            Training data targets
            - For classification: Labels in {1,2,3,4}
            - For regression: Continuous values
        """
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        
        if self.task == 'classification':
            self.classes_ = np.unique(y)
            if len(self.classes_) != 4:
                raise ValueError("Example 2.4 requires exactly 4 classes")
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict using the kernel estimator.
        
        Parameters:
        -----------
        X : array-like
            Test data features
        
        Returns:
        --------
        y_pred : array-like
            - For classification: Predicted class labels
            - For regression: Predicted continuous values
        """
        if not self.is_fitted:
            raise RuntimeError("Estimator has not been fitted yet.")
        
        X = np.asarray(X)
        y_pred = np.zeros(len(X))
        
        for i in range(len(X)):
            pred = self._kernel_estimator(X[i])
            if self.task == 'regression':
                y_pred[i] = pred
            else:  # classification
                y_pred[i] = self.classes_[np.argmax(pred)]
        
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities for classification.
        Only available for classification task (Example 2.4).
        
        Parameters:
        -----------
        X : array-like
            Test data features
        
        Returns:
        --------
        probas : array-like of shape (n_samples, n_classes)
            Class probabilities for each sample
        """
        if self.task != 'classification':
            raise ValueError("predict_proba is only available for classification")
        
        if not self.is_fitted:
            raise RuntimeError("Estimator has not been fitted yet.")
        
        X = np.asarray(X)
        probas = np.zeros((len(X), len(self.classes_)))
        
        for i in range(len(X)):
            probas[i] = self._kernel_estimator(X[i])
        
        return probas 