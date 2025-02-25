#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
k-Nearest Neighbors (kNN) Estimator implementation for both:
- Classification (Example 2.4): 4-class problem in 2D
- Regression (Example 3.1): 1D regression with oscillating pattern

This follows Example 2.3 and 3.1 from the paper, implementing
equation (2.3) for both classification and regression tasks.
"""

import numpy as np
from models.base_estimator import BaseEstimator
from scipy.spatial.distance import cdist


class KNNEstimator(BaseEstimator):
    """
    k-Nearest Neighbors (kNN) Estimator.
    
    Parameters:
    -----------
    task : str, default='classification'
        The task to perform. Either 'classification' or 'regression'
    k : int, default=5
        Number of nearest neighbors to use (as in paper Example 2.3)
    metric : str, default='euclidean'
        Distance metric to use for neighbor computation
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, task='classification', k=5, metric='euclidean', random_state=None):
        super().__init__(task=task, random_state=random_state)
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None
        self.classes_ = None  # For classification
    
    def _get_neighbors(self, X):
        """
        Find k nearest neighbors for each point in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Points to find neighbors for
            
        Returns:
        --------
        indices : array-like of shape (n_samples, k)
            Indices of k nearest neighbors for each point
        distances : array-like of shape (n_samples, k)
            Distances to k nearest neighbors for each point
        """
        # Compute distances between X and training data
        distances = cdist(X, self.X_train, metric=self.metric)
        
        # Get indices of k nearest neighbors
        neighbor_indices = np.argsort(distances, axis=1)[:, :self.k]
        
        # Get corresponding distances
        neighbor_distances = np.take_along_axis(distances, neighbor_indices, axis=1)
        
        return neighbor_indices, neighbor_distances
    
    def _knn_estimator(self, neighbors_idx, distances):
        """
        Compute the kNN estimator according to equation (2.3).
        
        Parameters:
        -----------
        neighbors_idx : array-like of shape (n_samples, k)
            Indices of k nearest neighbors
        distances : array-like of shape (n_samples, k)
            Distances to k nearest neighbors
            
        Returns:
        --------
        predictions : array-like of shape (n_samples,)
            Predicted values/labels
        """
        n_samples = len(neighbors_idx)
        
        if self.task == 'classification':
            predictions = np.zeros(n_samples)
            for i in range(n_samples):
                # Get labels of k nearest neighbors
                neighbor_labels = self.y_train[neighbors_idx[i]]
                # Return most common class
                unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
                predictions[i] = unique_labels[np.argmax(counts)]
        else:
            # For regression, compute weighted average based on distances
            weights = 1 / (distances + self.eps)  # Add eps to avoid division by zero
            weights /= weights.sum(axis=1, keepdims=True)
            predictions = np.sum(self.y_train[neighbors_idx] * weights, axis=1)
        
        return predictions
    
    def _knn_proba(self, neighbors_idx, distances):
        """
        Compute class probabilities for classification.
        
        Parameters:
        -----------
        neighbors_idx : array-like of shape (n_samples, k)
            Indices of k nearest neighbors
        distances : array-like of shape (n_samples, k)
            Distances to k nearest neighbors
            
        Returns:
        --------
        probas : array-like of shape (n_samples, n_classes)
            Class probabilities for each sample
        """
        n_samples = len(neighbors_idx)
        n_classes = len(self.classes_)
        probas = np.zeros((n_samples, n_classes))
        
        # Compute distance-based weights
        weights = 1 / (distances + self.eps)
        weights /= weights.sum(axis=1, keepdims=True)
        
        # For each sample
        for i in range(n_samples):
            # Get labels of k nearest neighbors
            neighbor_labels = self.y_train[neighbors_idx[i]]
            
            # Accumulate weighted votes for each class
            for j, label in enumerate(neighbor_labels):
                idx = np.where(self.classes_ == label)[0][0]
                probas[i, idx] += weights[i, j]
            
            # Normalize probabilities
            probas[i] /= probas[i].sum()
        
        return probas
    
    def fit(self, X, y):
        """
        Fit the estimator to the training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data features
        y : array-like of shape (n_samples,)
            Training data targets/labels
            
        Returns:
        --------
        self : object
            Returns self
        """
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        
        if self.task == 'classification':
            self.classes_ = np.unique(y)
            
        self.eps = np.finfo(float).eps  # Small constant to avoid division by zero
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict using the kNN estimator.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test data features
            
        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            Predicted values/labels
        """
        if not self.is_fitted:
            raise RuntimeError("Estimator has not been fitted yet.")
        
        X = np.asarray(X)
        
        # Find k nearest neighbors
        neighbors_idx, distances = self._get_neighbors(X)
        
        # Compute predictions
        return self._knn_estimator(neighbors_idx, distances)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for classification.
        Only available when task='classification'.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
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
        
        # Find k nearest neighbors
        neighbors_idx, distances = self._get_neighbors(X)
        
        # Compute probabilities
        return self._knn_proba(neighbors_idx, distances) 