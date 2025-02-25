#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base estimator module defining the common interface for all estimator methods.
All specific method implementations should inherit from this base class.
"""

import numpy as np
from abc import ABC, abstractmethod


class BaseEstimator(ABC):
    """
    Abstract base class for all estimator methods.
    
    This class defines the common interface that all estimator methods
    must implement, ensuring consistency across different implementations.
    """
    
    def __init__(self, task='classification', random_state=None):
        """
        Initialize the base estimator.
        
        Args:
            task: 'classification' or 'regression'
            random_state: Random seed for reproducibility
        """
        if task not in ['classification', 'regression']:
            raise ValueError(f"Task must be 'classification' or 'regression', got {task}")
        
        self.task = task
        self.random_state = random_state
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X, y):
        """
        Fit the estimator to the training data.
        
        Args:
            X: Training data features
            y: Training data targets
            
        Returns:
            self: The fitted estimator
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Test data features
            
        Returns:
            y_pred: Predicted values
        """
        if not self.is_fitted:
            raise RuntimeError("Estimator has not been fitted yet.")
        pass
    
    def score(self, X, y):
        """
        Calculate the score of the estimator on test data.
        
        For classification, this is accuracy.
        For regression, this is R^2.
        
        Args:
            X: Test data features
            y: Test data targets
            
        Returns:
            score: The score of the estimator
        """
        y_pred = self.predict(X)
        
        if self.task == 'classification':
            # Accuracy for classification
            return np.mean(y_pred == y)
        else:
            # R^2 for regression
            y_mean = np.mean(y)
            ss_total = np.sum((y - y_mean) ** 2)
            ss_residual = np.sum((y - y_pred) ** 2)
            
            if ss_total == 0:
                return 0  
            
            return 1 - (ss_residual / ss_total)
    
    def __repr__(self):
        """String representation of the estimator."""
        return f"{self.__class__.__name__}(task={self.task}, random_state={self.random_state})" 