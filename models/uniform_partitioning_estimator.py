#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Uniform Partitioning Estimator implementation for regression.

This estimator divides the feature space into uniform cells and makes predictions
based on the training samples that fall within the same cell as the test point.

The implementation follows the three-step process described in the paper:
1. Partitioning: Divides the feature space into 2^κ uniform cells
2. Localization: Determines which cell a given point belongs to
3. Estimation: Computes the prediction for a point by averaging the target values of training points in the same cell
"""

import numpy as np
from models.base_estimator import BaseEstimator


class UniformPartitioningEstimator(BaseEstimator):
    """
    Uniform Partitioning Estimator for regression.
    
    This estimator divides the feature space into uniform cells and makes predictions
    based on the training samples that fall within the same cell as the test point.
    
    Parameters:
    -----------
    kappa : int, default=3
        Parameter controlling the number of cells (2^kappa)
    tolerance : float, default=1e-6
        Tolerance for cell boundaries
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, kappa=3, tolerance=1e-6, random_state=None):
        super().__init__(task='regression', random_state=random_state)
        self.kappa = kappa
        self.tolerance = tolerance
        self.cell_matrix = None
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """
        Fit the estimator to the training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data features
        y : array-like of shape (n_samples,)
            Training data targets
            
        Returns:
        --------
        self : object
            Returns self
        """
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        
        # Get dimensionality of the feature space
        n_samples, n_features = self.X_train.shape
        
        # Create the cell matrix (partitioning of the feature space)
        self.cell_matrix = self._partitioning(self.kappa, self.X_train, self.tolerance, n_features)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict using the uniform partitioning estimator.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test data features
            
        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            Predicted values
        """
        if not self.is_fitted:
            raise RuntimeError("Estimator has not been fitted yet.")
        
        X = np.asarray(X)
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)
        
        # For each test point
        for i in range(n_samples):
            # Localize the point to its cell
            cell_index = self._localization(self.cell_matrix, X[i, :], 0, 0, self.kappa)
            
            # Compute the prediction using the partitioning estimator
            y_pred[i] = self._part_estimator(self.cell_matrix, cell_index, self.X_train, self.y_train)
        
        return y_pred
    
    def _partitioning(self, kappa, A, tolerance, d):
        """
        Partition the state space into 2^kappa uniform cells.
        Follows exactly Source Code 1 from the paper.
        """
        # Initialize cell matrix RR
        RR = np.empty(shape=(kappa+1, 2**(kappa)), dtype=object)
        
        # Initialize first entry of RR
        RR1 = np.zeros((d, 2))
        
        # Set up the first cell
        for i in range(d):
            RR1[i, 0] = np.amin(A[:, i]) - tolerance
            RR1[i, 1] = np.amax(A[:, i]) + tolerance
        
        # Specification of first cell
        RR[0][0] = RR1
        
        # Successively specify subsequent cells according to Binary Tree Cuboid structure
        for p in range(kappa):
            for q in range(2**p):
                # Find dimension with maximum width
                maxi = np.copy(RR[p][q][0][1] - RR[p][q][0][0])
                iter_dim = 0
                
                for i in range(d):
                    comp = np.copy(RR[p][q][i][1] - RR[p][q][i][0])
                    if comp > maxi:
                        iter_dim = i
                        maxi = np.copy(comp)
                
                # Create two new cells by splitting the current cell
                RR[p+1][2*q] = np.copy(RR[p][q])
                RR[p+1][2*q][iter_dim][1] = (RR[p][q][iter_dim][0] + RR[p][q][iter_dim][1]) / 2
                
                RR[p+1][2*q+1] = np.copy(RR[p][q])
                RR[p+1][2*q+1][iter_dim][0] = (RR[p][q][iter_dim][0] + RR[p][q][iter_dim][1]) / 2
        
        return RR
    
    def _localization(self, RR, point, counter, index, kappa):
        """
        Determine the cell in which a point is located using the binary tree structure.
        Follows exactly Source Code 3 from the paper.
        """
        if counter == kappa:  # stop if last row of RR is reached
            return index
            
        # Recursively utilize binary tree structure of RR
        if (RR[counter+1, 2*index][:, 0] <= point).all() and (point <= RR[counter+1, 2*index][:, 1]).all():
            newindex = int(2*index)
            return self._localization(RR, point, counter+1, newindex, kappa)
        else:
            newindex = int(2*index+1)
            return self._localization(RR, point, counter+1, newindex, kappa)
    
    def _part_estimator(self, RR, index, Data, targets):
        """
        Compute the partitioning estimator m_n(point) for regression.
        Follows exactly Source Code 4 from the paper.
        """
        factor1 = 0  # Numerator
        factor2 = 0  # Denominator
        
        # Compute numerator and denominator in equation (2.1)
        n_samples = len(targets)
        for i in range(n_samples):
            if (RR[-1, index][:, 0] <= Data[i]).all() and (Data[i] <= RR[-1, index][:, 1]).all():
                factor1 += targets[i]
                factor2 += 1
        
        # Compute the estimate
        if factor2 == 0:
            m_n = 0  # Return 0 if no points in cell
        else:
            m_n = factor1 / factor2
            
        return m_n 