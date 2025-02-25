
"""
Data-dependent partitioning estimator implementation for both:
- Classification (Example 2.4): 4-class problem in 2D
- Regression (Example 3.1): 1D regression with oscillating pattern

The estimator uses Binary Tree Cuboid (BTC) strategy to create
data-dependent partitions based on empirical standard deviation.
"""

import numpy as np
from models.base_estimator import BaseEstimator


class DataDependentPartitioningEstimator(BaseEstimator):
    """
    Data-dependent partitioning estimator using BTC strategy.
    
    Parameters:
    -----------
    task : str, default='classification'
        The task to perform. Either 'classification' or 'regression'
    kappa : int, default=7
        Parameter controlling the number of cells (2^kappa)
    tolerance : float, default=1e-6
        Tolerance for cell boundaries
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, task='classification', kappa=7, tolerance=1e-6, random_state=None):
        super().__init__(task=task, random_state=random_state)
        self.kappa = kappa
        self.tolerance = tolerance
        self.RR = None  # Cell matrix for rectangles
        self.Scell = None  # Cell matrix for points
        self.classes_ = None  # Unique classes for classification
        
    def _datadeppart(self, kappa, x, y, tolerance):
        """
        BTC strategy --> data-dependent partition.
        Implementation of Algorithm 2.2 from the manuscript.
        
        Parameters:
        -----------
        kappa : int
            Number of partition levels
        x : array-like
            Training data points
        y : array-like
            Training labels/values
        tolerance : float
            Distance tolerance for cell boundaries
        
        Returns:
        --------
        RR : array-like
            Cell matrix containing rectangles
        Scell : array-like
            Cell matrix containing points
        """
        d = x.shape[1]  
        
        # Initialize cell matrices
        SS = np.empty(shape=(kappa+1, 2**(kappa)), dtype=object)  # Points
        YS = np.empty(shape=(kappa+1, 2**(kappa)), dtype=object)  # Labels/values
        RR = np.empty(shape=(kappa + 1, 2 ** kappa), dtype=object)  # Rectangles
        
        
        SS[0][0] = x
        YS[0][0] = y
        
       
        RR1 = np.zeros((d, 2))
        for i in range(d):
            RR1[i, 0] = np.amin(x[:, i]) - tolerance
            RR1[i, 1] = np.amax(x[:, i]) + tolerance
        RR[0][0] = RR1
        
        # Successively specify subsequent cells according to BTC structure
        for p in range(kappa):
            for q in range(2**p):
                S = SS[p][q]  # Current points
                Y = YS[p][q]  # Current labels/values
                
                if S is None or len(S) == 0:
                    continue
                
                # Find component with largest empirical standard deviation
                std_vals = np.std(S, axis=0)
                l = np.argmax(std_vals)
                
                # Get median for the component with largest std
                med = np.median(S[:, l])
                
                # Sort points and labels/values
                sort_idx = S[:, l].argsort()
                S = S[sort_idx]
                Y = Y[sort_idx]
                
                # Split points into two cells
                t = len(S) // 2
                
                # Update points and labels matrices
                SS[p + 1][2*q] = S[:t]
                SS[p + 1][2*q + 1] = S[t:]
                YS[p + 1][2*q] = Y[:t]
                YS[p + 1][2*q + 1] = Y[t:]
                
                # Update rectangles
                RR[p + 1][2 * q] = np.copy(RR[p][q])
                RR[p + 1][2 * q][l][1] = med
                
                RR[p + 1][2 * q + 1] = np.copy(RR[p][q])
                RR[p + 1][2 * q + 1][l][0] = med
        
        # Last row contains final partition
        Scell = [(SS[-1][i], YS[-1][i]) for i in range(2**kappa)]
        
        return RR, Scell
    
    def _localization(self, RR, point, counter, index, kappa):
        """
        Determine the cell in which a point is located using the binary tree structure.
        
        Parameters:
        -----------
        RR : array-like
            Cell matrix containing rectangles
        point : array-like
            Point to localize
        counter : int
            Current level in the tree
        index : int
            Current index in the level
        kappa : int
            Maximum tree depth
            
        Returns:
        --------
        int
            Index of the cell containing the point
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
    
    def _part_estimator(self, cell_points, cell_values):
        """
        Compute the partitioning estimator.
        For regression: averages values in the cell
        For classification: returns most common class in the cell
        
        Parameters:
        -----------
        cell_points : array-like
            Points in the cell
        cell_values : array-like
            Target values/labels for the points
            
        Returns:
        --------
        float or int
            - For regression: average of values
            - For classification: most common class
        """
        if cell_points is None or len(cell_points) == 0:
            if self.task == 'classification':
                return self.classes_[0]  # Return first class as default
            return 0
        
        if self.task == 'classification':
            # Return most common class in the cell
            unique_classes, counts = np.unique(cell_values, return_counts=True)
            return unique_classes[np.argmax(counts)]
        else:
            # For regression, return mean
            return np.mean(cell_values)
    
    def _part_estimator_proba(self, cell_points, cell_values):
        """
        Compute class probabilities for classification.
        
        Parameters:
        -----------
        cell_points : array-like
            Points in the cell
        cell_values : array-like
            Class labels for the points
            
        Returns:
        --------
        array-like
            Class probabilities
        """
        if cell_points is None or len(cell_points) == 0:
            # Return uniform distribution if cell is empty
            return np.ones(len(self.classes_)) / len(self.classes_)
        
        # Count occurrences of each class
        class_counts = np.zeros(len(self.classes_))
        unique_classes, counts = np.unique(cell_values, return_counts=True)
        for cls, count in zip(unique_classes, counts):
            idx = np.where(self.classes_ == cls)[0][0]
            class_counts[idx] = count
        
        # Normalize to get probabilities
        return class_counts / len(cell_values)
    
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
        X = np.asarray(X)
        y = np.asarray(y)
        
        if self.task == 'classification':
            self.classes_ = np.unique(y)
        
        # Create the data-dependent partition
        self.RR, self.Scell = self._datadeppart(self.kappa, X, y, self.tolerance)
        
        # Store training data
        self.X_train = X
        self.y_train = y
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict using the partitioning estimator.
        
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
        y_pred = np.zeros(len(X))
        
        # For each test point
        for i in range(len(X)):
            # Localize the point to its cell
            cell_idx = self._localization(self.RR, X[i], 0, 0, self.kappa)
            
            # Get points and values in the cell
            cell_points, cell_values = self.Scell[cell_idx]
            
            # Compute prediction
            y_pred[i] = self._part_estimator(cell_points, cell_values)
        
        return y_pred
    
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
        probas = np.zeros((len(X), len(self.classes_)))
        
        # For each test point
        for i in range(len(X)):
            # Localize the point to its cell
            cell_idx = self._localization(self.RR, X[i], 0, 0, self.kappa)
            
            # Get points and values in the cell
            cell_points, cell_values = self.Scell[cell_idx]
            
            # Compute class probabilities
            probas[i] = self._part_estimator_proba(cell_points, cell_values)
        
        return probas 