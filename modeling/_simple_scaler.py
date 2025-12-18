import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SimpleStandardScaler(BaseEstimator, TransformerMixin):
    """
    A simplified re-implementation of StandardScaler.

    Computes column-wise mean and standard deviation during fit,
    and standardizes features during transform.
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_
