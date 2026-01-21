import numpy as np

class StandardScaler3D:
    def __init__(self, threshold=3.0):
        self.mean = None
        self.std = None
        self.threshold = threshold

    def fit(self, X):
        self.mean = np.mean(X, axis=(0, 1), keepdims=True)
        self.std = np.std(X, axis=(0, 1), keepdims=True) + 1e-9

    def transform(self, X):
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")
        X_scaled = (X - self.mean) / self.std
        return X_scaled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)