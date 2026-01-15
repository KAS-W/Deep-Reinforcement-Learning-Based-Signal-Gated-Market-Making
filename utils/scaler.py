from sklearn.preprocessing import QuantileTransformer
import numpy as np

class StandardScaler3D:
    def __init__(self, threshold=3.0):
        self.mean = None
        self.std = None
        self.threshold = threshold

    def fit(self, X):
        self.median = np.nanmedian(X, axis=(0, 1), keepdims=True)
        abs_deviation = np.abs(X - self.median)
        self.mad = np.nanmedian(abs_deviation, axis=(0, 1), keepdims=True) + 1e-9
        self.mad = self.mad * 1.4826
        # self.mean = np.mean(X, axis=(0, 1), keepdims=True)
        # self.std = np.std(X, axis=(0, 1), keepdims=True) + 1e-9

    def transform(self, X):
        if self.median is None or self.mad is None:
            raise ValueError("Scaler has not been fitted yet.")
        X_scaled = (X - self.median) / self.mad
        # return (X - self.mean) / self.std
        return np.clip(X_scaled, -self.threshold, self.threshold)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
class FeatureProcesser:
    def __init__(self, n_quantiles=1000, output_distribution='normal'):
        self.qt = QuantileTransformer(
            n_quantiles=n_quantiles, 
            output_distribution=output_distribution,
            subsample=100000, 
            random_state=42
        )

    def fit(self, X_3d):
        N, T, D = X_3d.shape
        X_2d = X_3d.reshape(-1, D)
        self.qt.fit(X_2d)
        return self
    
    def transform(self, X_3d):
        N, T, D = X_3d.shape
        X_2d = X_3d.reshape(-1, D)
        X_transformed_2d = self.qt.transform(X_2d)
        return X_transformed_2d.reshape(N, T, D)

    def fit_transform(self, X_3d):
        return self.fit(X_3d).transform(X_3d)
    
class RobustWinsorScaler3D:
    def __init__(self, limits=(0.005, 0.995)):
        self.limits = limits
        self.mean = None
        self.std = None
        self.lower_bound = None
        self.upper_bound = None

    def fit(self, X):
        X_flat = X.reshape(-1, X.shape[-1])
        
        self.lower_bound = np.percentile(X_flat, self.limits[0] * 100, axis=0)
        self.upper_bound = np.percentile(X_flat, self.limits[1] * 100, axis=0)
        
        X_clipped = np.clip(X_flat, self.lower_bound, self.upper_bound)
        self.mean = np.mean(X_clipped, axis=0, keepdims=True)
        self.std = np.std(X_clipped, axis=0, keepdims=True) + 1e-9

    def transform(self, X):
        N, T, D = X.shape
        X_flat = X.reshape(-1, D)
        X_clipped = np.clip(X_flat, self.lower_bound, self.upper_bound)
        X_scaled = (X_clipped - self.mean) / self.std
        return X_scaled.reshape(N, T, D)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)