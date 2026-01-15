import pandas as pd
import numpy as np
import torch
from utils.fast.oper import ts_log, ts_sub, ts_div, ts_add
from utils.fast.calculator import (
    ts_rolling_mean, ts_rolling_std, ts_rolling_max, 
    ts_rolling_min, ts_delay, ts_rolling_slope)

class MarketDatasetBase:
    """
    Feature engine
    """

    def __init__(self, snap: pd.DataFrame, device='cpu'):
        # sort by time
        self.df = snap.sort_values(['trade_date', 'trade_time']).reset_index(drop=True).copy()
        self.device = device
        # filter out zero-price data
        self.df = self.df[(self.df['bidprice1'] > 0) & (self.df['askprice1'] > 0)].copy()
        # norm time
        raw_times = self.df['trade_time'].values
        market_secs = np.array([self.get_effective_seconds(t) for t in raw_times])
        self.norm_time = torch.tensor(market_secs / 14400.0, dtype=torch.float32, device=device)
        self.hour_time = torch.tensor((raw_times // 10000000), dtype=torch.float32, device=device)
        
        self.close = torch.tensor(self.df['trade_price'].values, dtype=torch.float32, device=device)
        self.bid1 = torch.tensor(self.df['bidprice1'].values, dtype=torch.float32, device=device)
        self.ask1 = torch.tensor(self.df['askprice1'].values, dtype=torch.float32, device=device)
        self.mid = (self.bid1 + self.ask1) / 2.0
        self.bidvol1 = torch.tensor(self.df['bidvol1'].values, dtype=torch.float32, device=device)
        self.askvol1 = torch.tensor(self.df['askvol1'].values, dtype=torch.float32, device=device)

        self.m = self._compute_pseudo_mid()

    def get_effective_seconds(self, t):
        h, m, s = t // 10000000, (t // 100000) % 100, (t // 1000) % 100
        ts = h * 3600 + m * 60 + s
        return max(0, ts - 34200) if ts <= 41400 else 7200 + max(0, ts - 46800)
    
    def _compute_pseudo_mid(self):
        """Lee-Ready Algo"""
        mid = (self.bid1 + self.ask1) / 2.0
        trade_price = torch.tensor(self.df['trade_price'].values, dtype=torch.float32, device=self.device)
        is_buy = (trade_price > mid).float()
        is_sell = (trade_price < mid).float()
        has_trade = (self.vol.diff() > 0).float()

        p_buy_max = trade_price * is_buy
        p_sell_min = trade_price * is_sell

        m_i = torch.where(
            (is_buy > 0) & (is_sell == 0),
            (p_buy_max + self.bid1) / 2.0,
            torch.where(
                (is_sell > 0) & (is_buy == 0),
                (self.ask1 + p_sell_min) / 2.0,
                mid
            )
        )

        m_i = torch.where(has_trade > 0, m_i, mid)

        return m_i
    
    def _compute_features(self):
        f = {}
        log_mid = ts_log(self.mid)

        # log returns
        for k in [1, 2, 3, 5, 10]:
            f[f'f_log_ret_{k}'] = ts_sub(log_mid, ts_delay(log_mid, k))

        # volatility
        ret_1t = ts_sub(log_mid, ts_delay(log_mid, 1))
        for k in [10, 20, 30, 50, 100]:
            f[f'f_vol_{k}t'] = (ts_rolling_std(ret_1t, k))
            # f[f'f_vol_{k}t'] = ts_log(ts_rolling_std(ret_1t, k))

        # relative price position
        for k in [10, 20, 30, 50, 100]:
            r_max = ts_rolling_max(self.mid, k)
            r_min = ts_rolling_min(self.mid, k)
            f[f'f_pos_{k}t'] = ts_div(ts_sub(self.mid, r_min), ts_sub(r_max, r_min))

        # realized ranges
        log_r_max_10 = ts_log(ts_rolling_max(self.mid, 10))
        log_r_min_10 = ts_log(ts_rolling_min(self.mid, 10))
        past_rr_10 = ts_sub(log_r_max_10, log_r_min_10)
        for L in [1, 2, 3, 4, 5]:
            f[f'f_lag_rr_{L}'] = ts_delay(past_rr_10, L)
            # f[f'f_lag_rr_{L}'] = ts_log(ts_delay(past_rr_10, L))

        # obi
        f['f_obi_l1'] = ts_div(ts_sub(self.bidvol1, self.askvol1), ts_add(self.bidvol1, self.askvol1))
        # l-1 spread
        f['f_spread_1'] = ts_sub(self.ask1, self.bid1)
        # f['f_spread_1'] = ts_log(ts_sub(self.ask1, self.bid1))

        # window = 10
        # for k in list(f.keys()):
        #     if k == 'f_norm_time': 
        #         continue

        #     f_mean = ts_rolling_mean(f[k], window)
        #     f_std = ts_rolling_std(f[k], window) + 1e-9

        #     f[k] = ts_div(ts_sub(f[k], f_mean), f_std)

        # for k in list(f.keys()):
        #     if k == 'f_norm_time': 
        #         continue
        #     f_val = f[k]
        #     f_val = torch.nan_to_num(f_val, nan=0.0)
        #     f[k] = torch.tanh(f_val / (f_val.std() + 1e-9))

        # norm time
        f['f_norm_time'] = self.norm_time
        

        return f
    
class SGU1Dataset(MarketDatasetBase):
    """
    Using XGBoost to predict the RR in next 10 steps
    """

    def compute_labels(self, k_future=10):
        log_mid = ts_log(self.mid)
        r_max = ts_rolling_max(log_mid, k_future)
        r_min = ts_rolling_min(log_mid, k_future)

        future_max = pd.Series(r_max.cpu().numpy()).shift(-k_future)
        future_min = pd.Series(r_min.cpu().numpy()).shift(-k_future)
        return pd.Series(future_max - future_min, name=f'label_rr_{k_future}')
    
    def gen_dataset(self, use_event_sampling=True):
        f_dict = self._compute_features()
        features = pd.DataFrame({k: v.cpu().numpy() for k, v in f_dict.items()}, index=self.df.index)
        labels = self.compute_labels()

        dataset = pd.concat([features, labels], axis=1)
        dataset['trade_time'] = self.df['trade_time'].values
        dataset = dataset.dropna().replace([np.inf, -np.inf], np.nan).dropna()

        if use_event_sampling:
            # 19-event sampling
            mid_prices = (self.df['bidprice1'] + self.df['askprice1']) / 2.0
            change_indices = self.df.index[mid_prices.diff() != 0].tolist()
            step_indices = change_indices[::19]
            dataset = dataset.loc[dataset.index.intersection(step_indices)]

        # here we keep data in continuous auction phase
        dataset = dataset[dataset['trade_time'] >= 93000000]
        return dataset.drop(columns=['trade_time']).reset_index(drop=True)
    
class SGU2Dataset(MarketDatasetBase):
    """
    Predict price slope in next k steps
    """

    def compute_labels(self, k_future=10):
        log_mid = ts_log(self.mid)
        slope = ts_rolling_slope(log_mid, k_future)

        if isinstance(slope, torch.Tensor):
            slope = slope.cpu().numpy()

        future_slope = pd.Series(slope).shift(-k_future)
        return pd.Series(future_slope, name=f'label_slope_{k_future}')
    
    def gen_dataset(self, time_steps=50, k_future=10):
        f_dict = self._compute_features()
        features = pd.DataFrame({k: v.cpu().numpy() for k, v in f_dict.items()}, index=self.df.index)
        label_col = f'label_slope_{k_future}'
        labels = self.compute_labels(k_future=k_future)

        dataset = pd.concat([features, labels], axis=1)
        dataset['trade_time'] = self.df['trade_time'].values
        dataset['vol'] = self.df['vol'].values
        dataset = dataset[dataset['trade_time'] >= 93000000].dropna()
        # dataset['trade_time'] = self.df['trade_time'].values
        # dataset = dataset.dropna()

        # event-19 sampling
        mid_prices = (self.df['bidprice1'] + self.df['askprice1']) / 2.0
        change_indices = self.df.index[mid_prices.diff() != 0].tolist()
        step_indices = change_indices[::19]

        # vol-bucket sampling
        # cum_vol = dataset['vol'].cumsum()
        # bar_counts = cum_vol // 5000
        # step_indices = dataset.index[bar_counts.diff() > 0].tolist()
      
        dataset = dataset.loc[dataset.index.intersection(step_indices)]
        dataset = dataset[dataset['trade_time'] >= 93000000]

        X = dataset.drop(columns=[label_col, 'trade_time']).values
        y = dataset[f'label_slope_{k_future}'].values * 1000

        X_3d, y_3d = [], []
        for i in range(time_steps, len(X)):
            X_3d.append(X[i-time_steps:i])
            y_3d.append(y[i])

        return np.array(X_3d), np.array(y_3d)
    
# class StandardScaler3D:
#     def __init__(self):
#         self.mean = None
#         self.std = None

#     def fit(self, X):

#         self.mean = np.mean(X, axis=(0, 1), keepdims=True)
#         self.std = np.std(X, axis=(0, 1), keepdims=True) + 1e-9

#     def transform(self, X):
#         if self.mean is None or self.std is None:
#             raise ValueError("Scaler has not been fitted yet.")
#         return (X - self.mean) / self.std

#     def fit_transform(self, X):
#         self.fit(X)
#         return self.transform(X)
    
# class FeatureProcesser:
#     def __init__(self, n_quantiles=1000, output_distribution='normal'):
#         self.qt = QuantileTransformer(
#             n_quantiles=n_quantiles, 
#             output_distribution=output_distribution,
#             subsample=100000, 
#             random_state=42
#         )

#     def fit(self, X_3d):
#         N, T, D = X_3d.shape
#         X_2d = X_3d.reshape(-1, D)
#         self.qt.fit(X_2d)
#         return self
    
#     def transform(self, X_3d):
#         N, T, D = X_3d.shape
#         X_2d = X_3d.reshape(-1, D)
#         X_transformed_2d = self.qt.transform(X_2d)
#         return X_transformed_2d.reshape(N, T, D)

#     def fit_transform(self, X_3d):
#         return self.fit(X_3d).transform(X_3d)