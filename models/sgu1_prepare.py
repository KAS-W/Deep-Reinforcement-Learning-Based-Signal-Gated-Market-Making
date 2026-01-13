import pandas as pd
import numpy as np
import torch
from utils.fast.oper import ts_log, ts_sub, ts_div, ts_add
from utils.fast.calculator import (
    ts_rolling_std, ts_rolling_max, 
    ts_rolling_min, ts_delay)

class SGU1_dataset:
    """
    Generate features and labels for SGU1
    """

    def __init__(self, snap:pd.DataFrame, device='cpu'):
        self.df = snap.sort_values(['trade_date', 'trade_time']).reset_index(drop=True).copy()
        self.device = device
        # filter out zero price, since no trades/quotes available
        self.df = self.df[(self.df['bidprice1'] > 0) & (self.df['askprice1'] > 0)].copy()

        times = self.df['trade_time'].apply(self.get_seconds).values
        self.norm_time = torch.tensor((times - times.min()) / (times.max() - times.min()),
                                       dtype=torch.float32, device=device)

        # etf price should div by 10000 and turn to Tensor
        self.close = torch.tensor(self.df['trade_price'].values, dtype=torch.float32, device=device)
        self.bid1 = torch.tensor(self.df['bidprice1'].values, dtype=torch.float32, device=device)
        self.ask1 = torch.tensor(self.df['askprice1'].values, dtype=torch.float32, device=device)

        # mid price
        self.mid = (self.bid1 + self.ask1) / 2.0

        # level 1 volumes
        self.bidvol1 = torch.tensor(self.df['bidvol1'].values, dtype=torch.float32, device=device)
        self.askvol1 = torch.tensor(self.df['askvol1'].values, dtype=torch.float32, device=device)

    def compute_features(self):
        f = {}
        log_mid = ts_log(self.mid)

        # log returns: k=1, 2, 3, 5, 10
        for k in [1, 2, 3, 5, 10]:
            f[f'f_log_ret_{k}'] = ts_sub(log_mid, ts_delay(log_mid, k))

        # volatility
        ret_1t = ts_sub(log_mid, ts_delay(log_mid, 1))
        for k in [10, 20, 30, 50, 100]:
            f[f'f_vol_{k}t'] = ts_rolling_std(ret_1t, k)
        
        # relative price positions
        for k in [10, 20, 30, 50, 100]:
            r_max = ts_rolling_max(self.mid, k)
            r_min = ts_rolling_min(self.mid, k)
            f[f'f_pos_{k}t'] = ts_div(ts_sub(self.mid, r_min), ts_sub(r_max, r_min))

        # realized ranges / lagged labels
        # first we calculate the past RR over a window of 10
        log_r_max_10 = ts_log(ts_rolling_max(self.mid, 10))
        log_r_min_10 = ts_log(ts_rolling_min(self.mid, 10))
        past_rr_10 = ts_sub(log_r_max_10, log_r_min_10)
        for L in [1, 2, 3, 4, 5]:
            f[f'f_lag_rr_{L}'] = ts_delay(past_rr_10, L)
        
        # orderbook imbalance
        f['f_obi_l1'] = ts_div(ts_sub(self.bidvol1, self.askvol1),
                               ts_add(self.bidvol1, self.askvol1))
        
        # bid-ask spread
        f['f_spread_1'] = ts_sub(self.ask1, self.bid1)

        # normalized time
        f['f_norm_time'] = self.norm_time

        
        return f
    
    def compute_labels(self, k_future=10):
        """
        Label RR_t,10: Log Realized Range of the NEXT 10 ticks.

        RR_t,k = ln(max(P_{t+1...t+k}) / min(P_{t+1...t+k}))
        """
        # Calculate backward rolling max/min for window 10
        log_mid = ts_log(self.mid)
        r_max = ts_rolling_max(log_mid, k_future)
        r_min = ts_rolling_min(log_mid, k_future)

        # Shift backward by 10 to align future max/min with current time t
        future_max = pd.Series(r_max.cpu().numpy()).shift(-k_future)
        future_min = pd.Series(r_min.cpu().numpy()).shift(-k_future)

        # Log Range: ln(max) - ln(min)
        label_rr = future_max - future_min
        return pd.Series(label_rr, name=f'label_rr_{k_future}')
    
    def gen_task_dataset(self, use_event_sampling=True):
        """Start at 09:30:00.000"""
        f_dict = self.compute_features()
        features = pd.DataFrame({k: v.cpu().numpy() for k, v in f_dict.items()}, index=self.df.index)
        labels = self.compute_labels()

        dataset = pd.concat([features, labels], axis=1)
        dataset['trade_time'] = self.df['trade_time'].values
        dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataset = dataset.dropna()

        if use_event_sampling:
            # per 19-changes as a bucket
            mid_prices = (self.df['bidprice1'] + self.df['askprice1']) / 2.0
            change_indices = self.df.index[mid_prices.diff() != 0].tolist()
            step_indices = change_indices[::19]
            dataset = dataset.loc[dataset.index.intersection(step_indices)]

        dataset = dataset[dataset['trade_time'] >= 93000000]

        return dataset.drop(columns=['trade_time']).reset_index(drop=True)
    
    def get_seconds(self, t):
        h = t // 10000000
        m = (t // 100000) % 100
        s = (t // 1000) % 100
        return h * 3600 + m * 60 + s