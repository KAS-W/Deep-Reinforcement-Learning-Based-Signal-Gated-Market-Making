import pandas as pd
import numpy as np
import torch
from feature.tick_engineering import v_impluse
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
        self.vol = torch.tensor(self.df['vol'].values, dtype=torch.float32, device=device)

        self.m = self._compute_pseudo_mid()

    def get_effective_seconds(self, t):
        h, m, s = t // 10000000, (t // 100000) % 100, (t // 1000) % 100
        ts = h * 3600 + m * 60 + s
        return max(0, ts - 34200) if ts <= 41400 else 7200 + max(0, ts - 46800)
    
    def _compute_pseudo_mid(self):
        """Lee-Ready Algo"""
        mid = (self.bid1 + self.ask1) / 2.0
        trade_price = torch.tensor(self.df['trade_price'].values, dtype=torch.float32, device=self.device)
        
        vol_diff = torch.diff(self.vol, prepend=self.vol[:1]) 
        has_trade = (vol_diff > 0).float()

        is_buy = (trade_price > mid).float()
        is_sell = (trade_price < mid).float()

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
            step_indices = change_indices[::7]
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
        
        mid_prices = (self.df['bidprice1'] + self.df['askprice1']) / 2.0
        event_indices = self.df.index[mid_prices.diff() != 0].tolist()
        
        log_mid_events = torch.tensor(np.log(mid_prices.loc[event_indices].values), device=self.device, dtype=torch.float32)
        slope_events = ts_rolling_slope(log_mid_events, k_future)

        if isinstance(slope_events, torch.Tensor):
            slope_events = slope_events.cpu().numpy()

        # shift in event flows
        future_slope_series = pd.Series(slope_events, index=event_indices).shift(-k_future)
        
        dataset = features.loc[event_indices].copy()
        dataset['label_slope'] = future_slope_series
        dataset['trade_time'] = self.df['trade_time'].values[event_indices]

        dataset = dataset.iloc[::8].dropna()

        label_col = 'label_slope'
        X_all = dataset.drop(columns=[label_col, 'trade_time']).values
        y_all = dataset[label_col].values * 1000

        X_3d, y_3d = [], []

        for i in range(time_steps, len(X_all)):
            X_3d.append(X_all[i-time_steps:i])
            y_3d.append(y_all[i])

        return np.array(X_3d), np.array(y_3d)
    
class HFTMarketBase:
    def __init__(self, snap_df: pd.DataFrame, tick_df: pd.DataFrame, device='cpu'):
        self.device = device

        snap_df = snap_df.sort_values(['trade_date', 'trade_time']).reset_index(drop=True).copy(deep=True)

        # calc direction in 1s 
        tick_df = tick_df.sort_values('trade_time').copy(deep=True)
        # agg tick level to snap level
        tick_agg = v_impluse(tick_df)

        self.df = pd.merge_asof(
            snap_df, 
            tick_agg, 
            on='trade_time', 
            direction='backward'
        ).fillna(0)

        self.df = self.df[(self.df['bidprice1'] > 0) & (self.df['askprice1'] > 0)].copy()

        self.norm_time = self.compute_norm_time()

        self.bid1 = torch.tensor(self.df['bidprice1'].values, dtype=torch.float32, device=device)
        self.ask1 = torch.tensor(self.df['askprice1'].values, dtype=torch.float32, device=device)

        # l2 features
        self.trade_flow = torch.tensor(self.df['f_tick_flow'].values, dtype=torch.float32, device=device)
        # self.trade_flow = torch.tensor(self.df['f_flow_imbalance'].values, dtype=torch.float32, device=device)
        # self.f_net_pressure = torch.tensor(self.df['f_net_pressure'].values, dtype=torch.float32, device=device)
        # self.f_sweep_ratio = torch.tensor(self.df['f_sweep_ratio'].values, dtype=torch.float32, device=device)
        # self.f_price_impact = torch.tensor(self.df['f_price_impact'].values, dtype=torch.float32, device=device)

        self.mid = self._compute_pseudo_mid()
        
        self.ask_prices = [torch.tensor(self.df[f'askprice{i}'].values, device=device) for i in range(1, 11)]
        self.ask_vols = [torch.tensor(self.df[f'askvol{i}'].values, device=device) for i in range(1, 11)]
        self.bid_prices = [torch.tensor(self.df[f'bidprice{i}'].values, device=device) for i in range(1, 11)]
        self.bid_vols = [torch.tensor(self.df[f'bidvol{i}'].values, device=device) for i in range(1, 11)]

        # event: mid-price shifts
        self.event_mask = torch.cat([torch.tensor([True], device=device), 
                                     (self.mid[1:] != self.mid[:-1])
                                    ])
    
    def _compute_pseudo_mid(self):
        """Rebuild pseudo mid-price"""
        mid = (self.bid1 + self.ask1) / 2.0

        m_i = torch.where(
            self.trade_flow > 0,
            (mid + self.ask1) / 2.0, 
            torch.where(
                self.trade_flow < 0,
                (mid + self.bid1) / 2.0, 
                mid                     
            )
        )
        return m_i
    
    def compute_norm_time(self):
        t = self.df['trade_time'].values // 1000
        minutes = (t // 10000) * 60 + (t // 100 % 100)
        progress = np.where(
        t <= 113000,
        minutes - (9 * 60 + 30),
        minutes - (9 * 60 + 30) - 90
        )

        norm_time = np.clip(progress / 240.0, 0, 1)
        return torch.tensor(norm_time, dtype=torch.float32, device=self.device)
    
    def feature_engineering(self):
        f = {}
        log_mid = ts_log(self.mid)

        for i in range(5):
            f[f'f_obi_l{i+1}'] = ts_div(ts_sub(self.bid_vols[i], self.ask_vols[i]), 
                                        ts_add(self.bid_vols[i], self.ask_vols[i]))
            f[f'f_spread_l{i+1}'] = ts_sub(self.ask_prices[i], self.bid_prices[i])

        # f['f_net_pressure'] = self.f_net_pressure
        # f['f_sweep_ratio'] = self.f_sweep_ratio
        # f['f_price_impact'] = self.f_price_impact
        f['f_tick_flow'] = self.trade_flow
        # f['f_tick_flow'] = torch.fft.irfft(torch.fft.rfft(self.trade_flow)[:len(self.trade_flow)//20], n=len(self.trade_flow))
        # f['f_tick_flow'] = torch.fft.irfft(torch.fft.rfft(self.trade_flow) * torch.linspace(1, 0, len(torch.fft.rfft(self.trade_flow)), device=self.device)**2, n=len(self.trade_flow))
        
        f['f_log_ret_5'] = ts_sub(log_mid, ts_delay(log_mid, 5))
        f['f_past_rr_10'] = ts_sub(ts_log(ts_rolling_max(self.mid, 10)), 
                                    ts_log(ts_rolling_min(self.mid, 10)))
        
        f['f_norm_time'] = self.norm_time

        features = pd.DataFrame({k: v.cpu().numpy() for k, v in f.items()}, index=self.df.index)

        return features

class SGU1DataPro(HFTMarketBase):
    def compute_labels(self, k_future=10):
        log_mid = ts_log(self.mid)
        r_max = ts_rolling_max(log_mid, k_future)
        r_min = ts_rolling_min(log_mid, k_future)

        rr = ts_sub(r_max, r_min)
        return pd.Series(rr.cpu().numpy()).shift(-k_future)

    def gen_dataset(self, k_future=10, event_step=19):
        features = self.feature_engineering()
        labels = self.compute_labels(k_future).rename('label_rr')

        dataset = pd.concat([features, labels], axis=1)

        event_indices = np.where(self.event_mask.cpu().numpy())[0]
        dataset = dataset.iloc[event_indices]

        dataset = dataset.iloc[::event_step].dropna()
        trade_times = self.df['trade_time'].iloc[dataset.index]
        dataset = dataset[trade_times >= 93000000]

        return dataset.reset_index(drop=True)
    
class SGU2DataPro(HFTMarketBase):
    def compute_labels(self, k_future=10):
        log_mid = ts_log(self.mid)
        slope = ts_rolling_slope(log_mid, k_future)

        if isinstance(slope, torch.Tensor):
            slope = slope.cpu().numpy()

        return pd.Series(slope).shift(-k_future)
    
    def gen_dataset(self, k_future=10, time_steps=10, event_step=19):
        features = self.feature_engineering()
        label_col = f'label_slope_{k_future}'

        event_mask_ts = torch.diff(self.mid, prepend=self.mid[:1]) != 0
        log_mid_events = torch.log(self.mid[event_mask_ts]).to(torch.float32)

        event_mask_np = event_mask_ts.cpu().numpy()
        event_indices = self.df.index[event_mask_np].tolist()

        slope_events = ts_rolling_slope(log_mid_events, k_future)
        if isinstance(slope_events, torch.Tensor):
            slope_events = slope_events.cpu().numpy()

        labels = pd.Series(slope_events, index=event_indices, name=label_col).shift(-k_future)
        dataset = pd.concat([features, labels], axis=1)
        dataset['trade_time'] = self.df['trade_time'].values

        step_indices = event_indices[::event_step]
        dataset = dataset.loc[dataset.index.intersection(step_indices)]

        dataset = dataset[dataset['trade_time'] >= 93000000].dropna()

        X_cols = [c for c in dataset.columns if c not in [label_col, 'trade_time']]
        X = dataset[X_cols].values
        y = dataset[label_col].values

        num_samples = len(X) - time_steps
        if num_samples <= 0:
            return np.array([]), np.array([])
        
        X_3d = np.zeros((num_samples, time_steps, len(X_cols)))
        y_3d = np.zeros(num_samples)
        for i in range(num_samples):
            X_3d[i] = X[i : i + time_steps]
            y_3d[i] = y[i + time_steps - 1]

        return X_3d, y_3d