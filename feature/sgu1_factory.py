import pandas as pd
from utils.fast.oper import ts_log, ts_sub, ts_div
from utils.fast.calculator import (
    ts_rolling_mean, ts_rolling_std, ts_rolling_max, 
    ts_rolling_min, ts_delay
)

class SGU1Features:
    """Produce features for SGU1"""

    def __init__(self, minute_df: pd.DataFrame):
        self._minute_df = minute_df.copy(deep=True)
        self.close = self.self._minute_df['close']
        self.vol = self.self._minute_df['vol']
        self.amount = self.self._minute_df['amount']

    def compute_all_features(self):
        features = {}

        # 1. log returns: f1-f5
        log_p = ts_log(self.close)
        # we take k = 1, 5, 10, 30, 60
        for k in [1, 5, 10, 30, 60]:
            features[f'f_log_ret_{k}'] = ts_sub(log_p, ts_delay(log_p, k))

        # 2. rolling volatility: f6-f10
        ret_1m = ts_sub(log_p, ts_delay(log_p, 1))
        for k in [5, 10, 30, 60]:
            features[f'f_volat_{k}'] = ts_rolling_std(ret_1m, k)

        # 3. relative price position: f11-f15
        for k in [5, 10, 30, 60]:
            r_max = ts_rolling_max(self.close, k)
            r_min = ts_rolling_min(self.close, k)
            # LP = (p - min) / (max - min)
            num = ts_sub(self.close, r_min)
            den = ts_sub(r_max, r_min)
            features[f'f_price_pos_{k}'] = ts_div(num, den)

        # 4. relative volume: f16
        features['f_rel_vol'] = ts_div(self.vol, ts_rolling_mean(self.vol, 5))

        # domain specific
        features['f_time_idx'] = self._get_normalized_time()
        # lagged labels
        raw_rr = ts_sub(ts_log(self.df['high']), ts_log(self.df['low']))
        for L in range(1, 6):
            features[f'f_lag_label_{L}'] = ts_delay(raw_rr, L)

        return pd.DataFrame(features, index=self.df.index)
    
    def _get_normalized_time(self):
        """Convert YYYYMMDDHHMMSS to minutes since market open"""
        time_series = pd.to_datetime(self.df['trade_time'], format='%Y%m%d%H%M%S')
        hours = time_series.dt.hour
        minutes = time_series.dt.minute

        # Map A-share sessions to 0-240 minutes
        # This is a specific logic not suitable for general calculator.py
        total_minutes = (hours - 9) * 60 + minutes - 30
        # Adjust for lunch break (11:30 - 13:00)
        total_minutes = total_minutes.where(hours < 13, total_minutes - 90)
        return total_minutes / 240.0