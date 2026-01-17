import pandas as pd
import numpy as np
import torch
from utils.fast.calculator import ts_delay

class HFTMarketBase:
    def __init__(self, snap_df: pd.DataFrame, tick_df: pd.DataFrame, device='cpu'):
        self.device = device
    
        self.snap = snap_df.sort_values('trade_time').reset_index(drop=True)
        self.tick = tick_df.sort_values('trade_time').reset_index(drop=True)

        self.event_mask = (self.snap['bidprice1'].diff() != 0) | \
                          (self.snap['askprice1'].diff() != 0) | \
                          (self.snap['bidvol1'].diff() != 0) | \
                          (self.snap['askvol1'].diff() != 0)
        self.event_df = self.snap[self.event_mask].copy()

        t = self.tick
       
        t['p_buy'] = np.where(t['side'] == 1, t['Price'], np.nan)
        t['p_sell'] = np.where(t['side'] == -1, t['Price'], np.nan)
        t['v_buy'] = np.where(t['side'] == 1, t['Volume'], 0)
        t['v_sell'] = np.where(t['side'] == -1, t['Volume'], 0)
        t['vwap_p'] = t['Price'] * t['Volume']

        tick_stats = t.groupby('trade_time').agg({
            'p_buy': 'max',       
            'p_sell': 'min',     
            'v_buy': 'sum',      
            'v_sell': 'sum',      
            'Volume': 'sum',      
            'Price': 'count',     
            'vwap_p': 'sum'       
        }).reset_index()

        tick_stats.columns = ['trade_time', 'p_buy_max', 'p_sell_min', 
                              'v_buy_sum', 'v_sell_sum', 'vol_sum', 
                              'trade_count', 'vwap_num']
        
        self.event_df = pd.merge_asof(self.event_df, tick_stats, on='trade_time', direction='backward')

        self.tick.drop(columns=['p_buy', 'p_sell', 'v_buy', 'v_sell', 'vwap_p'], inplace=True)

class SGU1DataPro(HFTMarketBase):
    def gen_dataset(self, delta_t=19):
        delta_t = 19
        periods = []
        for i in range(0, len(self.event_df) - delta_t, delta_t):
            group = self.event_df.iloc[i : i + delta_t]
            
            p_buy_max = group['p_buy_max'].max()
            p_sell_min = group['p_sell_min'].min()
            ask_mean = group['askprice1'].mean()
            bid_mean = group['bidprice1'].mean()

            if not np.isnan(p_buy_max) and not np.isnan(p_sell_min):
                label = p_buy_max - p_sell_min
            elif np.isnan(p_buy_max) and not np.isnan(p_sell_min):
                label = ask_mean - p_sell_min
            elif not np.isnan(p_buy_max) and np.isnan(p_sell_min):
                label = p_buy_max - bid_mean
            else:
                label = ask_mean - bid_mean

            periods.append({
                'label': round(label, 4),
                'tr_count': group['trade_count'].sum(), #
                'vwap': group['vwap_num'].sum() / (group['vol_sum'].sum() + 1e-9), #
                'mid': (group['askprice1'].iloc[-1] + group['bidprice1'].iloc[-1]) / 2.0,
                'vol_imb': (group['v_buy_sum'].sum() - group['v_sell_sum'].sum()) / (group['vol_sum'].sum() + 1e-9),
                'spread': group['askprice1'].iloc[0] - group['bidprice1'].iloc[0],
                'tot_vol': group['vol_sum'].sum(),
                'hour': group['trade_time'].iloc[0] // 10000000
            })
            
        p_df = pd.DataFrame(periods)
        f = pd.DataFrame()

        for p in [1, 2, 3, 5, 10]:
            f[f'f_trades_{p}'] = p_df['tr_count'].rolling(p, min_periods=1).sum().shift(1)
                    
        for r in [1, 3, 5]:
            f[f'f_vwap_{r}'] = p_df['vwap'].rolling(r, min_periods=1).mean().shift(1)
                    
        def get_slope(y):
            if len(y) < 2: return 0.0
            x = np.arange(len(y))
            return np.polyfit(x, y, 1)[0]

        for s in [1, 3, 5]:
            f[f'f_slope_{s}'] = p_df['mid'].rolling(s, min_periods=1).apply(get_slope, raw=True).shift(1)
                    
        for L in [1, 2, 3, 4, 5]:
            f[f'f_lag_rr_{L}'] = p_df['label'].shift(L)

        f['f_spread'] = p_df['spread'].shift(1)
        f['f_vol_imb'] = p_df['vol_imb'].shift(1)
        f['f_total_vol'] = p_df['tot_vol'].shift(1)
        f['f_hour'] = p_df['hour']
        f['label'] = p_df['label']    

        final = f.ffill().bfill().dropna().reset_index(drop=True)

        return final
    
class SGU2DataPro(HFTMarketBase):
    def gen_dataset(self, event_step=19, time_steps=10):
        m_list = []
        for i in range(0, len(self.event_df) - event_step, event_step):
            group = self.event_df.iloc[i : i + event_step]

            buy_max, sell_min = group['p_buy_max'].max(), group['p_sell_min'].min()
            a_mean, b_mean = group['askprice1'].mean(), group['bidprice1'].mean()

            if not np.isnan(buy_max) and not np.isnan(sell_min):
                m = 0.5 * (buy_max + sell_min)
            elif np.isnan(buy_max) and not np.isnan(sell_min):
                m = 0.5 * (a_mean + sell_min)
            elif not np.isnan(buy_max) and np.isnan(sell_min):
                m = 0.5 * (buy_max + b_mean)
            else:
                m = 0.5 * (a_mean + b_mean)
            m_list.append(m)

        if len(m_list) <= time_steps + 1: 
            return np.array([]), np.array([])

        m_series = pd.Series(m_list).ffill() 
        m_tensor = torch.tensor(m_series.values, dtype=torch.float32, device=self.device)
        m_tensor[m_tensor < 1e-5] = 1e-5

        m_lag = ts_delay(m_tensor, 1)

        returns = (m_tensor - m_lag) / (m_lag + 1e-9)
        ret_np = returns.cpu().numpy()

        ret_np = ret_np[1:]

        X, y = [], []
        for t in range(time_steps, len(ret_np)):
            X.append(ret_np[t - time_steps : t].reshape(-1, 1))
            y.append(ret_np[t])

        X, y = np.array(X), np.array(y)
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)

        return X, y