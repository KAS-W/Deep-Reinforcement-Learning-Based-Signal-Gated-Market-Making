import pandas as pd
import numpy as np

class StrategyRecorder:
    def __init__(self):
        self.data = []

    def record(self, step, mid, ask, bid, action, reward, inv, cash, info):
        self.data.append([
            step, mid, ask, bid, 
            action[0], action[1], 
            reward, inv, cash, 
            info['pnl_reward'], info['inventory_reward'],
            info.get('fee_paid', 0.0),
            (info['fill_buy'] or info['fill_sell'])
        ])

    def record_detailed(self, step, mid, ask, bid, action, reward, inv, cash, info, s1, s2):
        self.data.append({
            'step': step,
            'mid': mid,
            'best_ask': ask,
            'best_bid': bid,
            'off_a': action[0],
            'off_b': action[1],
            'reward': reward,
            'inventory': inv,
            'cash': cash,
            'pnl_reward': info['pnl_reward'],
            'inventory_reward': info['inventory_reward'],
            'fee_paid': info.get('fee_paid', 0.0),
            'fill_buy': info['fill_buy'],
            'fill_sell': info['fill_sell'],
            's1_pred': s1, 
            's2_pred': s2 
        })

    def to_dataframe(self):
        if len(self.data) > 0 and isinstance(self.data[0], list):
            columns = ['step', 'mid', 'ask', 'bid', 'off_a', 'off_b', 'reward', 'inventory', 'cash', 'pnl_reward', 'inventory_reward', 'fee_paid', 'is_trade']
            df = pd.DataFrame(self.data, columns=columns)
        else:
            df = pd.DataFrame(self.data)

        df['spread'] = df['ask'] - df['bid']
        df['wealth'] = df['cash'] + df['inventory'] * df['mid']
        df['cum_reward'] = df['reward'].cumsum()
        df['skew'] = df['off_b'] - df['off_a']
        df['cum_fees'] = df['fee_paid'].cumsum()
        df['realized_pnl'] = df['cash']
        df['unrealized_pnl'] = df['inventory'] * df['mid']

        return df