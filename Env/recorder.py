import pandas as pd

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

    def to_dataframe(self):
        columns = ['step', 'mid', 'ask', 'bid', 'off_a', 'off_b', 'reward', 'inventory', 'cash', 'pnl_reward', 'inventory_reward', 'fee_paid', 'is_trade']
        df = pd.DataFrame(self.data, columns=columns)

        df['spread'] = df['ask'] - df['bid']
        df['wealth'] = df['cash'] + df['inventory'] * df['mid']
        df['cum_reward'] = df['reward'].cumsum()
        df['skew'] = df['off_b'] - df['off_a']
        df['cum_fees'] = df['fee_paid'].cumsum()
        df['realized_pnl'] = df['cash']
        df['unrealized_pnl'] = df['inventory'] * df['mid']

        return df