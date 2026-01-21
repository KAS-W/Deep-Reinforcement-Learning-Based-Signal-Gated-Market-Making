import numpy as np

class FTPEnv:
    """
    Match-making engine by FTP
    """

    def __init__(self, phi=0.01, tick_size=0.01, fee_rate=0.0000):
        self.fee_rate = fee_rate
        self.phi = phi 
        self.tick_size = tick_size
        self.inventory = 0
        self.cash = 0.0
        self.i_max = 2
        self.i_min = -2

    def reset(self):
        self.inventory = 0
        self.cash = 0.0
        return self.inventory, self.cash
    
    def step(self, action, mid_next, best_ask, best_bid, buy_max, sell_min, adv_action=None):
        off_a, off_b = action[0], action[1]

        if adv_action is not None:
            delta_a, delta_b = np.round(adv_action).astype(int)
            off_a += delta_a
            off_b += delta_b

        my_ask = best_ask + off_a * self.tick_size
        my_bid = best_bid - off_b * self.tick_size

        # check inventory
        can_buy = self.inventory < self.i_max
        can_sell = self.inventory > self.i_min

        fill_buy = 1 if (can_buy and my_bid >= sell_min) else 0
        fill_sell = 1 if (can_sell and my_ask <= buy_max) else 0
        
        pnl_reward = 0.0
        fee_paid = 0.0

        # match successful
        if fill_buy:
            self.inventory += 1
            current_fee = my_bid * self.fee_rate
            self.cash -= (my_bid + current_fee)
            pnl_reward += (mid_next - my_bid) - current_fee
            fee_paid += current_fee
        if fill_sell:
            self.inventory -= 1
            current_fee = my_ask * self.fee_rate
            self.cash += (my_ask - current_fee)
            pnl_reward += (my_ask - mid_next) - current_fee
            fee_paid += current_fee

        inv_penalty = self.phi * abs(self.inventory)
        reward = pnl_reward - inv_penalty

        info = {
            'pnl_reward': pnl_reward, 
            'inventory_reward': -inv_penalty,
            'fee_paid': fee_paid,
            'fill_buy': fill_buy, 
            'fill_sell': fill_sell
        }
        return reward, info