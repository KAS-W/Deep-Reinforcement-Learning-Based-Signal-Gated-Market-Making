import os
import torch
import numpy as np
import pandas as pd
from Env.market_env import FTPEnv
from Env.recorder import StrategyRecorder
from models.model import TradingPolicy

def run_drl_backtest(symbol, method_name, weight_path, bundle, phi, fee_rate, train_stats):
    s1, s2, mid_next, best_ask, best_bid, buy_max, sell_min = bundle

    policy = TradingPolicy()
    if os.path.exists(weight_path):
        policy.load_state_dict(torch.load(weight_path, weights_only=True))
        policy.eval()
    else:
        raise FileNotFoundError(f"Weight not found at {weight_path}")
    
    env = FTPEnv(phi=phi, tick_size=0.001, fee_rate=fee_rate)
    recorder = StrategyRecorder()

    save_dir = f"output/{symbol}/{method_name}"
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for t in range(len(mid_next)):
            state = torch.tensor([[(s1[t]-train_stats['s1_m'])/train_stats['s1_s'], 
                                   (s2[t]-train_stats['s2_m'])/train_stats['s2_s'], 
                                   env.inventory/2.0]], dtype=torch.float32)
            # inference
            raw_act = policy.forward(state).squeeze().cpu().numpy()
            action = np.round(raw_act * 5.0).astype(int)

            reward, info = env.step(action, mid_next[t], best_ask[t], best_bid[t], buy_max[t], sell_min[t])

            recorder.record_detailed(t, mid_next[t], best_ask[t], best_bid[t], action, 
                                     reward, env.inventory, env.cash, info, s1[t], s2[t])
            
    df = recorder.to_dataframe()
    save_path = f"{save_dir}/backtest_{phi}.parquet"
    df.to_parquet(save_path, index=False)
    print(f"DRL ({method_name}) backtest saved to: {save_path}")