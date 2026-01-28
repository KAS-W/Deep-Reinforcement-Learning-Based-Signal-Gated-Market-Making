# tutorial for running the DRL pipeline

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
import pickle
import shutil
from Env.market_env import FTPEnv
from Env.recorder import StrategyRecorder
from models.model import TradingPolicy
from Env.benchmarks import FOICPolicy, GLFTPolicy
from models.GateUnits import SGU1, SGU2
from pipeline.agent_trainer import load_signals_bundle
from pipeline.sgu_trainer import run_sgu_training
from pipeline.agent_trainer import run_agent_training_pipeline


# set up parameters
symbol = '510300'
train_range = (20240401, 20240528)
val_range = (20240529, 20240612)
sgu_train_range = (20240401, 20240528)
event_step = 19
time_steps = 10
phi_val = 0.0001
base_path = os.path.abspath("checkpoints") 
checkpoint_dir = os.path.join(base_path, symbol)


# train sgu1 and sgu2
run_sgu_training(symbol=symbol, train_range=train_range, val_range=val_range, event_step=event_step, time_steps=time_steps)

# train DRL agent
run_agent_training_pipeline(symbol='510300', sgu_train_range=(20240401, 20240528), PHI=0.0001, TICK_SIZE=0.001, fee_rate=0.0000, USE_FEE=False, USE_ARL=False, nb_mode=False)

# train ARL agent
run_agent_training_pipeline(symbol='510300', sgu_train_range=(20240401, 20240528), PHI=0.0001, TICK_SIZE=0.001, fee_rate=0.0000, USE_FEE=False, USE_ARL=True, nb_mode=False)

# train DRL with transaction costs
run_agent_training_pipeline(symbol='510300', sgu_train_range=(20240401, 20240528), PHI=0.0001, TICK_SIZE=0.001, fee_rate=0.0003, USE_FEE=True, USE_ARL=False, nb_mode=False)


# run backtest on test sets, using stored agent models
def run_drl_backtest(symbol, method_name, weight_path, bundle, phi, fee_rate, train_stats):
    s1, s2, mid, ask, bid, b_max, s_min = bundle 

    policy = TradingPolicy()
    if os.path.exists(weight_path):
        policy.load_state_dict(torch.load(weight_path, weights_only=True))
        policy.eval()
    else:
        print(f"Warning: Weights not found at {weight_path}")
        return None
    
    env = FTPEnv(phi=phi, tick_size=0.001, fee_rate=fee_rate) 
    recorder = StrategyRecorder()

    with torch.no_grad():
        for t in range(len(mid)):
            n_s = torch.tensor([[(s1[t]-train_stats['s1_m'])/train_stats['s1_s'], 
                                 (s2[t]-train_stats['s2_m'])/train_stats['s2_s'], 
                                 env.inventory/2.0]], dtype=torch.float32)
            
            raw_act = policy.forward(n_s).squeeze().cpu().numpy()
            scaled_act = np.round(raw_act * 5.0).astype(int) 

            reward, info = env.step(scaled_act, mid[t], ask[t], bid[t], b_max[t], s_min[t])

            record_data = {
                'step': t,
                'mid': mid[t],
                'ask': ask[t],
                'bid': bid[t],
                'off_a': scaled_act[0], 
                'off_b': scaled_act[1], 
                'action': scaled_act,
                'reward': reward,
                'inventory': env.inventory,
                'cash': env.cash,
                'fee_paid': info.get('fee_paid', 0.0), 
                's1_pred': s1[t],
                's2_pred': s2[t]
            }
            record_data.update(info)
            recorder.data.append(record_data)
            
    df = recorder.to_dataframe() 
    save_path = f"output/{symbol}/{method_name}/backtest_{phi}.parquet"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_parquet(save_path, index=False)
    print(f"Success: {method_name} backtest completed.")
    return df

def run_benchmark_and_save(strategy_name, policy, bundle, phi_param, fee_rate, save_dir, train_stats):
    s1, s2, mid, ask, bid, b_max, s_min = bundle
    env = FTPEnv(phi=phi_param, tick_size=0.001, fee_rate=fee_rate)
    recorder = StrategyRecorder()

    for t in range(len(mid)):
        raw_offsets = policy.get_action(env.inventory)
        mid_p = (ask[t] + bid[t]) / 2.0

        if strategy_name == 'glft':
            off_a = ((mid_p + raw_offsets[0]) - ask[t]) / 0.001
            off_b = (bid[t] - (mid_p - raw_offsets[1])) / 0.001
            action = np.round([off_a, off_b]).astype(int)
        else:
            action = np.round(raw_offsets).astype(int)
        
        reward, info = env.step(action, mid[t], ask[t], bid[t], b_max[t], s_min[t])

        record_data = {
            'step': t, 'mid': mid[t], 'ask': ask[t], 'bid': bid[t],
            'off_a': action[0], 'off_b': action[1], 
            'action': action, 'reward': reward, 'inventory': env.inventory, 'cash': env.cash,
            'fee_paid': info.get('fee_paid', 0.0),
            's1_pred': s1[t], 's2_pred': s2[t]
        }
        record_data.update(info)
        recorder.data.append(record_data)
        
    df = recorder.to_dataframe()
    save_path = f"{save_dir}/backtest_{phi_param}.parquet"
    os.makedirs(save_dir, exist_ok=True)
    df.to_parquet(save_path, index=False)
    print(f"Success: {strategy_name} benchmark completed.")
    return df


# loading signals and scaler
m1 = SGU1()
s1_raw_path = os.path.join(checkpoint_dir, f"sgu1_{sgu_train_range[0]}_{sgu_train_range[1]}.json")
tmp_s1_path = "tmp_sgu1_model.json"
shutil.copy2(s1_raw_path, tmp_s1_path)
try:
    m1.load(tmp_s1_path)
    print(">>> SGU1 loaded successfully.")
finally:
    if os.path.exists(tmp_s1_path):
        os.remove(tmp_s1_path)

m2 = SGU2(input_size=1, hidden_size=10)
s2_path = os.path.join(checkpoint_dir, f"sgu2_{sgu_train_range[0]}_{sgu_train_range[1]}.pth")
m2.load(s2_path)
print(">>> SGU2 (LSTM) loaded successfully.")

scaler_path = os.path.join(checkpoint_dir, f"sgu2_scaler_{sgu_train_range[0]}_{sgu_train_range[1]}.pkl")
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
    
s3_start_date = sgu_train_range[1]
print(f">>> Scaler loaded. S3 evaluation starts after: {s3_start_date}")


# load all bundles for test set
snap_dir = f'data/{symbol}/snap'
all_dates = sorted([f[:8] for f in os.listdir(snap_dir) if f.endswith('.parquet')])
s3_dates = [d for d in all_dates if int(d) > s3_start_date]
n_s3 = len(s3_dates)
if n_s3 < 5:
    raise ValueError(f"Insufficient S3 data after {s3_start_date}.")

train_idx = int(n_s3 * 0.70)
val_idx = int(n_s3 * 0.85)

train_dates = s3_dates[:train_idx]
test_dates = s3_dates[val_idx:]

print(f">>> S3 Data Pool: {n_s3} days | Train-subset: {len(train_dates)} | Test-subset: {len(test_dates)}")

print("Bundling S3-Train data...")
train_bundle = load_signals_bundle(symbol, train_dates, m1, m2, scaler)
print("Bundling S3-Test data...")
s3_bundle = load_signals_bundle(symbol, test_dates, m1, m2, scaler)


# all scalers for DL or the inventory level must be frozen on training set S1
# otherwise there are will be data leakages
s1_t, s2_t = train_bundle[0], train_bundle[1]

train_stats = {
    's1_m': np.mean(s1_t), 
    's1_s': np.std(s1_t) + 1e-9,
    's2_m': np.mean(s2_t), 
    's2_s': np.std(s2_t) + 1e-9
}

print(f">>> Normalization stats locked: {train_stats}")


# compute and compare with benchmarks
# benchmark1: FOIC
# benchmark2: GLFT
path_drl = os.path.join(checkpoint_dir, f"without_adv/agent_best_val_{phi_val}.pth")
path_arl = os.path.join(checkpoint_dir, f"with_adv/agent_best_val_{phi_val}.pth")
# DRL
run_drl_backtest(symbol, "drl", path_drl, s3_bundle, phi_val, 0, train_stats)
# Adv DRL
run_drl_backtest(symbol, "arl", path_arl, s3_bundle, phi_val, 0, train_stats)
# GLFT
glft_p = GLFTPolicy(gamma=0.0001, kappa=3000, A=0.1, sigma=0.0005)
run_benchmark_and_save("glft", glft_p, s3_bundle, phi_val, 0, f"output/{symbol}/glft", train_stats)
# FOIC
foic_p = FOICPolicy(offset_a=0, offset_b=0)
run_benchmark_and_save("foic", foic_p, s3_bundle, phi_val, 0, f"output/{symbol}/foic", train_stats)

print("\n>>> All backtests completed. Parquet datasets generated in 'output/' directory.")


# load models 
base_path = f"output/{symbol}"
methods = ['arl', 'drl', 'glft', 'foic']

dfs = {}
for m in methods:
    path = f"{base_path}/{m}/backtest_{phi_val}.parquet"
    dfs[m] = pd.read_parquet(path)

print(f">>> Successfully loaded datasets for: {methods}")


# hypothesis tests
def dm_test(actual, pred1, pred2, h=1, crit="MSE"):
    e1 = (actual - pred1)**2 if crit=="MSE" else abs(actual - pred1)
    e2 = (actual - pred2)**2 if crit=="MSE" else abs(actual - pred2)
    d = e1 - e2
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    dm_stat = mean_d / np.sqrt(var_d / len(d))
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value


df_arl = dfs['arl']
actual_mid_change = df_arl['mid'].shift(-1) - df_arl['mid']
pred_sgu = df_arl['s2_pred']
naive_pred = np.zeros_like(pred_sgu)
dm_stat, p_val = dm_test(actual_mid_change[:-1], pred_sgu[:-1], naive_pred[:-1])
print(f"H1 (DM Test): Statistic={dm_stat:.4f}, p-value={p_val:.4f}")


dw_stat = durbin_watson(dfs['arl']['reward'])
print(f"H2: DW-Stat = {dw_stat:.4f}")


def get_mbb_sharpe(wealth_series, block_size=100, n_boot=500):
    """
    Calculate per-step Sharpe Ratio using Moving Block Bootstrap with safety checks.
    """
    returns = wealth_series.pct_change().dropna().values
    n = len(returns)
    
    if n <= block_size or n == 0:
        std = np.std(returns)
        sr = np.mean(returns) / std if std > 1e-9 else 0
        return sr, (sr, sr)

    boot_srs = []
    num_blocks = max(1, n // block_size)
    
    for _ in range(n_boot):
        max_start_idx = n - block_size
        indices = np.random.randint(0, max_start_idx + 1, size=num_blocks)
        
        resampled_list = [returns[i:i+block_size] for i in indices]
        
        if not resampled_list:
            boot_srs.append(0)
            continue
            
        resampled = np.concatenate(resampled_list)
        
        ret_std = np.std(resampled)
        if ret_std > 1e-9:
            sr = np.mean(resampled) / ret_std
        else:
            sr = 0
        boot_srs.append(sr)
        
    return np.mean(boot_srs), np.percentile(boot_srs, [2.5, 97.5])


print("H3: 95% CI of Per-Step Sharpe:")
for m in methods:
    if m in dfs and not dfs[m].empty:
        try:
            mean_sr, ci = get_mbb_sharpe(dfs[m]['wealth'], block_size=100)
            print(f"  - {m.upper():<4}: {mean_sr:>8.6f} | CI: [{ci[0]:>8.6f}, {ci[1]:>8.6f}]")
        except Exception as e:
            print(f"  - {m.upper():<4}: Error processing data - {e}")
    else:
        print(f"  - {m.upper():<4}: No data found in DataFrame.")


print("Execution Statistics:")
for m in methods:
    if m in dfs:
        total_fills = dfs[m]['fill_buy'].sum() + dfs[m]['fill_sell'].sum()
        print(f"  - {m.upper():<4}: {total_fills} orders filled")
    else:
        print(f"  - {m.upper():<4}: No data found")


def get_clean_stats(df, name):
    pnl_series = df['wealth'] - df['wealth'].iloc[0]
    rolling_map = df['inventory'].abs().expanding().mean()
    pnl_map_series = pnl_series / np.maximum(rolling_map, 1e-2)
    
    returns = df['wealth'].diff().dropna()
    sharpe = returns.mean() / (returns.std() + 1e-9)
    mdd = (df['wealth'] - df['wealth'].cummax()).min()
    fills = df['fill_buy'].sum() + df['fill_sell'].sum()
    
    return {
        'Method': name.upper(),
        'PnL-to-MAP': round(pnl_map_series.iloc[-1], 4),
        'Sharpe': round(sharpe, 4),
        'Fills': int(fills)
    }

results = [get_clean_stats(dfs[m], m) for m in methods if m in dfs]
stats_df = pd.DataFrame(results)
print("\n=== Performance Summary ===")
print(stats_df.to_string(index=False))

plt.figure(figsize=(12, 6))

for m in methods:
    if m in dfs:
        df = dfs[m]
        pnl_series = df['wealth'] - df['wealth'].iloc[0]
        rolling_map = df['inventory'].abs().expanding().mean()
        safe_map = np.maximum(rolling_map, 0.1)
        pnl_map_series = pnl_series / safe_map
        plt.plot(pnl_map_series.index, pnl_map_series.values, label=f"{m.upper()}", linewidth=2.0)

plt.title(f"Rolling PnL-to-MAP Ratio (OOS: {symbol})")
plt.xlabel("Environmental Step")
plt.ylabel("PnL / MAP")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(10, 6))
for m, color in zip(['arl', 'drl'], ['blue', 'orange']):
    if m in dfs:
        sample_df = dfs[m].sample(n=min(3000, len(dfs[m])))
        sns.regplot(data=sample_df, x='inventory', y='skew', 
                    label=f'{m.upper()} Rationality', 
                    scatter_kws={'alpha':0.1, 'color': color}, 
                    line_kws={'color': color})

plt.axhline(0, color='black', lw=1, ls='--')
plt.axvline(0, color='black', lw=1, ls='--')
plt.title("Inventory vs Price Skew Correlation")
plt.xlabel("Inventory Level")
plt.ylabel("Price Skew (off_b - off_a)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()