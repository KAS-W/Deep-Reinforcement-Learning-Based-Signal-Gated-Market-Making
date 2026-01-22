import os
import torch
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from models.GateUnits import SGU1, SGU2
from models.model import TradingPolicy
from loaders.HFTLoader import SGU1DataPro, SGU2DataPro
from Env.market_env import FTPEnv
from Env.recorder import StrategyRecorder
from Env.drl_engine import DRLEngine
from analytics.mm_analyzer import StrategyAnalytics, BacktestVisualizer

def load_signals_bundle(symbol, date_list, m1, m2, scaler):
    sgu1_sigs, sgu2_sigs = [], []
    mids, asks, bids = [], [], []
    buy_maxs, sell_mins = [], []

    snap_dir = f'data/{symbol}/snap'
    tick_dir = f'data/{symbol}/tick'

    for date_str in tqdm(date_list, desc="Bundling Signals", leave=False):
        snap = pd.read_parquet(os.path.join(snap_dir, f"{date_str}.parquet"))
        tick = pd.read_parquet(os.path.join(tick_dir, f"{date_str}.parquet"))

        snap = snap[(snap['trade_time'] >= 93000000) & (snap['askprice1'] > 0) & (snap['bidprice1'] > 0)].copy()
        if snap.empty: 
            continue

        # generate sgu1 signals
        loader1 = SGU1DataPro(tick_df=tick, snap_df=snap)
        df1 = loader1.gen_dataset(event_step=19)
        if df1.empty: 
            continue
        s1_pred = m1.predict(df1.drop(columns=['label']))

        # generate sgu2 signals
        loader2 = SGU2DataPro(tick_df=tick, snap_df=snap)
        X2, _ = loader2.gen_dataset(event_step=19, time_steps=10)
        if X2.size == 0: 
            continue
        X2_scaled = scaler.transform(X2)
        s2_pred = m2.predict(X2_scaled).flatten()

        # align signals and extract mid-price
        min_len = min(len(s1_pred), len(s2_pred))
        s1_aligned, s2_aligned = s1_pred[-min_len:], s2_pred[-min_len:]
        event_df_sampled = loader2.event_df.iloc[::19].iloc[-min_len:]

        # we need max & min within each step: 19 shifts
        # these ticks' path matters
        step_buy_max, step_sell_min = [], []
        for i in range(len(event_df_sampled) - 1):
            idx_start = event_df_sampled.index[i]
            idx_end = event_df_sampled.index[i+1]
            window = loader2.event_df.loc[idx_start : idx_end]
            step_buy_max.append(window['p_buy_max'].max())
            step_sell_min.append(window['p_sell_min'].min())

        sgu1_sigs.append(s1_aligned[:-1])
        sgu2_sigs.append(s2_aligned[:-1])
        buy_maxs.append(np.array(step_buy_max))
        sell_mins.append(np.array(step_sell_min))

        p_current = event_df_sampled.iloc[:-1]
        # mid-price of next step is for trade matchings
        p_next = event_df_sampled.iloc[1:]
        # bid-ask must use current price
        asks.append(p_current['askprice1'].values) 
        bids.append(p_current['bidprice1'].values)
        # reward is determined at next step
        mids.append(((p_next['askprice1'] + p_next['bidprice1']) / 2).values)

    return (np.concatenate(sgu1_sigs), np.concatenate(sgu2_sigs), 
            np.concatenate(mids), np.concatenate(asks), np.concatenate(bids),
            np.concatenate(buy_maxs), np.concatenate(sell_mins))

def run_agent_training_pipeline(symbol, sgu_train_range, PHI=0.001, TICK_SIZE=0.01, fee_rate=0.00005, USE_FEE=False, USE_ARL=True):

    checkpoint_dir = f"checkpoints/{symbol}"
    if USE_FEE == False:
        fee_rate = 0.0

    # Assets are locked based on the SGU training session
    m1 = SGU1()
    m1.load(os.path.join(checkpoint_dir, f"sgu1_{sgu_train_range[0]}_{sgu_train_range[1]}.json"))
    
    m2 = SGU2(input_size=1, hidden_size=10)
    m2.load(os.path.join(checkpoint_dir, f"sgu2_{sgu_train_range[0]}_{sgu_train_range[1]}.pth"))

    scaler_path = os.path.join(checkpoint_dir, f"sgu2_scaler_{sgu_train_range[0]}_{sgu_train_range[1]}.pkl")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Automatic inference: Filter dates to ensure they are strictly AFTER the SGU range
    snap_dir = f'data/{symbol}/snap'
    all_dates = sorted([f[:8] for f in os.listdir(snap_dir) if f.endswith('.parquet')])
    
    # Define S3: Only use data that SGU has never 'seen' during its training/validation
    s3_start_date = sgu_train_range[1]
    s3_dates = [d for d in all_dates if int(d) > s3_start_date]
    
    n_s3 = len(s3_dates)
    if n_s3 < 5:
        raise ValueError(f"Insufficient S3 data after {s3_start_date}. Found only {n_s3} days.")

    # Apply 70/15/15 split EXCLUSIVELY within the S3 date pool
    train_idx = int(n_s3 * 0.70)
    val_idx = int(n_s3 * 0.85)

    train_dates = s3_dates[:train_idx]
    val_dates = s3_dates[train_idx:val_idx]
    test_dates = s3_dates[val_idx:]

    print(f">>> Agent Dates after: {s3_start_date}:")
    print(f"    S3 Total: {n_s3} days | Train: {len(train_dates)} | Val: {len(val_dates)} | Test: {len(test_dates)}")

    # Load Bundles only for these S3 segments
    train_bundle = load_signals_bundle(symbol, train_dates, m1, m2, scaler)
    val_bundle = load_signals_bundle(symbol, val_dates, m1, m2, scaler)
    test_bundle = load_signals_bundle(symbol, test_dates, m1, m2, scaler)

    # Stats are computed and locked based on the S3-Train segment
    s1_t, s2_t = train_bundle[0], train_bundle[1]
    train_stats = {
        's1_m': np.mean(s1_t), 's1_s': np.std(s1_t) + 1e-9,
        's2_m': np.mean(s2_t), 's2_s': np.std(s2_t) + 1e-9
    }

    # Training engine execution
    if USE_ARL:
        agent_path = f"checkpoints/{symbol}/with_adv"
    else:
        agent_path = f"checkpoints/{symbol}/without_adv"
    engine = DRLEngine(pop_size=50, sigma=0.05, phi=PHI, tick_size=TICK_SIZE, save_dir=agent_path, use_arl=USE_ARL)
    best_agent, _ = engine.train(train_bundle, val_bundle, train_stats, generations=100)

    # Blind Test on the last 15% of S3
    recorder = StrategyRecorder()
    env = FTPEnv(phi=PHI, tick_size=TICK_SIZE, fee_rate=fee_rate)
    s1, s2, mid, ask, bid, buy_max, sell_min = test_bundle

    with torch.no_grad():
        for t in range(len(mid)):
            n_s = torch.tensor([[(s1[t]-train_stats['s1_m'])/train_stats['s1_s'], 
                                 (s2[t]-train_stats['s2_m'])/train_stats['s2_s'], 
                                 env.inventory/2.0]], dtype=torch.float32)
            
            raw_act = best_agent.forward(n_s).squeeze().cpu().numpy()
            scaled_act = np.round(raw_act * 5).astype(int) 
            reward, info = env.step(scaled_act, mid[t], ask[t], bid[t], buy_max[t], sell_min[t], adv_action=None)
            recorder.record(t, mid[t], ask[t], bid[t], scaled_act, reward, env.inventory, env.cash, info)

    res_df = recorder.to_dataframe()
    metrics = StrategyAnalytics(res_df).summary_dict
    report_path = f"output/{symbol}/phi_{PHI}_S3_TEST.png"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    BacktestVisualizer.plot_professional_report(res_df, metrics, save_path=report_path, show_fees=USE_FEE)
    print(f">>> Complete. Report saved to: {report_path}")


# if __name__ == '__main__':
#     run_agent_training_pipeline(
#         symbol="688981", 
#         sgu_train_range=(20240401, 20240528), 
#         PHI=0.01
#     )