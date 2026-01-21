import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from models.GateUnits import SGU1, SGU2
from loaders.HFTLoader import SGU1DataPro, SGU2DataPro
from utils.scaler import StandardScaler3D

def get_date_list(snap_dir, start, end):
    """
    Fetch available dates by reading files
    """
    all_files = [f for f in os.listdir(snap_dir) if f.endswith('.parquet')]
    dates = sorted([f[:8] for f in all_files if start <= int(f[:8]) <= end])
    return dates

def load_sgu_bundles(symbol, start, end, event_step=19, time_steps=10):
    """
    Read data synchronizedly and departure to loaders
    """
    snap_dir = f'data/{symbol}/snap'
    tick_dir = f'data/{symbol}/tick'

    dates = get_date_list(snap_dir, start, end)
    s1_x_list, s1_y_list = [], []  # training
    s2_x_list, s2_y_list = [], []  # validation

    for date_str in tqdm(dates, desc=f"Processing {symbol} ({start}-{end})", leave=False):
        snap_path = os.path.join(snap_dir, f"{date_str}.parquet")
        tick_path = os.path.join(tick_dir, f"{date_str}.parquet")

        if not os.path.exists(tick_path): 
            continue

        snap_df = pd.read_parquet(snap_path)
        tick_df = pd.read_parquet(tick_path)

        # filter out data before the market open
        snap_df = snap_df[snap_df['trade_time'] >= 93000000].copy()
        tick_df = tick_df[tick_df['trade_time'] >= 92500000].copy()

        # departure to sgu1
        loader1 = SGU1DataPro(tick_df=tick_df, snap_df=snap_df)
        df1 = loader1.gen_dataset(event_step=event_step)
        if not df1.empty:
            s1_x_list.append(df1.drop(columns=['label']))
            s1_y_list.append(df1['label'] * 1000)

        # departure to sgu2
        loader2 = SGU2DataPro(tick_df=tick_df, snap_df=snap_df)
        X2, y2 = loader2.gen_dataset(event_step=event_step, time_steps=time_steps)
        if X2.size > 0:
            s2_x_list.append(X2)
            s2_y_list.append(y2 * 10000)

    if s1_x_list:
        s1_data = (pd.concat(s1_x_list, axis=0), pd.concat(s1_y_list, axis=0))
    else:
        s1_data = (None, None)
    
    if s2_x_list:
        s2_data = (np.concatenate(s2_x_list, axis=0), np.concatenate(s2_y_list, axis=0))
    else:
        s2_data = (None, None)
    
    return s1_data, s2_data

def run_sgu_training(symbol, train_range, val_range, event_step=19, time_steps=10):
    print(f"\n>>> Loading Data for {symbol}...")
    (s1_x_train, s1_y_train), (s2_x_train, s2_y_train) = load_sgu_bundles(symbol, train_range[0], train_range[1], event_step, time_steps)
    (s1_x_val, s1_y_val), (s2_x_val, s2_y_val) = load_sgu_bundles(symbol, val_range[0], val_range[1], event_step, time_steps)

    checkpoint_dir = f"checkpoints/{symbol}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # training xgb for sgu1
    if s1_x_train is not None:
        m1 = SGU1()
        m1.model.set_params(verbose=False) 
        m1.train(s1_x_train, s1_y_train, s1_x_val, s1_y_val)

        preds = m1.predict(s1_x_val)
        corr = np.corrcoef(preds, s1_y_val)[0, 1]
        mse = np.mean((preds - s1_y_val)**2)

        m1.save(os.path.join(checkpoint_dir, f"sgu1_{train_range[0]}_{train_range[1]}.json"))
        print(f"[SGU1] Validation -> Corr: {corr:.4f} | MSE: {mse:.6f}")

    # training lstm for sgu2
    if s2_x_train is not None:
        scaler = StandardScaler3D()
        X2_train_scaled = scaler.fit_transform(s2_x_train)
        X2_val_scaled = scaler.transform(s2_x_val)

        m2 = SGU2(input_size=X2_train_scaled.shape[2], hidden_size=10)
        m2.train(X2_train_scaled, s2_y_train, X2_val_scaled, s2_y_val, epochs=50, batch_size=256)

        preds = m2.predict(X2_val_scaled).flatten()
        y_true = s2_y_val.flatten()
        corr = np.corrcoef(preds, y_true)[0, 1]
        mse = np.mean((preds - y_true)**2)

        m2.save(os.path.join(checkpoint_dir, f"sgu2_{train_range[0]}_{train_range[1]}.pth"))
        print(f"[SGU2] Validation -> Corr: {corr:.4f} | MSE: {mse:.6f}")

        # save scaler on training set for validation and testing
        scaler_path = os.path.join(checkpoint_dir, f"sgu2_scaler_{train_range[0]}_{train_range[1]}.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

    print(f">>> SGU training complete. Weights frozen in {checkpoint_dir}")


# if __name__ == '__main__':
#     run_sgu_training(
#         symbol="688981", 
#         train_range=(20240401, 20240528), 
#         val_range=(20240529, 20240612)
#     )