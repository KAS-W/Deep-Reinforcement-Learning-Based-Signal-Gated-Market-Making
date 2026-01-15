import os
import glob
import pandas as pd
import torch
import numpy as np
import random
from tqdm import tqdm
from utils.analyzer import evaluate_sgu1_comparison, plot_sgu2_loss
from models.dataloader import SGU2Dataset
from models.dataloader import SGU2DataPro
from models.sgu2 import SGU2
from utils.scaler import StandardScaler3D
from utils.scaler import FeatureProcesser

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dataloader_sgu2(data_pro='snap', start=20240401, end=20240531, time_steps=10, k_future=10):
    data_dir = rf'D:/UW/Course/2026 WINTER/522_trade_sys/replication/data/{data_pro}'
    all_files = sorted(glob.glob(os.path.join(data_dir, '*.parquet')))
    sample_files = [f for f in all_files if start <= int(os.path.basename(f)[:8]) <= end]

    if not sample_files:
        return np.array([]), np.array([])
    
    X_list, y_list = [], []

    for f in tqdm(sample_files, desc=f"Loading SGU2 (3D) Data ({start}-{end})"):
        day_df = pd.read_parquet(f)
        if day_df.empty: 
            continue
    
        day_df = day_df[day_df['trade_time'] >= 93000000].copy()

        loader = SGU2Dataset(day_df)
        # gen 3-d: (N, TimeSteps, 23)
        X_day, y_day = loader.gen_dataset(time_steps=time_steps, k_future=k_future)
        
        if X_day.size > 0:
            X_list.append(X_day)
            y_list.append(y_day)

    if not X_list:
        return np.array([]), np.array([])
    
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

def sgu2_pipeline():
    X_train, y_train = dataloader_sgu2(start=20240401, end=20240512)
    # scaler = FeatureProcesser()
    scaler = StandardScaler3D()
    # scaler = RobustWinsorScaler3D()
    # X_train = scaler.fit_transform(X_train)
    X_val, y_val = dataloader_sgu2(start=20240513, end=20240531)
    # X_val = scaler.transform(X_val)

    model = SGU2(input_size=23, hidden_size=64)
    history = model.train(X_train, y_train, X_val, y_val, batch_size=256, epochs=50)

    os.makedirs('checkpoints/sample_lstm', exist_ok=True)
    save_path = "checkpoints/sample_lstm/sgu2_model.pth"
    model.save(save_path)
    print(f"SGU2 Model saved to {save_path}")

    # plot val
    evaluate_sgu1_comparison(model, X_val, y_val, 'D:/UW/Course/2026 WINTER/522_trade_sys/replication/img/sgu2_graph/sgu2_val.png')

    # plot importance
    plot_sgu2_loss(history, 'D:/UW/Course/2026 WINTER/522_trade_sys/replication/img/sgu2_graph/sgu2_loss.png')

    return model

def pro_dataloader_sgu2(start:int=20240501, end:int=20240531, time_steps=50, k_future=10):
    snap_dir = r'D:/UW/Course/2026 WINTER/522_trade_sys/replication/data/snap'
    tick_dir = r'D:/UW/Course/2026 WINTER/522_trade_sys/replication/data/tick'
    
    all_snap_files = sorted(glob.glob(os.path.join(snap_dir, '*.parquet')))
    sample_dates = [
        os.path.basename(f)[:8] for f in all_snap_files 
        if start <= int(os.path.basename(f)[:8]) <= end
    ]

    X_list, y_list = [], []

    for date_str in tqdm(sample_dates, desc=f"Loading Pro SGU2 Data ({start}-{end})"):
        snap_path = os.path.join(snap_dir, f"{date_str}.parquet")
        tick_path = os.path.join(tick_dir, f"{date_str}.parquet")

        if not os.path.exists(tick_path):
            print(f"Tick file missing for {date_str}, skipping...")
            continue

        snap_df = pd.read_parquet(snap_path)
        tick_df = pd.read_parquet(tick_path)

        snap_df = snap_df[snap_df['trade_time'] >= 93000000].copy()
        tick_df = tick_df[tick_df['trade_time'] >= 93000000].copy()

        loader = SGU2DataPro(tick_df=tick_df, snap_df=snap_df)

        X_day, y_day = loader.gen_dataset(time_steps=time_steps, k_future=k_future, event_step=8)

        if X_day.size > 0:
            X_list.append(X_day)
            y_list.append(y_day)

    if not X_list:
        return np.array([]), np.array([])
    
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

def pro_sgu2_pipeline(s1=(20240501, 20240518), s2=(20240519, 20240525)):
    X_train, y_train = pro_dataloader_sgu2(start=s1[0], end=s1[1])
    X_val, y_val = pro_dataloader_sgu2(start=s2[0], end=s2[1])

    scaler = StandardScaler3D()
    # scaler = FeatureProcesser()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    model = SGU2(input_size=X_train.shape[2], hidden_size=64)

    history = model.train(X_train, y_train, X_val, y_val, batch_size=256, epochs=100)

    val_preds = model.predict(X_val)
    val_corr = np.corrcoef(val_preds, y_val)[0, 1]
    val_rmse = np.sqrt(np.mean((val_preds - y_val)**2))
    print(f"\nPro S2 Slope Correlation: {val_corr:.4f}")
    print(f"Pro S2 Slope RMSE:        {val_rmse:.6f}")

    os.makedirs('checkpoints/sample_lstm_pro', exist_ok=True)
    save_path = f"checkpoints/sample_lstm_pro/sgu2_model_{s1[0]}_{s1[1]}.pth"
    model.save(save_path)
    print(f"SGU2 Pro Model saved to {save_path}")

    img_dir = 'D:/UW/Course/2026 WINTER/522_trade_sys/replication/img/sgu2_graph'
    os.makedirs(img_dir, exist_ok=True)
    evaluate_sgu1_comparison(model, X_val, y_val, os.path.join(img_dir, 'sgu2_val_pro.png'))
    plot_sgu2_loss(history, os.path.join(img_dir, 'sgu2_loss_pro.png'))

    return model


if __name__ == '__main__':
    set_seed(114514)
    # model = sgu2_pipeline()
    model = pro_sgu2_pipeline()