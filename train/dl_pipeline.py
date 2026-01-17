import os
import glob
import pandas as pd
import torch
import numpy as np
import random
from tqdm import tqdm
from models.HFTLoader import SGU2DataPro
from models.sgu2 import SGU2
from utils.scaler import StandardScaler3D
from utils.analyzer import evaluate_sgu2, plot_sgu2_loss

def set_seed(seed=114514):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pro_dataloader_sgu2(start, end, time_steps=10, event_step=19):
    snap_dir = r'D:/UW/Course/2026 WINTER/522_trade_sys/replication/data/snap'
    tick_dir = r'D:/UW/Course/2026 WINTER/522_trade_sys/replication/data/tick'
    
    all_snap_files = sorted(glob.glob(os.path.join(snap_dir, '*.parquet')))
    sample_dates = [
        os.path.basename(f)[:8] for f in all_snap_files 
        if start <= int(os.path.basename(f)[:8]) <= end
    ]

    X_list, y_list = [], []

    for date_str in tqdm(sample_dates, desc=f"Loading SGU2 Data ({start}-{end})"):
        snap_path = os.path.join(snap_dir, f"{date_str}.parquet")
        tick_path = os.path.join(tick_dir, f"{date_str}.parquet")

        if not os.path.exists(tick_path):
            continue

        snap_df = pd.read_parquet(snap_path)
        tick_df = pd.read_parquet(tick_path)

        snap_df = snap_df[snap_df['trade_time'] >= 93000000].copy()
        tick_df = tick_df[tick_df['trade_time'] >= 92500000].copy()

        loader = SGU2DataPro(tick_df=tick_df, snap_df=snap_df)
        X_day, y_day = loader.gen_dataset(event_step=event_step, time_steps=time_steps)

        if X_day.size > 0:
            X_list.append(X_day)
            y_list.append(y_day)

    if not X_list:
        return np.array([]), np.array([])
    
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

def pro_sgu2_pipeline(s1=(20240401, 20240420), s2=(20240421, 20240430), time_steps=10, event_step=19):
    X_train, y_train = pro_dataloader_sgu2(start=s1[0], end=s1[1], time_steps=time_steps, event_step=event_step)
    X_val, y_val = pro_dataloader_sgu2(start=s2[0], end=s2[1], time_steps=time_steps, event_step=event_step)

    if X_train.size == 0 or X_val.size == 0:
        print("Data loading failed. Check your data paths.")
        return None

    y_train, y_val = y_train * 10000, y_val * 10000
    
    scaler = StandardScaler3D()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    model = SGU2(input_size=X_train.shape[2], hidden_size=10)

    history = model.train(X_train, y_train, X_val, y_val, batch_size=128, epochs=100)

    val_preds = model.predict(X_val).flatten()
    y_val_flat = y_val.flatten()
    current_corr = np.corrcoef(val_preds, y_val_flat)[0, 1]
    
    lag_corr = np.corrcoef(val_preds[1:], y_val_flat[:-1])[0, 1]

    img_dir = 'D:/UW/Course/2026 WINTER/522_trade_sys/replication/img/sgu2_graph'
    os.makedirs(img_dir, exist_ok=True)
    
    evaluate_sgu2(model, X_val, y_val, os.path.join(img_dir, 'sgu2_val_sample.png'))
    plot_sgu2_loss(history, os.path.join(img_dir, 'sgu2_loss_sample.png'))

    save_path = f"checkpoints/sample_sgu2/sgu2_model_{s1[0]}_{s1[1]}.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    
    return model


if __name__ == '__main__':
    set_seed(114514)
    pro_sgu2_pipeline(
        s1=(20240401, 20240510),
        s2=(20240511, 20240520),
        time_steps=10, 
        event_step=19  
    )
