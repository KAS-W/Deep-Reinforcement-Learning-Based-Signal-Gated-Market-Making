import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.analyzer import evaluate_sgu1_comparison, plot_importance
from models.HFTLoader import SGU1DataPro
from models.sgu1 import SGU1

def pro_dataloader(start: int, end: int, event_step: int = 19, start_flag:int=935):
    snap_dir = r'D:/UW/Course/2026 WINTER/522_trade_sys/replication/data/snap'
    tick_dir = r'D:/UW/Course/2026 WINTER/522_trade_sys/replication/data/tick'
    
    all_snap_files = sorted(glob.glob(os.path.join(snap_dir, '*.parquet')))
    sample_dates = [
        os.path.basename(f)[:8] for f in all_snap_files 
        if start <= int(os.path.basename(f)[:8]) <= end
    ]

    day_datasets = []

    for date_str in tqdm(sample_dates, desc=f"Loading SGU1 Data ({start}-{end})"):
        snap_path = os.path.join(snap_dir, f"{date_str}.parquet")
        tick_path = os.path.join(tick_dir, f"{date_str}.parquet")

        if not os.path.exists(tick_path):
            continue

        snap_df = pd.read_parquet(snap_path)
        tick_df = pd.read_parquet(tick_path)
        
        flag_point = int(start_flag * 100000)
        snap_df = snap_df[snap_df['trade_time'] >= flag_point].copy()
        tick_df = tick_df[tick_df['trade_time'] >= flag_point].copy()

        loader = SGU1DataPro(tick_df=tick_df, snap_df=snap_df)

        day_data = loader.gen_dataset(delta_t=event_step)

        if not day_data.empty:
            day_datasets.append(day_data)

    if not day_datasets:
        return pd.DataFrame()
    
    full_data = pd.concat(day_datasets, axis=0).reset_index(drop=True)

    target = 'label'
    if target in full_data.columns:
        full_data[target] = full_data[target] * 1000
        
    return full_data

def pro_sgu1_pipeline(s1=(20240401, 20240512), s2=(20240513, 20240531), event_step=19):
    s1_data = pro_dataloader(start=s1[0], end=s1[1], event_step=event_step)
    s2_data = pro_dataloader(start=s2[0], end=s2[1], event_step=event_step)

    target = 'label'
    if s1_data.empty or s2_data.empty:
        print("Data loading failed. Check file paths and dates.")
        return None, 0.0

    X_train, y_train = s1_data.drop(columns=[target]), s1_data[target]
    X_val, y_val = s2_data.drop(columns=[target]), s2_data[target]
    
    # train
    model = SGU1()
    model.train(X_train, y_train, X_val, y_val)
    # eval
    val_preds = model.predict(X_val)
    val_corr = np.corrcoef(val_preds, y_val)[0, 1]
    val_rmse = np.sqrt(np.mean((val_preds - y_val)**2))
    val_mape = np.mean(np.abs((y_val - val_preds) / (y_val + 1e-9)))
    print("\n--- SGU1 Validation Results ---")
    print(f"Correlation: {val_corr:.4f}")
    print(f"RMSE:        {val_rmse:.6f}")
    print(f"MAPE:        {val_mape:.4f}")
    os.makedirs('checkpoints/paper_xgb', exist_ok=True)
    save_path = f"checkpoints/paper_xgb/sgu1_xgboost_{s1[0]}_{s1[1]}.json"
    model.save(save_path)
    print(f"Model saved to: {save_path}")

    img_dir = 'D:/UW/Course/2026 WINTER/522_trade_sys/replication/img/sgu1_graph'
    os.makedirs(img_dir, exist_ok=True)

    evaluate_sgu1_comparison(model, X_val, y_val, os.path.join(img_dir, 'sgu1_val_paper.png'))
    plot_importance(model, os.path.join(img_dir, 'sgu1_importance_paper.png'))

    return model, val_corr


if __name__ == '__main__':
    sgu1_model, correlation = pro_sgu1_pipeline(
        s1=(20240401, 20240420),
        s2=(20240421, 20240430),
        event_step=19
    )