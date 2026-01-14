import os
import glob
import pandas as pd
import torch
import numpy as np
import random
from tqdm import tqdm
from utils.analyzer import evaluate_sgu1_comparison, plot_sgu2_loss
from models.dataloader import SGU2Dataset
from models.sgu2 import SGU2
from utils.scaler import StandardScaler3D, RobustWinsorScaler3D

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dataloader_sgu2(data_pro='snap', start=20240401, end=20240531, time_steps=25, k_future=10):
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
    
        day_df = day_df[day_df['trade_time'] >= 92500000].copy()

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
    X_train = scaler.fit_transform(X_train)
    X_val, y_val = dataloader_sgu2(start=20240513, end=20240531)
    X_val = scaler.transform(X_val)

    model = SGU2(input_size=23, hidden_size=64)
    history = model.train(X_train, y_train, X_val, y_val, batch_size=256, epochs=50)

    val_preds = model.predict(X_val)
    val_corr = np.corrcoef(val_preds, y_val)[0, 1]
    print(f"\nS2 Slope Correlation: {val_corr:.4f}")

    os.makedirs('checkpoints/sample_lstm', exist_ok=True)
    save_path = "checkpoints/sample_lstm/sgu2_model.pth"
    model.save(save_path)
    print(f"SGU2 Model saved to {save_path}")

    # plot val
    evaluate_sgu1_comparison(model, X_val, y_val, 'D:/UW/Course/2026 WINTER/522_trade_sys/replication/img/sgu2_graph/sgu2_val.png')

    # plot importance
    plot_sgu2_loss(history, 'D:/UW/Course/2026 WINTER/522_trade_sys/replication/img/sgu2_graph/sgu2_loss.png')

    return model


if __name__ == '__main__':
    set_seed(114514)
    model = sgu2_pipeline()