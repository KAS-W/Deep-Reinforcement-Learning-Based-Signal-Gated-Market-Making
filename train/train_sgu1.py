import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.analyzer import evaluate_sgu1_comparison, plot_importance
from models.dataloader import SGU1Dataset
from models.sgu1 import SGU1

def dataloader(data_pro:str='', start:int=20240501, end:int=20240531):
    if data_pro == 'snap':
        data_dir = r'D:/UW/Course/2026 WINTER/522_trade_sys/replication/data/snap'
    elif data_pro == 'tick':
        data_dir = r'D:/UW/Course/2026 WINTER/522_trade_sys/replication/data/tick'
    else:
        raise ValueError('Datatype D.N.E.')
    
    all_files = sorted(glob.glob(os.path.join(data_dir, '*.parquet')))
    sample_files = [
        f for f in all_files 
        if start <= int(os.path.basename(f)[:8]) <= end
    ]

    if not sample_files:
        print(f"No files found between {start} and {end}.")
        return pd.DataFrame()
    
    day_datasets = []

    for f in tqdm(sample_files, desc=f"Loading SGU1 Data ({start}-{end})"):
        day_df = pd.read_parquet(f)
        if day_df.empty:
            continue

        day_df = day_df[day_df['trade_time'] >= 92500000].copy()

        loader = SGU1Dataset(day_df)

        day_data = loader.gen_dataset(use_event_sampling=True)

        if not day_data.empty:
            day_datasets.append(day_data)

    if not day_datasets:
        return pd.DataFrame()
    
    full_data = pd.concat(day_datasets, axis=0).reset_index(drop=True)

    target = 'label_rr_10'
    if target in full_data.columns:
        full_data[target] = full_data[target] * 1000
    return full_data

def sgu1_pipeline(data_pro='snap', s1=(20240501, 20240518), s2=(20240519, 20240525)):
    s1_data = dataloader(data_pro, start=s1[0], end=s1[1])
    s2_data = dataloader(data_pro, start=s2[0], end=s2[1])

    target = 'label_rr_10'

    X_train, y_train = s1_data.drop(columns=[target]), s1_data[target]
    X_val, y_val = s2_data.drop(columns=[target]), s2_data[target]
    
    model = SGU1()
    model.train(X_train, y_train, X_val, y_val)

    val_preds = model.predict(X_val)

    val_corr = np.corrcoef(val_preds, y_val)[0, 1]
    val_rmse = np.sqrt(np.mean((val_preds - y_val)**2))
    val_mape = np.mean(np.abs((y_val - val_preds) / (y_val + 1e-9)))
    print(f"S2 Correlation: {val_corr:.4f}")
    print(f"S2 RMSE:        {val_rmse:.6f}")
    print(f"S2 MAPE:        {val_mape:.4f}")

    os.makedirs('models', exist_ok=True)
    save_path = f"checkpoints/sample_xgb/sgu1_xgboost_{s1[0]}_{s1[1]}.json"
    model.save(save_path)
    print(f"\nModel saved to {save_path}")

    # plot val
    evaluate_sgu1_comparison(model, X_val, y_val, 'D:/UW/Course/2026 WINTER/522_trade_sys/replication/img/sgu1_graph/sgu1_val.png')

    # plot importance
    plot_importance(model, 'D:/UW/Course/2026 WINTER/522_trade_sys/replication/img/sgu1_graph/sgu1_importance.png')
    
    return model, val_corr


if __name__ == '__main__':
    sgu1, val_corr = sgu1_pipeline()