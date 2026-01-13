import pandas as pd
import numpy as np
from pathlib import Path
import py7zr
import tempfile
from tqdm import tqdm

def process_tick(tick_df, date_val):
    tick_df['trade_date'] = np.int32(date_val)
    t = pd.to_datetime(tick_df['Time'], format='%H:%M:%S')
    tick_df['trade_time'] = t.dt.hour * 10000000 + t.dt.minute * 100000 + t.dt.second * 1000
    tick_df['side'] = tick_df['Type'].map({'B': 1, 'S': -1}).fillna(0).astype(np.int8)
    del tick_df['Time']
    del tick_df['Type']
    del t   
    return tick_df

def process_snap(snap_df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        '自然日': 'trade_date',
        '时间': 'trade_time',
        '成交价': 'trade_price',
        '成交量': 'vol',
        '成交额': 'amt',
        '成交笔数': 'cnt',
        '当日累计成交量': 'cum_vol',
        '当日成交额': 'cum_amt',
        '最高价': 'high',
        '最低价': 'low',
        '开盘价': 'open',
        '前收盘': 'pre_close',
        '加权平均叫卖价': 'avg_ask',
        '加权平均叫买价': 'avg_bid',
        '叫卖总量': 'total_ask',
        '叫买总量': 'total_bid'
    }
    
    for i in range(1, 11):
        col_map[f'申卖价{i}'] = f'askprice{i}'
        col_map[f'申卖量{i}'] = f'askvol{i}'
        col_map[f'申买价{i}'] = f'bidprice{i}'
        col_map[f'申买量{i}'] = f'bidvol{i}'

    snap_df = snap_df[list(col_map.keys())].rename(columns=col_map)

    # for snapshot data, all prices should div by 10000 to represent the real price
    price_cols = ['trade_price', 'high', 'low', 'open', 'pre_close', 'avg_ask', 'avg_bid'] + \
                 [f'askprice{i}' for i in range(1, 11)] + [f'bidprice{i}' for i in range(1, 11)]
    for col in price_cols:
        if col in snap_df.columns:
            snap_df[col] = snap_df[col] / 10000.0
    
    snap_df['trade_time'] = snap_df['trade_time'].astype(np.int64)

    return snap_df

def save_tick(pth: str = 'data/tick'):
    source_base = Path('E:/Data/etf/trade')
    target_base = Path(pth)
    target_base.mkdir(parents=True, exist_ok=True) 
    archive_files = list(source_base.rglob("*.7z")) 
    if not archive_files:
        return
    
    pbar = tqdm(archive_files, unit='file', dynamic_ncols=True)
    for archive_path in pbar:
        try:
            with py7zr.SevenZipFile(archive_path, mode='r') as archive:
                all_names = archive.getnames()
                target_names = [n for n in all_names if n.endswith("510300.csv")]
                
                if not target_names:
                    continue
                
                with tempfile.TemporaryDirectory() as tmp_dir:
                    archive.extract(targets=target_names, path=tmp_dir)

                    for internal_path in target_names:
                        extracted_file_path = Path(tmp_dir) / internal_path
                        if not extracted_file_path.exists():
                            continue
                    
                        norm_path = internal_path.replace('\\', '/')
                        if '/' in norm_path:
                            folder_part = norm_path.split('/')[0]
                            date_str = folder_part.replace('-', '').replace('_', '')
                        else:
                            date_str = archive_path.stem.replace('-', '').replace('_', '')

                        if len(date_str) > 8:
                            date_str = date_str[:8]

                        date_val = int(date_str) 
                        save_path = target_base / f"{date_str}.parquet"

                        df = pd.read_csv(extracted_file_path)
                        if not df.empty:
                            df_new = process_tick(df, date_val)
                            df_new.to_parquet(save_path, index=False, engine='pyarrow') 

        except Exception as e:
            tqdm.write(f"Error processing {archive_path.name}: {str(e)}")


def save_snap(pth: str = 'data/snap'):
    source_base = Path('E:/Data/etf/snapshot')
    target_base = Path(pth)
    target_base.mkdir(parents=True, exist_ok=True) 
    archive_files = list(source_base.rglob("*.7z")) 
    if not archive_files:
        return
    
    pbar = tqdm(archive_files, unit='file', dynamic_ncols=True)
    for archive_path in pbar:
        try:
            with py7zr.SevenZipFile(archive_path, mode='r') as archive:
                all_names = archive.getnames()
                target_names = [n for n in all_names if n.endswith("510300.csv")]
                
                if not target_names:
                    continue
                
                with tempfile.TemporaryDirectory() as tmp_dir:
                    archive.extract(targets=target_names, path=tmp_dir)

                    for internal_path in target_names:
                        extracted_file_path = Path(tmp_dir) / internal_path
                        if not extracted_file_path.exists():
                            continue
                    
                        norm_path = internal_path.replace('\\', '/')
                        if '/' in norm_path:
                            folder_part = norm_path.split('/')[0]
                            date_str = folder_part.replace('-', '').replace('_', '')
                        else:
                            date_str = archive_path.stem.replace('-', '').replace('_', '')

                        if len(date_str) > 8:
                            date_str = date_str[:8]

                        save_path = target_base / f"{date_str}.parquet"

                        df = pd.read_csv(extracted_file_path, encoding='gbk')
                        if not df.empty:
                            df_new = process_snap(df)
                            df_new.to_parquet(save_path, index=False, engine='pyarrow') 

        except Exception as e:
            tqdm.write(f"Error processing {archive_path.name}: {str(e)}")


if __name__ == '__main__':
    # save_tick('data/tick')             
    save_snap('data/snap')     