import pandas as pd
import numpy as np
from pathlib import Path
import py7zr
import tempfile
from tqdm import tqdm
import argparse

def process_tick(tick_df, date_val):
    tick_df['trade_date'] = np.int32(date_val)
    t = pd.to_datetime(tick_df['Time'], format='%H:%M:%S')
    tick_df['trade_time'] = t.dt.hour * 10000000 + t.dt.minute * 100000 + t.dt.second * 1000
    tick_df['side'] = tick_df['Type'].map({'B': 1, 'S': -1}).fillna(0).astype(np.int8)
    del tick_df['Time'], tick_df['Type'], t
    return tick_df

def process_snap(snap_df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        '自然日': 'trade_date', '时间': 'trade_time', '成交价': 'trade_price',
        '成交量': 'vol', '成交额': 'amt', '成交笔数': 'cnt',
        '当日累计成交量': 'cum_vol', '当日成交额': 'cum_amt',
        '最高价': 'high', '最低价': 'low', '开盘价': 'open',
        '前收盘': 'pre_close', '加权平均叫卖价': 'avg_ask',
        '加权平均叫买价': 'avg_bid', '叫卖总量': 'total_ask', '叫买总量': 'total_bid'
    }

    for i in range(1, 11):
        col_map[f'申卖价{i}'] = f'askprice{i}'
        col_map[f'申卖量{i}'] = f'askvol{i}'
        col_map[f'申买价{i}'] = f'bidprice{i}'
        col_map[f'申买量{i}'] = f'bidvol{i}'

    snap_df = snap_df[list(col_map.keys())].rename(columns=col_map)

    price_cols = ['trade_price', 'high', 'low', 'open', 'pre_close', 'avg_ask', 'avg_bid'] + \
                 [f'askprice{i}' for i in range(1, 11)] + [f'bidprice{i}' for i in range(1, 11)]
    for col in price_cols:
        if col in snap_df.columns:
            snap_df[col] = snap_df[col] / 10000.0
    
    snap_df['trade_time'] = snap_df['trade_time'].astype(np.int64)
    return snap_df

def extract_and_register(symbol, data_type='tick'):
    source_root = Path('E:/Data/etf/trade' if data_type == 'tick' else 'E:/Data/etf/snapshot')
    target_root = Path(f'data/{symbol}/{data_type}')
    target_root.mkdir(parents=True, exist_ok=True)

    archive_files = list(source_root.rglob("*.7z"))
    if not archive_files:
        print(f"Failed to find {data_type} zip in {source_root}")
        return
    
    print(f">>> Extracting {data_type} data from {symbol}:")
    for archive_path in tqdm(archive_files, leave=False):
        try:
            with py7zr.SevenZipFile(archive_path, mode='r') as archive:
                all_names = archive.getnames()
                target_names = [n for n in all_names if n.endswith(f"{symbol}.csv")]
                if not target_names: 
                    continue

                norm_path = target_names[0].replace('\\', '/')
                date_str = norm_path.split('/')[0].replace('-', '')[:8] if '/' in norm_path else archive_path.stem[:8]
                save_path = target_root / f"{date_str}.parquet"
                if save_path.exists(): 
                    continue

                with tempfile.TemporaryDirectory() as tmp_dir:
                    archive.extract(targets=target_names, path=tmp_dir)
                    file_path = Path(tmp_dir) / target_names[0]

                    if data_type == 'tick':
                        df = pd.read_csv(file_path)
                        if not df.empty:
                            process_tick(df, int(date_str)).to_parquet(save_path, index=False)
                    else:
                        df = pd.read_csv(file_path, encoding='gbk')
                        if not df.empty:
                            process_snap(df).to_parquet(save_path, index=False)
        except Exception as e:
            print(f"Error {archive_path.name} for: {str(e)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Asset Data Extraction")
    parser.add_argument("--symbol", type=str, required=True, help="asset codes like 510300")
    args = parser.parse_args()

    extract_and_register(args.symbol, data_type='tick')
    extract_and_register(args.symbol, data_type='snap')