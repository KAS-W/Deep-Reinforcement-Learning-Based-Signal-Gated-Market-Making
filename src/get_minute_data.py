import os
import pandas as pd
import time 
from tqdm import tqdm
from utils.data_utils.tusharesql import AShareQueryTool

def _process_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    Process DataFrame into correct format:
    - remove `ts_code`
    - turn `trade_date` into `int64`
    """
    data = df.copy(deep=True)
    data.drop(columns=['ts_code'], inplace=True, errors='ignore')
    data['trade_time'] = pd.to_datetime(data['trade_time']).dt.strftime('%Y%m%d%H%M%S').astype('int64')
    data.sort_values(by='trade_time', ascending=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    del df
    return data

def get_1min_data(ts_code, start_date, end_date):
    """Get 1min A-Share data from Tushare"""
    query = AShareQueryTool()

    # parse dirs
    symbol = ts_code.split('.')[0]

    # get all trade days within the whold period
    trade_cal = query.trade_cal(start_date=start_date, end_date=end_date, is_open='1')
    trade_days = trade_cal['cal_date'].tolist()
    if not trade_days:
        print(f'Error: No trade days between {start_date} and {end_date}')
        return 

    for trade_date in tqdm(trade_days, dynamic_ncols=True):
        # must indicate the hour-minute for each day
        # otherwise, there will be no data: 09:00:00 - 09:00:00 has no increments
        start_ts = f"{trade_date} 09:00:00"
        end_ts = f"{trade_date} 16:00:00"
        try:
            slice_df = query.stk_mins(ts_code=ts_code, start_date=start_ts, end_date=end_ts)

            # if slice_df is not None and not slice_df.empty:
                # check validation for each day
                # expected_columns = ['close', 'open', 'high', 'low', 'vol', 'amount']
                # for col in expected_columns:
                #     assert col in slice_df.columns, f"Error: Missing expected column '{col}'"
            # print(f"DEBUG: {trade_date} | Rows: {len(slice_df) if slice_df is not None else 'None'}")
            if slice_df is None or slice_df.empty:
                continue

            expected_columns = ['close', 'open', 'high', 'low', 'vol', 'amount']
            if not all(col in slice_df.columns for col in expected_columns):
                print(f"Warning: {trade_date} columns mismatch.")
                continue

            slice_df_with_correct_format = _process_data(slice_df)

            name = f'{symbol}_{trade_date}.parquet'
            regis_path = os.path.join('data', '1min_base', name)
            os.makedirs(os.path.dirname(regis_path), exist_ok=True)
            slice_df_with_correct_format.to_parquet(regis_path)

        except Exception as e:
            print(f"Error on {trade_date}: {e}")
            continue

        finally:
            if 'slice_df_with_correct_format' in locals(): del slice_df_with_correct_format

        time.sleep(0.2)


if __name__ == '__main__':
    get_1min_data('510300.SH', '20200101', '20251231')