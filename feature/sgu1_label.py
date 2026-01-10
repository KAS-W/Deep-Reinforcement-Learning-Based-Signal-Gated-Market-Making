import pandas as pd 
from utils.fast.oper import ts_log, ts_sub
from utils.fast.calculator import ts_rolling_max, ts_rolling_min

def generate_sgu1_labels(df: pd.DataFrame, k: int=10):
    """
    Generate labels for SGU1

    RR_t,k = ln(max(H_{t+1...t+k}) / min(L_{t+1...t+k}))
    """
    # we need the max/min of the NEXT k periods
    # achieve this by calculating rolling max/min and shifting backward
    # Calculate rolling max of High and rolling min of Low
    # window=k, but these are centered at t

    fwd_high = ts_rolling_max(df['high'], window=k)
    fwd_low = ts_rolling_min(df['low'], window=k)

    # Shift backward by k periods to align future info with current time t
    # After shifting, target[t] contains the max/min from [t+1, t+k]

    target_high = fwd_high.shift(-k)
    target_low = fwd_low.shift(-k)

    # Calculate Log Realized Range: ln(H_max) - ln(L_min)

    log_h = ts_log(target_high)
    log_l = ts_log(target_low)

    label = ts_sub(log_h, log_l)

    return label.rename(f'target_rr_{k}')