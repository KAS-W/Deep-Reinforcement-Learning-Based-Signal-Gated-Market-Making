import torch
import numpy as np
from torch import Tensor
from numpy import ndarray
from pandas import DataFrame, Series
from typing import Union
from utils.fast.oper import *

def _align_output(x, result, window):
    """fill nan into rolling results"""
    pad_size = window - 1
    if isinstance(x, Tensor):
        # fill Nan to head
        pad = torch.full((pad_size, *result.shape[1:]), float('nan'), device=x.device, dtype=x.dtype)
        return torch.cat([pad, result], dim=0)
    elif isinstance(x, ndarray):
        pad = np.full((pad_size, *result.shape[1:]), np.nan)
        return np.concatenate([pad, result], axis=0)
    # pandas has automatic alignment
    return result

def ts_rolling_mean(x, window: int):
    view = _get_rolling_view(x, window)
    if isinstance(x, Tensor):
        res = ts_mean(view, axis=1)
        return _align_output(x, res, window)
    elif isinstance(x, (DataFrame, Series)):
        return x.rolling(window=window).mean()
    else:
        res = ts_mean(view, axis=1)
        return _align_output(x, res, window)
    
def ts_rolling_std(x, window: int):
    view = _get_rolling_view(x, window)
    if isinstance(x, Tensor):
        res = ts_std(view, axis=1)
        return _align_output(x, res, window)
    elif isinstance(x, (DataFrame, Series)):
        return x.rolling(window=window).std()
    else:
        res = ts_std(view, axis=1)
        return _align_output(x, res, window)

def ts_rolling_max(x, window: int):
    view = _get_rolling_view(x, window)
    if isinstance(x, Tensor):
        res = ts_max(view, axis=1)
        return _align_output(x, res, window)
    elif isinstance(x, (DataFrame, Series)):
        return x.rolling(window=window).max()
    else:
        res = ts_max(view, axis=1)
        return _align_output(x, res, window)

def ts_rolling_min(x, window: int):
    view = _get_rolling_view(x, window)
    if isinstance(x, Tensor):
        res = ts_min(view, axis=1)
        return _align_output(x, res, window)
    elif isinstance(x, (DataFrame, Series)):
        return x.rolling(window=window).min()
    else:
        res = ts_min(view, axis=1)
        return _align_output(x, res, window)
    
def ts_rolling_slope(x, window: int):
    """
    Calculates rolling OLS slope. 
    
    Note: Pandas does not have a native rolling slope.
    We apply the static ts_slope to the windowed view.
    """
    view = _get_rolling_view(x, window)
    if isinstance(x, (Tensor, ndarray)):
        res = ts_slope(view, axis=1)
        return _align_output(x, res, window)
    elif isinstance(x, (DataFrame, Series)):
        # Apply static slope logic over the rolling window
        # extremyly slow, avoid pandas if you can
        return x.rolling(window).apply(lambda win: ts_slope(win.values))
    
def ts_delay(x: Union[Tensor, ndarray, DataFrame, Series], k: int):
    """
    Standard delay operator used for return calculations: r_t,k = ln(p_t / p_t-k)
    """
    if isinstance(x, (DataFrame, Series)):
        return x.shift(k)
    elif isinstance(x, Tensor):
        # Pad with NaNs to maintain length alignment
        out = torch.full_like(x, float('nan'), dtype=torch.float32)
        out[k:] = x[:-k]
        return out
    elif isinstance(x, ndarray):
        out = np.full_like(x, np.nan)
        out[k:] = x[:-k]
        return out
    
def _get_rolling_view(x: Union[Tensor, ndarray, DataFrame, Series], window: int):
    """
    Creates a rolling window view of the input data.
    """
    if isinstance(x, Tensor):
        # Result shape: [N - window + 1, window, Features]
        return x.unfold(dimension=0, size=window, step=1)
    elif isinstance(x, (DataFrame, Series)):
        return x.rolling(window=window)
    elif isinstance(x, ndarray):
        from numpy.lib.stride_tricks import sliding_window_view
        return sliding_window_view(x, window_shape=window, axis=0)
    else:
        raise WrongInputTypeError