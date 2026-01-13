import torch
import numpy as np
from torch import Tensor
from numpy import ndarray
from pandas import DataFrame, Series
from typing import Union
from utils.fast.oper import *

def ts_rolling_mean(x, window: int):
    view = _get_rolling_view(x, window)
    if isinstance(x, Tensor):
        return ts_mean(view, axis=-1)
    elif isinstance(x, (DataFrame, Series)):
        return view.mean()
    else:
        return ts_mean(view, axis=-1)
    
def ts_rolling_std(x, window: int):
    view = _get_rolling_view(x, window)
    if isinstance(x, Tensor):
        return ts_std(view, axis=-1)
    elif isinstance(x, (DataFrame, Series)):
        return view.std()
    else:
        return ts_std(view, axis=-1)
    
def ts_rolling_max(x, window: int):
    view = _get_rolling_view(x, window)
    if isinstance(x, Tensor):
        return ts_max(view, axis=-1)
    elif isinstance(x, (DataFrame, Series)):
        return view.max()
    else:
        return ts_max(view, axis=-1)
    
def ts_rolling_min(x, window: int):
    view = _get_rolling_view(x, window)
    if isinstance(x, Tensor):
        return ts_min(view, axis=-1)
    elif isinstance(x, (DataFrame, Series)):
        return view.min()
    else:
        return ts_min(view, axis=-1)
    
def ts_rolling_slope(x, window: int):
    """
    Calculates rolling OLS slope. 
    
    Note: Pandas does not have a native rolling slope.
    We apply the static ts_slope to the windowed view.
    """
    view = _get_rolling_view(x, window)
    if isinstance(x, (Tensor, ndarray)):
        return ts_slope(view)
    elif isinstance(x, (DataFrame, Series)):
        # Apply static slope logic over the rolling window
        return view.apply(lambda win: ts_slope(win.values))
    
def ts_delay(x: Union[Tensor, ndarray, DataFrame, Series], k: int):
    """
    Standard delay operator used for return calculations: r_t,k = ln(p_t / p_t-k)
    """
    if isinstance(x, (DataFrame, Series)):
        return x.shift(k)
    elif isinstance(x, Tensor):
        # Pad with NaNs to maintain length alignment
        out = torch.full_like(x, float('nan'))
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