import torch
import numpy as np
from torch import Tensor
from numpy import ndarray
from pandas import DataFrame, Series
from typing import Union

class WrongInputTypeError(Exception):
    pass

def ts_log(x: Union[Tensor, ndarray, DataFrame, Series]) -> Union[Tensor, ndarray, DataFrame, Series]:
    """
    Calculate the log on a time series
    """
    if isinstance(x, Tensor):
        return torch.log(x)
    elif isinstance(x, (ndarray, DataFrame, Series)):
        return np.log(x)
    else:
        raise WrongInputTypeError
    
def ts_sub(x: Union[Tensor, ndarray, DataFrame, Series], 
           y: Union[Tensor, ndarray, DataFrame, Series]) -> Union[Tensor, ndarray, DataFrame, Series]:
    if isinstance(x, Tensor):
        return torch.sub(x, y)
    elif isinstance(x, (ndarray, DataFrame, Series)):
        return np.subtract(x, y)
    else:
        raise WrongInputTypeError
    
def ts_div(x: Union[Tensor, ndarray, DataFrame, Series], 
           y: Union[Tensor, ndarray, DataFrame, Series]) -> Union[Tensor, ndarray, DataFrame, Series]:
    if isinstance(x, Tensor):
        return torch.div(x, y)
    elif isinstance(x, (ndarray, DataFrame, Series)):
        return np.divide(x, y)
    else:
        raise WrongInputTypeError
    
def ts_std(x: Union[Tensor, ndarray, DataFrame, Series], axis=0) -> Union[Tensor, ndarray, DataFrame, Series]:
    if isinstance(x, Tensor):
        return torch.std(x, dim=axis, unbiased=True)
    elif isinstance(x, (ndarray, DataFrame, Series)):
        return np.std(x, axis=axis, ddof=1)
    else:
        raise WrongInputTypeError
    
def ts_max(x: Union[Tensor, ndarray, DataFrame, Series], axis=0) -> Union[Tensor, ndarray, DataFrame, Series]:
    if isinstance(x, Tensor):
        return torch.max(x, dim=axis)[0]
    elif isinstance(x, (ndarray, DataFrame, Series)):
        return np.max(x, axis=axis)
    else:
        raise WrongInputTypeError
    
def ts_min(x: Union[Tensor, ndarray, DataFrame, Series], axis=0) -> Union[Tensor, ndarray, DataFrame, Series]:
    if isinstance(x, Tensor):
        return torch.min(x, dim=axis)[0]
    elif isinstance(x, (ndarray, DataFrame, Series)):
        return np.min(x, axis=axis)
    else:
        raise WrongInputTypeError
    
def ts_mean(x: Union[Tensor, ndarray, DataFrame, Series], axis=0) -> Union[Tensor, ndarray, DataFrame, Series]:
    if isinstance(x, Tensor):
        return torch.mean(x, dim=axis)
    elif isinstance(x, (ndarray, DataFrame, Series)):
        return np.mean(x, axis=axis)
    else:
        raise WrongInputTypeError
    
def ts_slope(x: Union[Tensor, ndarray, DataFrame, Series]) -> Union[Tensor, ndarray, DataFrame, Series]:
    n = x.shape[0]
    # generate time-sereis variable t = [0, 1, 2, ..., n-1]
    if isinstance(x, Tensor):
        t = torch.arange(n, device=x.device, dtype=x.dtype).view(-1, 1)
        # OLS: (t't)^-1 t'x
        t_mean = torch.mean(t)
        x_mean = torch.mean(x, dim=0)
        numerator = torch.sum((t - t_mean) * (x - x_mean), dim=0)
        denominator = torch.sum((t - t_mean)**2)
        return numerator / denominator
    elif isinstance(x, (ndarray, DataFrame, Series)):
        t = np.arange(n)
        t_mean = np.mean(t)
        x_mean = np.mean(x, axis=0)
        num = np.sum((t - t_mean)[:, None] * (x - x_mean), axis=0)
        den = np.sum((t - t_mean)**2)
        return num / den