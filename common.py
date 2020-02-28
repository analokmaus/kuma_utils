import os
import sys
import time
import datetime
import re
from pathlib import Path
from copy import deepcopy, copy
import traceback
import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd


def clean_value(x):
    if isinstance(x, str):
        if x.isnumeric():
            return float(x)
        else:
            return x
    elif isinstance(x, int):
        return float(x)
    elif isinstance(x, float):
        return x
    else:
        return x

    
class KumaNumpy:
    '''
    Enhanced numpy operation
    '''

    clean = np.vectorize(clean_value, otypes=[object])

    @classmethod
    def unique(self, x, return_counts=False):
        val_cnt = pd.Series(x).value_counts()
        val = val_cnt.index.values
        if return_counts:
            cnt = val_cnt.values
            return val, cnt
        else:
            return val
    
    @classmethod
    def nunique(self, x):
        return len(self.unique(x))

    @classmethod
    def to_numpy(self, x):
        assert isinstance(x, (np.ndarray, pd.DataFrame, pd.Series, list))

        if isinstance(x, pd.Series):
            return x.values
        elif isinstance(x, pd.DataFrame):
            return x.values
        elif isinstance(x, list):
            return np.array(x)
        else:
            return x

    @classmethod
    def to_numeric(self, x, dtypes=[np.float], verbose=False):
        if isinstance(dtypes, Iterable):
            for dtype in dtypes:
                # np.integer cannot handle nan
                if np.issubdtype(dtype, np.integer) and np.sum(x!=x) > 0:
                    continue

                try:
                    return x.astype(dtypes)
                except:
                    if verbose:
                        print(f'failed to transform: {dtype}')
                    pass
        else:
            if np.issubdtype(dtypes, np.integer) and np.sum(x != x) > 0:
                return x
                
            try:
                return x.astype(dtypes)
            except:
                if verbose:
                    print(f'failed to transform: {dtypes}')
                pass

        return x
        
    @classmethod
    def fillna(self, x, val):
        _x = x.copy()
        _x[_x!=_x] = val
        return _x

    @classmethod
    def dropna(self, x):
        return x[x==x]

    @classmethod
    def isin(self, x1, x2):
        return pd.Series(x1).isin(pd.Series(x2)).values

    @classmethod
    def replace(self, x, d):
        return pd.Series(x).map(d).values

    @classmethod
    def mode(self, x, **kwargs):
        return pd.Series(x).mode(**kwargs).values[0]
