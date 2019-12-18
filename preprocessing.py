import os
import sys
import time
import datetime
import argparse
import re
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy, copy
import traceback

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.stats import ks_2samp

import warnings


'''
Misc.
'''


def reduce_mem_usage(df):
    """ 
    iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(
        100 * (start_mem - end_mem) / start_mem))

    return df


'''
Categorical encoder
'''

class CatEncoder:
    '''
    Scikit-learn API like categorical feature encoder
    USAGE:
        target_encoder = CatEncoder('target', noise=0.1)
        target_encoder.fit(x_train.feature , y_train)
        x_train.feature = target_encoder.transform(x_train.feature)
        x_test.feature = target_encoder.transform(x_test.feature)
    '''

    ENCODINGS = {'label', 'count', 'target'}

    def __init__(self, encoding='label', verbose=False, 
                 smoothing=False, noise=0, 
                 handle_missing='value', handle_unknown='value'):
        assert encoding in self.ENCODINGS
        self.encoding = encoding
        self.verbose = verbose
        self.smoothing = smoothing
        self.k = 0
        self.f = 200
        self.noise = noise
        self.handle_missing = handle_missing
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        x = self._all2array(X.copy())
        if y is not None:
            assert len(X) == len(y)
            y = self._all2array(y).copy()

        if self.encoding == 'label':
            self._label_encode(x, y)
        elif self.encoding == 'count':
            self._count_encode(x, y)
        elif self.encoding == 'target':
            self._target_encode(x, y)
        else:
            raise ValueError(self.encoding)

    def transform(self, X):
        x = self._all2array(X.copy())
        nan_idx = (x != x)
        common_idx = np.isin(x, np.array(list(self.encode_dict.keys())))

        x = self._replace(x, self.encode_dict)

        # deal with unknown class
        if self.handle_missing == 'return_nan':
            unknown_idx = ~common_idx & ~nan_idx
        elif self.handle_missing == 'value':
            unknown_idx = ~common_idx
        else:
            raise ValueError(self.handle_missing)

        if self.handle_unknown == 'value':
            if self.encoding == 'label':
                x[unknown_idx] = max(self.encode_dict.values()) + 1
            elif self.encoding == 'count':
                x[unknown_idx] = 0
            elif self.encoding == 'target':
                x[unknown_idx] = self.mean_target_all
            else:
                raise ValueError(self.encoding)
        elif self.handle_unknown == 'return_nan':
            x[unknown_idx] = np.nan
        else:
            raise ValueError(self.handle_unknown)

        x = self._to_numeric(x)
        
        return self._add_noise(x, self.noise)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def _all2array(self, x):
        assert isinstance(x, (np.ndarray, pd.DataFrame, pd.Series))

        if isinstance(x, pd.Series):
            x = x.values
        elif isinstance(x, pd.DataFrame):
            x = x.values[:, 0]
        else:
            pass
        
        if self.handle_missing == 'return_nan':
            return x.astype(object).squeeze()
        elif self.handle_missing == 'value':
            return x.astype(str).squeeze()  # nan -> 'nan'
        else:
            raise ValueError(self.handle_missing)

    def _sigmoid(self, x):
        '''Sigmoid like function (sigmoid when k=0 f=1)'''
        return 1 / (1 + np.exp((self.k - x)/self.f))

    def _to_numeric(self, x):
        if self.encoding == 'target':
            return x.astype(np.float16)
        else:
            try:
                return x.astype(np.int)
            except:
                return x.astype(np.float16)
    
    @staticmethod
    def _dropna(x):
        return x[x==x]

    @staticmethod
    def _replace(x, d):
        return np.array(
            [d[v] if v in d.keys() else v for v in x], dtype=object)

    @staticmethod
    def _add_noise(x, noise_level):
        return x * (1 + noise_level * np.random.randn(len(x)))
    
    def _label_encode(self, x_train, y_train):
        self.encode_dict = {}
        x_train = self._dropna(x_train)

        for new, prev in enumerate(np.sort(np.unique(x_train))):
            self.encode_dict[prev] = new

        if self.verbose:
            print('label encoder: fitting completed.')

    def _count_encode(self, x_train, y_train):
        self.encode_dict = {}
        x_train = self._dropna(x_train)

        values, counts = np.unique(x_train, return_counts=True)
        for prev, new  in zip(values, counts):
            self.encode_dict[prev] = new

        if self.verbose:
            print('count encoder: fitting completed.')

    def _target_encode(self, x_train, y_train):
        self.encode_dict = {}
        x_train = self._dropna(x_train)
        y_train = y_train.astype(np.float16)
        mean_target_all = np.mean(y_train)

        for val in np.unique(x_train):
            idx_class = x_train == val
            n_class = np.sum(idx_class)
            mean_target_class = np.mean(y_train[idx_class])
            if self.smoothing:
                w = self._sigmoid(n_class)
                smooth_target = w * mean_target_class + (1 - w) * mean_target_all
                self.encode_dict[val] = smooth_target
            else:
                self.encode_dict[val] = mean_target_class
        
        self.mean_target_all = mean_target_all

        if self.verbose:
            print('target encoder: fitting completed.')


'''
Distribution transformer
'''

class DistTransformer:
    '''
    Scikit-learn API like distribution transformer
    USAGE:
        scaler = DistTransformer(transform='rankgauss')
        X[col] = scaler.fit_transform(X[col])
    '''

    TRANSFORMS = {
        'standard', 'min-max', 
        'box-cox', 'yeo-johnson', 
        'rankgauss'
    }

    def __init__(self, transform='standard', verbose=False, ):
        assert transform in self.TRANSFORMS
        self.t = transform
        self.verbose = verbose

    def fit(self, X):
        x = self._all2array(X).copy().reshape(-1, 1)

        if self.t == 'standard':
            self.transformer = StandardScaler()
        elif self.t == 'min-max':
            self.transformer = MinMaxScaler()
        elif self.t == 'box-cox':
            self.transformer = PowerTransformer(method='box-cox')
        elif self.t == 'yeo-johnson':
            self.transformer = PowerTransformer(method='yeo-johnson')
        elif self.t == 'rankgauss':
            self.transformer = QuantileTransformer(
                n_quantiles=len(x), random_state=0,
                output_distribution='normal'
            )
        else:
            raise ValueError(self.transform)

        self.transformer.fit(x)

    def transform(self, X):
        x = self._all2array(X).copy().reshape(-1, 1)
        return self.transformer.transform(x)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    @staticmethod
    def _all2array(x):
        assert isinstance(x, (np.ndarray, pd.DataFrame, pd.Series))

        if isinstance(x, pd.Series):
            return x.values
        elif isinstance(x, pd.DataFrame):
            return x.values[:, 0]
        else:
            return x


'''
Multivariate Imputation by Chained Equations (MICE) with NA flag
'''

class MICE(IterativeImputer):

    def __init__(self, with_flag=True, **kwargs):
        super(MICE, self).__init__(**kwargs)
        self.with_flag = with_flag

    def fit(self, X, y=None):
        _X = self._add_nan_flag(X)
        super().fit(_X, y)

    def transform(self, X):
        y_size = X.shape[1]
        _X = self._add_nan_flag(X)
        return super().transform(_X)[:, :y_size]
        
    def fit_transform(self, X, y=None):
        y_size = X.shape[1]
        _X = self._add_nan_flag(X)
        return super().fit_transform(X, y)[:, :y_size]

    @staticmethod
    def _add_nan_flag(X):
        if isinstance(X, (pd.DataFrame)):
            _X = X.values
        elif isinstance(X, np.ndarray):
            _X = X
        else:
            raise TypeError(type(X))
        
        flags = np.zeros_like(_X, np.uint8)
        for c in range(_X.shape[1]):
            xc = _X[:, c]
            flags[:, c] = (xc != xc).astype(np.uint8)

        return np.concatenate([_X, flags], axis=1)     
