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
from scipy.stats import ks_2samp

import warnings


'''
Misc.
'''

def KS_test(train, test, plot_rejected=False, plot_accepted=False, thres=0.05):
    '''
    Kolmogorov-Smirnov test
    '''
    assert isinstance(train, pd.DataFrame)
    assert type(train) == type(test)

    h0_accepted = []
    h0_rejected = []
    
    for col in test.columns:
        d, p = ks_2samp(train[col], test[col])

        if p >= thres:
            h0_accepted.append(col)
            if plot_accepted:
                plt.figure(figsize=(6, 3))
                plt.title("Kolmogorov-Smirnov test\n"
                          "feature: {}, statistics: {:.5f}, pvalue: {:5f}".format(col, d, p))
                sns.kdeplot(train[col], color='blue', shade=True, label='Train')
                sns.kdeplot(test[col], color='green', shade=True, label='Test')
                plt.show()
        else:  # pvalue < thres
            h0_rejected.append(col)
            if plot_rejected:
                plt.figure(figsize=(6, 3))
                plt.title("Kolmogorov-Smirnov test\n"
                          "feature: {}, statistics: {:.5f}, pvalue: {:5f}".format(col, d, p))
                sns.kdeplot(train[col], color='blue', shade=True, label='Train')
                sns.kdeplot(test[col], color='green', shade=True, label='Test')
                plt.show()

    return h0_accepted, h0_rejected


'''
Categorical encoder
'''

class CatEncoder:
    '''
    Scikit-learn API like categorical feature encoder
    USAGE:
        target_encoder = CatEncoder('target', noise_level=0.5)
        target_encoder.fit(x_train.feature , y_train)
        x_train.feature = target_encoder.transform(x_train.feature)
        x_test.feature = target_encoder.transform(x_test.feature)
    '''

    ENCODINGS = {'label', 'count', 'target'}


    def __init__(self, encoding='label', verbose=False, noise_level=0):
        assert encoding in self.ENCODINGS
        self.encoding = encoding
        self.verbose = verbose
        self.noise_level = noise_level


    def fit(self, X, y=None):
        x = self._all2array(X).copy()
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

    
    def transfrom(self, X):
        x = self._all2array(X).copy()
        common_idx = np.isin(x, np.array(list(self.encode_dict.keys())))

        x = self._replace(x, self.encode_dict)

        if self.encoding == 'label':
            x[~common_idx] = max(self.encode_dict.values()) + 1
        elif self.encoding == 'count':
            x[~common_idx] = 0
        elif self.encoding == 'target':
            x[~common_idx] = self.target_mean
        else:
            raise ValueError(self.encoding)

        return self._add_noise(x, self.noise_level)


    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transfrom(X)

    
    @staticmethod
    def _all2array(x):
        assert isinstance(x, (np.ndarray, pd.DataFrame, pd.Series))

        if isinstance(x, pd.Series):
            return x.values
        elif isinstance(x, pd.DataFrame):
            return x.values[:, 0]
        else:
            return x

    @staticmethod
    def _replace(x, d):
        return np.array([ d[v] if v in d.keys() else v for v in x ])

    @staticmethod
    def _add_noise(x, noise_level):
        return x * (1 + noise_level * np.random.randn(len(x)))


    def _label_encode(self, x_train, y_train):
        self.encode_dict = {}

        for new, prev in enumerate(np.sort(np.unique(x_train))):
            self.encode_dict[prev] = new

        if self.verbose:
            print('label encoder: fitting completed.')


    def _count_encode(self, x_train, y_train):
        self.encode_dict = {}

        values, counts = np.unique(x_train, return_counts=True)
        for prev, new  in zip(values, counts):
            self.encode_dict[prev] = new

        if self.verbose:
            print('count encoder: fitting completed.')


    def _target_encode(self, x_train, y_train):
        self.encode_dict = {}

        for prev in np.unique(x_train):
            labels = y_train[x_train == prev]
            self.encode_dict[prev] = np.mean(labels)
        self.target_mean = np.mean(y_train)

        if self.verbose:
            print('target encoder: fitting completed.')


'''
Distribution transformer
'''

class DistTransformer:
    '''
    Scikit-learn API like distribution transformer
    USAGE:
        
    '''

    TRANSFORMS = {
        'standardization', 'min-max', 
        'box-cox', 'yeo-johnson', 
        'rankgauss'
    }

    def __init__(self, transform='label', verbose=False, ):
        assert transform in self.TRANSFORMS
        self.t = transform
        self.verbose = verbose


    def fit(self, X):
        x = self._all2array(X).copy().reshape(-1, 1)

        if self.t == 'standardization':
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

