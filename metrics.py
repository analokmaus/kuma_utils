import os
import sys
import time
import datetime
import argparse
import re
from pathlib import Path
from copy import deepcopy, copy
import traceback
import warnings

import numpy as np
from numba import jit
import pandas as pd

from sklearn.metrics import roc_auc_score, confusion_matrix, mean_squared_error
from .preprocessing import DistTransformer


class MetricTemplate:
    '''
    Custom metric template

    # Usage
    general:    Metric()(target, approx)
    catboost:   eval_metric=Metric()
    lightgbm:   eval_metric=Metric().lgbm
    pytorch:    Metric().torch(output, labels)
    '''

    def __init__(self, maximize=False):
        self.maximize = maximize

    def __repr__(self):
        return f'{type(self).__name__}(maximize={self.maximize})'

    @jit
    def _test(self, target, approx):
        # Metric calculation
        pass

    def __call__(self, target, approx):
        return self._test(target, approx)

    ### CatBoost
    def get_final_error(self, error, weight):
        return error / weight

    def is_max_optimal(self):
        return self.maximize

    def evaluate(self, approxes, target, weight=None):
        # approxes - list of list-like objects (one object per approx dimension)
        # target - list-like object
        # weight - list-like object, can be None
        assert len(approxes[0]) == len(target)
        if not isinstance(target, np.ndarray):
            target = np.array(target)

        approx = np.array(approxes[0])
        error_sum = self._test(target, approx)
        weight_sum = 1.0

        return error_sum, weight_sum
    
    ### LightGBM
    def lgbm(self, target, approx):
        return self.__class__.__name__, self._test(target, approx), self.maximize

    ### PyTorch
    def torch(self, approx, target):
        return self._test(target.detach().cpu().numpy(),
                          approx.detach().cpu().numpy())


class SeUnderSp(MetricTemplate):
    '''
    Maximize sensitivity under specific specificity threshold
    '''
    def __init__(self, sp=0.9, maximize=True):
        self.sp = 0.9
        self.maximize = maximize

    def _get_threshold(self, target, approx):
        tn_idx = (target == 0)
        p_tn = np.sort(approx[tn_idx])

        return p_tn[int(len(p_tn) * self.sp)]

    def _test(self, target, approx):
        if not isinstance(target, np.ndarray):
            target = np.array(target)
        if not isinstance(approx, np.ndarray):
            approx = np.array(approx)

        if len(approx.shape) == 1:
            pass
        elif approx.shape[1] == 1:
            approx = np.squeeze(approx)
        elif approx.shape[1] == 2:
            approx = approx[:, 1]
        else:
            raise ValueError(f'Invalid approx shape: {approx.shape}')

        if min(approx) < 0:
            approx -= min(approx) # make all values positive
        target = target.astype(int)
        thres = self._get_threshold(target, approx)
        pred = (approx > thres).astype(int)
        tn, fp, fn, tp = confusion_matrix(target, pred).ravel()
        se = tp / (tp + fn)
        sp = tn / (tn + fp)

        return se


class RMSE(MetricTemplate):
    '''
    Root mean square error
    '''
    def _test(self, target, approx):
        return np.sqrt(mean_squared_error(target, approx))


class AUC(MetricTemplate):
    '''
    Area under ROC curve
    '''
    def __init__(self, maximize=True):
        self.maximize = maximize

    def _test(self, target, approx):
        if len(approx.shape) == 1:
            approx = approx
        elif approx.shape[1] == 1:
            approx = np.squeeze(approx)
        elif approx.shape[1] == 2:
            approx = approx[:, 1]
        else:
            raise ValueError(f'Invalid approx shape: {approx.shape}')
        return roc_auc_score(target, approx)
        

class Accuracy(MetricTemplate):
    '''
    Accuracy
    '''

    def __init__(self, maximize=True):
        self.maximize = maximize

    def _test(self, target, approx):
        assert(len(target) == len(approx))
        target = np.asarray(target, dtype=int)
        approx = np.asarray(approx, dtype=float)
        if len(approx.shape) == 1:
            approx = approx
        elif approx.shape[1] == 1:
            approx = np.squeeze(approx)
        elif approx.shape[1] >= 2:
            approx = np.argmax(approx, axis=1)
        approx = approx.round().astype(int)
        return np.mean((target == approx).astype(int))


class QWK(MetricTemplate):
    '''
    Quandric Weight Kappa :))
    '''

    def __init__(self, max_rat, maximize=True):
        self.max_rat = max_rat
        self.maximize = maximize

    def _test(self, target, approx):
        assert(len(target) == len(approx))
        target = np.asarray(target, dtype=int)
        approx = np.asarray(approx, dtype=float)
        if len(approx.shape) == 1:
            approx = approx
        elif approx.shape[1] == 1:
            approx = np.squeeze(approx)
        elif approx.shape[1] >= 2:
            approx = np.argmax(approx, axis=1)
        approx = np.clip(approx.round(), 0, self.max_rat-1).astype(int)

        hist1 = np.zeros((self.max_rat+1, ))
        hist2 = np.zeros((self.max_rat+1, ))

        o = 0
        for k in range(target.shape[0]):
            i, j = target[k], approx[k]
            hist1[i] += 1
            hist2[j] += 1
            o += (i - j) * (i - j)

        e = 0
        for i in range(self.max_rat + 1):
            for j in range(self.max_rat + 1):
                e += hist1[i] * hist2[j] * (i - j) * (i - j)

        e = e / target.shape[0]

        return 1 - o / e
