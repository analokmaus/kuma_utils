import os
import sys
import time
import datetime
import argparse
import re
from pathlib import Path
from copy import deepcopy, copy
import traceback

import numpy as np
import pandas as pd

import warnings

from sklearn.metrics import roc_auc_score, confusion_matrix
from .preprocessing import DistTransformer


'''
Custom metrics
'''

class SeUnderSp(object):
    '''
    Maximize sensitivity under specific specificity threshold
    USAGE:
        # general
        SeUnderSp(0.9).test(target, pred)
        # catboost
        ..., eval_metric=SeUnderSp(0.9), ...
        # lightgbm
        ..., eval_metric=SeUnderSp(0.9).lgbm, ...
    '''
    def __init__(self, sp=0.9, maximize=True):
        self.sp = 0.9
        self.maximize = maximize

    def _get_threshold(self, target, approx):
        tn_idx = (target == 0)
        p_tn = np.sort(approx[tn_idx])

        return p_tn[int(len(p_tn) * self.sp)]

    def _get_se_sp(self, target, approx):
        if not isinstance(target, np.ndarray):
            target = np.array(target)
        if not isinstance(approx, np.ndarray):
            approx = np.array(approx)
        
        if min(approx) < 0:
            approx -= min(approx) # make all values positive
        target = target.astype(int)
        thres = self._get_threshold(target, approx)
        pred = (approx > thres).astype(int)
        tn, fp, fn, tp = confusion_matrix(target, pred).ravel()
        se = tp / (tp + fn)
        sp = tn / (tn + fp)

        return thres, se, sp

    def test(self, target, approx):
        if not self.maximize:
            return 1 - self._get_se_sp(target, approx)[1]
        else:
            return self._get_se_sp(target, approx)[1]

    '''
    CatBoost
    '''
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

        approx = np.array([v for v in approxes[0]])
        thres, se, sp = self._get_se_sp(target, approx)
        
        error_sum = se if self.maximize else 1 - se
        weight_sum = 1.0

        return error_sum, weight_sum

    '''
    Lightgbm
    '''
    def lgbm(self, target, approx):
        se = self._get_se_sp(target, approx)[1]
        if self.maximize:
            return 'se', se, self.maximize
        else:
            return '1-se', 1-se, self.maximize
