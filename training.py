import os
import sys
import time
import datetime
import re
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy, copy
import traceback
import pickle
import random
from collections import Counter, defaultdict
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import seaborn as sns

from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from lightgbm import LGBMRegressor, LGBMClassifier, Dataset
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.metrics import roc_auc_score

try:
    import category_encoders as ce
    EXT_CE = True
except:
    EXT_CE = False
from .preprocessing import CatEncoder


'''
Training automation
'''

class Trainer:
    '''
    Make machine learning eazy again!

    # Usage
    model = Trainer(CatBoostClassifier(**CAT_PARAMS))
    model.train(x_train, y_train, x_valid, y_valid, fit_params={})
    '''

    MODELS = {
        'CatBoostRegressor', 'CatBoostClassifier', 
        'LGBMRegressor', 'LGBMClassifier',
        'RandomForestRegressor', 'RandomForestClassifier', 
        'LinearRegression', 'LogisticRegression', 
        'Ridge', 'Lasso',
        'SVR', 'SVC',
    }

    def __init__(self, model):
        model_type = type(model).__name__
        assert model_type in self.MODELS

        self.model = model
        self.model_type = model_type
    
    def train(self, X, y, X_valid=None, y_valid=None,
              cat_features=None, eval_metric=None, fit_params={}):

        if self.model_type[:8] == 'CatBoost':
            train_data = Pool(data=X, label=y, cat_features=cat_features)
            valid_data = Pool(data=X_valid, label=y_valid, cat_features=cat_features)
            self.model.fit(X=train_data, eval_set=valid_data, **fit_params)
            self.best_iteration = self.model.get_best_iteration()

        elif self.model_type[:4] == 'LGBM':
            if cat_features is None:
                cat_features = []
            # train_data = Dataset(data=X, label=y, categorical_feature=cat_features)
            # valid_data = Dataset(data=X_valid, label=y_valid, categorical_feature=cat_features)
            self.model.fit(X, y, eval_set=[(X, y), (X_valid, y_valid)], 
                           categorical_feature=cat_features, **fit_params)
            self.best_iteration = self.model.best_iteration_

        else:
            self.model.fit(X, y, **fit_params)
            self.best_iteration = -1

    def get_model(self):
        return self.model

    def get_best_iteration(self):
        return self.best_iteration

    def get_feature_importances(self):
        if self.model_type in ['CatBoostRegressor', 'CatBoostClassifier',
                               'LGBMRegressor', 'LGBMClassifier',
                               'RandomForestRegressor', 'RandomForestClassifier']:
            return self.model.feature_importances_
        elif self.model_type in ['LinearRegression', 'LogisticRegression',
                                 'Ridge', 'Lasso']:
            return self.model.coef_
        elif self.model_type in ['SVR', 'SVC']:
            if self.model.get_params()['kernel'] == 'linear':
                return self.model.coef_
            else:
                return 0
        else:
            return 0
    
    def get_params(self):
        if self.model_type[:8] == 'CatBoost':
            print(model.get_params())
        else:
            print('')

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def plot_feature_importances(self, columns):
        plt.figure(figsize=(5, int(len(columns) / 3)))
        imps = self.get_feature_importances()
        order = np.argsort(imps)
        colors = colormap.winter(np.arange(len(columns))/len(columns))
        plt.barh(np.array(columns)[order], imps[order], color=colors)
        plt.show()
        

class CrossValidator:
    '''
    Make cross validation beautiful again?
    
    # Usage
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cat_cv = CrossValidator(CatBoostClassifier(**CAT_PARAMS), skf)
    cat_cv.run(
        X, y, x_test, 
        eval_metric=roc_auc_score, prediction='predict', 
        train_params={'cat_features': CAT_IDXS, 'fit_params': CAT_FIT_PARAMS},
        verbose=0
    )
    '''

    def __init__(self, model, datasplit):
        self.basemodel = copy(model)
        self.datasplit = datasplit
        self.models = []
        self.oof = None
        self.pred = None
        self.imps = None

    @staticmethod
    def binary_proba(model, X):
        return model.predict_proba(X)[:, 1]

    @staticmethod
    def predict(model, X):
        return model.predict(X)

    def run(self, X, y, X_test=None, 
            group=None, n_splits=None, 
            eval_metric=None, prediction='predict',
            transform=None, train_params={}, verbose=True):

        if not isinstance(eval_metric, (list, tuple, set)):
            eval_metric = [eval_metric]

        if n_splits is None:
            K = self.datasplit.get_n_splits()
        else:
            K = n_splits
        self.oof = np.zeros(len(X), dtype=np.float)
        if X_test is not None:
            self.pred = np.zeros(len(X_test), dtype=np.float)
            
        self.imps = np.zeros((X.shape[1], K))
        self.scores = np.zeros((len(eval_metric), K))

        for fold_i, (train_idx, valid_idx) in enumerate(
            self.datasplit.split(X, y, group)):

            x_train, x_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]
            if X_test is not None:
                x_test = X_test.copy()

            if transform is not None:
                x_train, x_valid, y_train, y_valid, x_test = transform(
                    Xs=(x_train, x_valid), ys=(y_train, y_valid), 
                    X_test=x_test)

            if verbose > 0:
                print(f'\n-----\n {K} fold cross validation. \n Starting fold {fold_i+1}\n-----\n')
                print(f'[CV]train: {len(train_idx)} / valid: {len(valid_idx)}')
            if verbose <= 0 and 'fit_params' in train_params.keys():
                train_params['fit_params']['verbose'] = 0
            model = Trainer(copy(self.basemodel))
            model.train(x_train, y_train, x_valid, y_valid, **train_params)
            self.models.append(model.get_model())

            if verbose > 0:
                print(f'best iteration is {model.get_best_iteration()}')

            if prediction == 'predict':
                self.oof[valid_idx] = self.predict(model, x_valid)
            elif prediction == 'binary_proba':
                self.oof[valid_idx] = self.binary_proba(model, x_valid)
            else:
                self.oof[valid_idx] = self.predict(model, x_valid)

            if X_test is not None:
                if prediction == 'predict':
                    self.pred += self.predict(model, x_test) / K
                elif prediction == 'binary_proba':
                    self.pred += self.binary_proba(model, x_test) / K
                else:
                    self.pred += self.predict(model, x_test) / K
            
            self.imps[:, fold_i] = model.get_feature_importances()
            
            for i, _metric in enumerate(eval_metric):
                score = _metric(y_valid, self.oof[valid_idx])
                self.scores[i, fold_i] = score
            
            if verbose >= 0:
                log_str = f'[CV] Fold {fold_i}:'
                log_str += ''.join([f' m{i}={self.scores[i, fold_i]:.5f}' for i in range(len(eval_metric))])
                log_str += f' (iter {model.get_best_iteration()})'
                print(log_str)

        log_str = f'[CV] Overall:'
        log_str += ''.join(
            [f' m{i}={me:.5f}±{se:.5f}' for i, (me, se) in enumerate(zip(
                np.mean(self.scores, axis=1), 
                np.std(self.scores, axis=1)/np.sqrt(len(eval_metric))
            ))]
        )
        print(log_str)
        
    def plot_feature_importances(self, columns=None):
        if columns is None:
            columns = [f'feature_{i}' for i in range(len(self.imps))]
        plt.figure(figsize=(5, int(len(columns) / 3)))
        imps_mean = np.mean(self.imps, axis=1)
        imps_se = np.std(self.imps, axis=1) / np.sqrt(self.imps.shape[0])
        order = np.argsort(imps_mean)
        colors = colormap.winter(np.arange(len(columns))/len(columns))
        plt.barh(np.array(columns)[order],
                imps_mean[order], xerr=imps_se[order], color=colors)
        plt.show()

    def save_feature_importances(self, path, columns=None):
        if columns is None:
            columns = [f'feature_{i}' for i in range(len(self.imps))]
        plt.figure(figsize=(5, int(len(columns) / 3)))
        imps_mean = np.mean(self.imps, axis=1)
        imps_se = np.std(self.imps, axis=1) / np.sqrt(self.imps.shape[0])
        order = np.argsort(imps_mean)
        colors = colormap.winter(np.arange(len(columns))/len(columns))
        plt.barh(np.array(columns)[order],
                 imps_mean[order], xerr=imps_se[order], color=colors)
        plt.savefig(path)

    def save(self, path):
        objects = [
            self.basemodel, self.datasplit, 
            self.models, self.oof, self.pred, self.imps
        ]
        with open(path, 'wb') as f:
            pickle.dump(objects, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            objects = pickle.load(f)
        
        self.basemodel, self.datasplit, self.models, \
            self.oof, self.pred, self.imps = objects


class InfoldTargetEncoder:
    '''
    Target encoder w/o target leak
    '''

    def __init__(self, categorical_features, encoder=None):
        if encoder is None:
            if EXT_CE:
                self.encoder = ce.TargetEncoder(cols=np.arange(len(categorical_features)), 
                                                return_df=False)
            else:
                self.encoder = CatEncoder(encoding='target')
        else:
            self.encoder = encoder
        self.cat_idx = categorical_features

    def __call__(self, Xs, ys, X_test=None):
        assert len(Xs) == len(ys)
            
        self.encoder.fit(Xs[0][:, self.cat_idx], ys[0])
        for i in range(len(Xs)):
            Xs[i][:, self.cat_idx] = self.encoder.transform(
                Xs[i][:, self.cat_idx])
        if X_test is not None:
            X_test[:, self.cat_idx] = self.encoder.transform(
                X_test[:, self.cat_idx])
        
        return (*Xs, *ys, X_test)


'''
Feature selection
'''

class AdvFeatureSelection:
    '''
    Adversarial feature selection
    
    # Usage
    adv = AdvFeatureSelection(LogisticRegression(), X, y)
    adv.run(columns=X.columns, verbose=1)
    adv.get_importance_features(index=False)
    '''

    def __init__(self, model, X, y, eval_metric=roc_auc_score):
        self.X = X
        self.y = y

        self.model = Trainer(model)
        self.model.train(self.X, self.y)
        self.eval_metric = eval_metric
        self.basescore = self.eval_metric(self.y, self.model.predict_proba(self.X)[:, 1])

    def run(self, target=None, columns=None, verbose=False):
        if columns is None:
            columns = np.arange(self.X.shape[1])
        else:
            if not isinstance(columns, np.ndarray):
                columns = np.array(columns)
            assert len(columns) == self.X.shape[1]
        
        if target is None:
            target = columns
        else:
            if not isinstance(target, np.ndarray):
                target = np.array(target)
            assert len(target) <= len(columns)

        self.columns = columns
        target = np.where(np.isin(columns, target))[0]

        improvements = []
        for icol in target:
            _X = np.delete(self.X, icol, axis=1)
            self.model.train(_X, self.y)
            score = self.eval_metric(self.y, self.model.predict_proba(_X)[:, 1])
            delta = self.basescore - score
            improvements.append(delta)
            good = '*' if delta > 0 else ' '
            if verbose:
                print(f'{good} {icol}_{columns[icol]}: {delta:.5f}')

        self.improvements = improvements

    def get_importance_features(self, n=5, index=False):
        if index:
            return np.argsort(self.improvements)[::-1][:n]
        else:
            return self.columns[np.argsort(self.improvements)[::-1][:n]]


'''
Model selection
'''

class StratifiedGroupKFold:
    '''
    scikit-learn like inplementation of StratifiedGroupKFold
    '''

    def __init__(self, n_splits, random_state=None):

        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X, y, groups):

        labels_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        for label, g in zip(y, groups):
            y_counts_per_group[g][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        def eval_y_counts_per_fold(y_counts, fold):
            y_counts_per_fold[fold] += y_counts
            std_per_label = []
            for label in range(labels_num):
                label_std = np.std(
                    [y_counts_per_fold[i][label] / y_distr[label] for i in range(self.n_splits)])
                std_per_label.append(label_std)
            y_counts_per_fold[fold] -= y_counts
            return np.mean(std_per_label)

        groups_and_y_counts = list(y_counts_per_group.items())
        random.Random(self.random_state).shuffle(groups_and_y_counts)

        for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for i in range(self.n_splits):
                fold_eval = eval_y_counts_per_fold(y_counts, i)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(g)

        all_groups = set(groups)
        for i in range(self.n_splits):
            train_groups = all_groups - groups_per_fold[i]
            test_groups = groups_per_fold[i]

            train_indices = [i for i, g in enumerate(
                groups) if g in train_groups]
            test_indices = [i for i, g in enumerate(
                groups) if g in test_groups]

            yield train_indices, test_indices
