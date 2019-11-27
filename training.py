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
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from lightgbm import LGBMRegressor, LGBMClassifier, Dataset
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso


'''
Training automation
'''

class Trainer:
    '''
    Make machine learning eazy again!
    '''

    MODELS = {
        'CatBoostRegressor', 'CatBoostClassifier', 
        'LGBMRegressor', 'LGBMClassifier',
        'RandomForestRegressor', 'RandomForestClassifier', 
        'LinearRegression', 'LogisticRegression', 
        'Ridge', 'Lasso',
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
            # train_data = Dataset(data=X, label=y, categorical_feature=cat_features)
            # valid_data = Dataset(data=X_valid, label=y_valid, categorical_feature=cat_features)
            self.model.fit(X, y, eval_set=[(X, y), (X_valid, y_valid)], 
                           categorical_feature=cat_features, **fit_params)
            self.best_iteration = self.model.best_iteration_

        else:
            self.model.fit(X, y)
            self.best_iteration = -1


    def get_model(self):
        return self.model


    def get_best_iteration(self):
        return self.best_iteration

    
    def get_feature_importances(self):
        try:
            return self.model.feature_importances_
        except:
            print('this model does not have feature importances.')
            return 0


    def predict(self, X):
        return self.model.predict(X)


    def predict_proba(self, X):
        return self.model.predict_proba(X)
        

class CrossValidator:
    '''
    Make cross validation beautiful again?
    '''

    def __init__(self, model, datasplit):
        self.basemodel = copy(model)
        self.datasplit = datasplit
        self.models = []
        self.oofs = None
        self.preds = None
        self.imps = None


    @staticmethod
    def binary_proba(model, X):
        return model.predict_proba(X)[:, 1]


    @staticmethod
    def predict(model, X):
        return model.predict(X)


    def run(self, X, y, X_test=None, 
            eval_metric=None, prediction='predict',
            train_params={}, verbose=True):

        K = self.datasplit.n_splits
        self.oofs = np.zeros(len(X), dtype=np.float)
        if X_test is not None:
            self.preds = np.zeros(len(X_test), dtype=np.float)
        self.imps = np.zeros((X.shape[1], K))
        self.scores = np.zeros(K)

        for fold_i, (train_idx, valid_idx) in enumerate(
            self.datasplit.split(X, y)):

            x_train, x_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            if verbose:
                print(f'\n-----\n {K} fold cross validation. \n Starting fold {fold_i+1}\n-----\n')
                print(f'train: {len(train_idx)} / valid: {len(valid_idx)}')
            
            model = Trainer(copy(self.basemodel))
            model.train(x_train, y_train, x_valid, y_valid, **train_params)
            self.models.append(model.get_model())

            if verbose:
                print(f'best iteration is {model.get_best_iteration()}')

            if prediction == 'predict':
                self.oofs[valid_idx] = self.predict(model, x_valid)
            elif prediction == 'binary_proba':
                self.oofs[valid_idx] = self.binary_proba(model, x_valid)
            else:
                self.oofs[valid_idx] = self.predict(model, x_valid)

            if X_test is not None:
                if prediction == 'predict':
                    self.preds += self.predict(model, X_test) / K
                elif prediction == 'binary_proba':
                    self.preds += self.binary_proba(model, X_test) / K
                else:
                    self.preds += self.predict(model, X_test) / K
            
            self.imps[:, fold_i] = model.get_feature_importances()
            self.scores[fold_i] = eval_metric(y_valid, self.oofs[valid_idx])
        
        print(
            f'\nOverall cv is {np.mean(self.scores):.3f} ± {np.std(self.scores):.3f}')


    def plot_feature_importances(self, columns):
        plt.figure(figsize=(5, int(len(columns) / 3)))
        imps_mean = np.mean(self.imps, axis=1)
        imps_se = np.std(self.imps, axis=1) / np.sqrt(self.imps.shape[0])
        order = np.argsort(imps_mean)
        plt.barh(np.array(columns)[order],
                imps_mean[order], xerr=imps_se[order])
        plt.show()


    def save(self, path):
        objects = [
            self.basemodel, self.datasplit, 
            self.models, self.oofs, self.preds, self.imps
        ]
        with open(path, 'wb') as f:
            pickle.dump(objects, f)
    
    
    def load(self, path):
        with open(path, 'rb') as f:
            objects = pickle.load(f)
        
        self.basemodel, self.datasplit, self.models, \
            self.oofs, self.preds, self.imps = objects
