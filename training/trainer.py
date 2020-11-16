import numpy as np
import pandas as pd

from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor
from catboost import Pool as CatPool
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor, LGBMRanker, LGBMModel
from lightgbm.compat import _LGBMLabelEncoder
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor, XGBRanker, XGBRFRegressor, XGBRFClassifier, XGBModel
from xgboost.compat import XGBoostLabelEncoder
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import seaborn as sns

from .utils import booster2sklearn
from ..utils import vector_normalize
from .logger import LGBMLogger
from .optuna_utils import OPTUNA_ZOO

try:
    import optuna
    import optuna.integration.lightgbm as lgb_tune
except ModuleNotFoundError:
    print('optuna not found.')

import pickle
from pathlib import Path


MODEL_ZOO = {
    'xgb': ['XGBClassifier', 'XGBRegressor', 'XGBRanker', 'XGBRFRegressor', 'XGBRFClassifier', 'XGBModel'],
    'lgb': ['LGBMClassifier', 'LGBMRegressor', 'LGBMRanker', 'LGBMModel'],
    'cat': ['CatBoost', 'CatBoostClassifier', 'CatBoostRegressor']
}


class Trainer:
    '''
    Amazing wrapper for sklearn like models

    Some useful features:
    - One line cross validation
    - Get various kinds of feature importance
    - Plot calibration curve
    - Built-in probality calibration (WIP)
    - Automated parameter tuning using Optuna
    - Export training log
    - Save and load your model
    '''

    def __init__(self, model=None, path=None, serial='trainer0'):
        if model is not None:
            self.model_type = model
            self.model_name = type(self.model_type()).__name__
            self.is_trained = False
        elif path is not None:
            self.load(path)
        else:
            raise ValueError('either model or path must be given.')
        self.feature_importance = None
        self.serial = serial

    def _data_check(self, data):
        assert isinstance(data, (list, tuple))
        assert len(data) in [0, 2, 3]
        for i, d in enumerate(data):
            assert isinstance(d, (pd.Series, pd.DataFrame, np.ndarray))
            if i == 0:
                dshape = len(d)
            assert dshape == len(d)
            dshape = len(d)

    def train(self, 
              # Dataset
              train_data, valid_data=(), cat_features=None, 
              # Model
              params={}, fit_params={}, convert_to_sklearn=True, 
              # Probability calibration
              calibration=False, calibration_method='isotonic', calibration_cv=5,
              # Optuna
              tune_model=False, optuna_params=None, maximize=True, 
              eval_metric=None, n_trials=None, timeout=None, 
              # Misc
              logger=None, n_jobs=-1):

        self._data_check(train_data)
        self._data_check(valid_data)
        self.n_features = train_data[0].shape[1]
        self.n_classes = len(np.unique(train_data[1]))
        if isinstance(train_data[0], pd.DataFrame):
            self.feature_names = train_data[0].columns.tolist()
        else:
            self.feature_names = [f'f{i}' for i in range(self.n_features)]

        if self.model_name in MODEL_ZOO['cat']:
            ''' Catboost '''
            dtrain = CatPool(
                data=train_data[0], 
                label=train_data[1], 
                weight=train_data[2] if len(train_data) == 3 else None, 
                cat_features=cat_features,
                thread_count=n_jobs)
            if len(valid_data) > 0:
                dvalid = CatPool(
                    data=valid_data[0],
                    label=valid_data[1],
                    weight=valid_data[2] if len(train_data) == 3 else None,
                    cat_features=cat_features,
                    thread_count=n_jobs)
            else:
                dvalid = dtrain
            
            # Optuna integration
            if tune_model:
                _fit_params = fit_params.copy()
                if 'verbose_eval' in _fit_params.keys():
                    _fit_params.update({'verbose_eval': False})
                def cat_objective(trial):
                    _params = params.copy()
                    _optuna_params = optuna_params
                    if _optuna_params is None:
                        _optuna_params = OPTUNA_ZOO[self.model_name](trial)
                    _params.update(_optuna_params)
                    if _params["bootstrap_type"] == "Bayesian":
                        _params["bagging_temperature"] = \
                            trial.suggest_float( "bagging_temperature", 0, 10)
                    elif _params["bootstrap_type"] == "Bernoulli":
                        _params["subsample"] = trial.suggest_float("subsample", 0.1, 1)
                    self.model = self.model_type(**_params)
                    self.model.fit(X=dtrain, eval_set=dvalid, **_fit_params)
                    if eval_metric is None:
                        score = self.model.best_score_['validation'][params['eval_metric']]
                    else:
                        score = eval_metric(self.model, valid_data)
                    return score
                study = optuna.create_study(direction='maximize' if maximize else 'minimize')
                study.optimize(
                    cat_objective, n_trials=n_trials, timeout=timeout, n_jobs=1)
                _params = params.copy()
                _params.update(study.best_trial.params)
            else:
                _params = params.copy()

            self.model = self.model_type(**_params)
            self.model.fit(X=dtrain, eval_set=dvalid, **fit_params)
            self.best_iteration = self.model.get_best_iteration()
            self.best_score = self.model.best_score_['validation'][params['eval_metric']]

        elif self.model_name in MODEL_ZOO['lgb']:
            ''' LightGBM '''
            dtrain = lgb.Dataset(
                data=train_data[0],
                label=train_data[1],
                weight=train_data[2] if len(train_data) == 3 else None,
                categorical_feature=cat_features)
            if len(valid_data) > 0:
                dvalid = lgb.Dataset(
                    data=valid_data[0],
                    label=valid_data[1],
                    weight=valid_data[2] if len(train_data) == 3 else None,
                    categorical_feature=cat_features)
            else:
                dvalid = dtrain

            # Optuna intergration
            if tune_model:
                _fit_params = fit_params.copy()
                if 'verbose_eval' in _fit_params.keys():
                    _fit_params.update({'verbose_eval': False})
                self.model = lgb_tune.train(
                    params, train_set=dtrain, valid_sets=[dtrain, dvalid], **_fit_params)
                # _params = self.model.params.copy()
                self.best_iteration = self.model.best_iteration
            else:
                _params = params.copy()

                if isinstance(logger, (str, Path)):
                    callbacks = [LGBMLogger(logger, params, fit_params)]
                elif isinstance(logger, LGBMLogger):
                    callbacks = [logger]
                elif logger is None:
                    callbacks = None
                else:
                    raise ValueError('invalid logger.')
                
                res = {}
                self.model = lgb.train(
                    _params, train_set=dtrain, valid_sets=[dtrain, dvalid], 
                    callbacks=callbacks, evals_result=res, **fit_params)
                self.best_iteration = self.model.best_iteration
                if maximize:
                    self.best_score = np.max(res['valid_1'][_params['metric']])
                else:
                    self.best_score = np.min(res['valid_1'][_params['metric']])

            if convert_to_sklearn:
                self.model = booster2sklearn(
                    self.model, self.model_type, self.n_features, self.n_classes)
                if self.model_name == 'LGBMClassifier':
                    self.model._le = _LGBMLabelEncoder().fit(train_data[1]) # internal label encoder 

        elif self.model_name in MODEL_ZOO['xgb']:
            ''' XGBoost '''
            dtrain = xgb.DMatrix(
                data=train_data[0],
                label=train_data[1],
                weight=train_data[2] if len(train_data) == 3 else None,
                nthread=n_jobs)
            if len(valid_data) > 0:
                dvalid = xgb.DMatrix(
                    data=valid_data[0],
                    label=valid_data[1],
                    weight=valid_data[2] if len(train_data) == 3 else None,
                    nthread=n_jobs)
            else:
                dvalid = dtrain

            # Optuna integration
            if tune_model:
                _fit_params = fit_params.copy()
                if 'verbose_eval' in _fit_params.keys():
                    _fit_params.update({'verbose_eval': False})
                def xgb_objective(trial):
                    _params = params.copy()
                    _optuna_params = optuna_params
                    if _optuna_params is None:
                        _optuna_params = OPTUNA_ZOO[self.model_name](trial)
                    _params.update(_optuna_params)
                    if _params["booster"] == "gbtree" or _params["booster"] == "dart":
                        _params["max_depth"] = trial.suggest_int("max_depth", 1, 9)
                        _params["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
                        _params["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
                        _params["grow_policy"] = trial.suggest_categorical(
                            "grow_policy", ["depthwise", "lossguide"])
                    if _params["booster"] == "dart":
                        _params["sample_type"] = trial.suggest_categorical(
                            "sample_type", ["uniform", "weighted"])
                        _params["normalize_type"] = trial.suggest_categorical(
                            "normalize_type", ["tree", "forest"])
                        _params["rate_drop"] = trial.suggest_float(
                            "rate_drop", 1e-8, 1.0, log=True)
                        _params["skip_drop"] = trial.suggest_float(
                            "skip_drop", 1e-8, 1.0, log=True)
                    res = {}
                    self.model = xgb.train(
                        _params, dtrain=dtrain, 
                        evals=[(dtrain, 'train'), (dvalid, 'valid')],
                        evals_result=res, **_fit_params)
                    if eval_metric is None:
                        if maximize:
                            score = np.max(res['valid'][_params['eval_metric']])
                        else:
                            score = np.min(res['valid'][_params['eval_metric']])
                    else:
                        score = eval_metric(self.model, dvalid)
                    return score
                study = optuna.create_study(direction='maximize' if maximize else 'minimize')
                study.optimize(
                    xgb_objective, n_trials=n_trials, timeout=timeout, n_jobs=1)
                _params = params.copy()
                _params.update(study.best_trial.params)
            else:
                _params = params.copy()

            if isinstance(logger, (str, Path)):
                callbacks = [LGBMLogger(logger, params, fit_params)]
            elif isinstance(logger, LGBMLogger):
                callbacks = [logger]
            elif logger is None:
                callbacks = None
            else:
                raise ValueError('invalid logger.')
            
            res = {}
            self.model = xgb.train(
                _params, dtrain=dtrain, evals=[(dtrain, 'train'), (dvalid, 'valid')], 
                callbacks=callbacks, evals_result=res, **fit_params)
            self.best_iteration = self.model.best_iteration
            if maximize:
                self.best_score = np.max(res['valid'][_params['eval_metric']])
            else:
                self.best_score = np.min(res['valid'][_params['eval_metric']])
            if convert_to_sklearn:
                self.model = booster2sklearn(
                    self.model, self.model_type, self.n_features, self.n_classes)
                if self.model_name == 'XGBClassifier':
                    self.model._le = XGBoostLabelEncoder().fit(train_data[1])

        else:
            # Optuna integration
            if tune_model:
                if eval_metric is None:
                    raise ValueError('eval_metric is necessary for optuna.')
                def sklearn_objective(trial):
                    _params = params.copy()
                    _optuna_params = optuna_params
                    if _optuna_params is None:
                        _optuna_params = OPTUNA_ZOO[self.model_name](trial)
                    _params.update(_optuna_params)
                    self.model = self.model_type(**_params)
                    self.model.fit(train_data[0], train_data[1], **fit_params)
                    score = eval_metric(self.model, valid_data)
                    return score
                study = optuna.create_study(direction='maximize' if maximize else 'minimize')
                study.optimize(
                    sklearn_objective, n_trials=n_trials, timeout=timeout, n_jobs=1)
            self.model = self.model_type(**params)
            self.model.fit(train_data[0], train_data[1], **fit_params)
            self.best_iteration = -1
            self.best_score = eval_metric(self.model, valid_data)

        self.is_trained = True

        if calibration:
            pass
            # self.calibrate_model = CalibratedClassifierCV(
            #     self.model, method=calibration_method, cv=calibration_cv)
            # self.calibrate_model.fit(train_data[0], train_data[1])

    def predict(self, X, **kwargs):
        assert self.is_trained
        return self.model.predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        assert self.is_trained
        return self.model.predict_proba(X, **kwargs)

    def get_model(self):
        return self.model

    def get_best_iteration(self):
        return self.best_iteration

    def get_best_score(self):
        return self.best_score

    def get_permutation_importance(self, fit_params):
        assert fit_params is not None
        res = permutation_importance(self.model, **fit_params)
        return res['importances_mean'], res['importances_std']

    def get_feature_importance(self, importance_type='auto', normalize=True, fit_params=None):
        if self.model_name in MODEL_ZOO['cat']:
            ''' CatBoost '''
            if importance_type == 'auto':
                imp = self.model.get_feature_importance()
            elif importance_type == 'split':
                print(f'model type {self.model_name} does not support importance type {importance_type}')
                imp = self.model.get_feature_importance()
            elif importance_type == 'gain':
                imp = self.model.get_feature_importance()
            elif importance_type == 'permutation':
                imp = self.get_permutation_importance(fit_params)[0]
            elif importance_type == 'null':
                # WIP
                pass
            else:
                raise ValueError(f'Unsupported importance type {importance_type}')

        elif self.model_name in MODEL_ZOO['lgb']:
            ''' LightGBM '''
            if isinstance(self.model, lgb.Booster):
                _model = self.model
            else:
                _model = self.model._Booster
            
            if importance_type == 'auto':
                imp = _model.feature_importance(importance_type='split')
            elif importance_type == 'split':
                imp = _model.feature_importance(importance_type='split')
            elif importance_type == 'gain':
                imp = _model.feature_importance(importance_type='gain')
            elif importance_type == 'permutation':
                imp = self.get_permutation_importance(fit_params)[0]
            elif importance_type == 'null':
                # WIP
                pass
            else:
                raise ValueError(
                    f'Unsupported importance type {importance_type}')

        elif self.model_name in MODEL_ZOO['xgb']:
            ''' XGBoost '''
            if isinstance(self.model, xgb.Booster):
                _model = self.model
            else:
                _model = self.model.get_booster()

            if importance_type == 'auto':
                _imp = _model.get_score(importance_type='weight')
            elif importance_type == 'split':
                _imp = _model.get_score(importance_type='weight')
            elif importance_type == 'gain':
                _imp = _model.get_score(importance_type='gain')
            elif importance_type == 'permutation':
                imp = self.get_permutation_importance(fit_params)[0]
            elif importance_type == 'null':
                # WIP
                pass
            else:
                raise ValueError(
                    f'Unsupported importance type {importance_type}')
            try:
                imp
            except:
                imp = [_imp[f] if f in _imp.keys() else 0  for f in self.feature_names]

        else:
            ''' Others '''
            if importance_type in ['auto', 'split', 'gain']:
                try:
                    imp = self.model.feature_importances_
                except:
                    raise ValueError(f'model type {self.model_name} does not support importance type {importance_type}')
            elif importance_type == 'permutation':
                imp = self.get_permutation_importance(fit_params)[0]
            elif importance_type == 'null':
                # WIP
                pass
            else:
                raise ValueError(
                    f'Unsupported importance type {importance_type}')
            
        if normalize:
            imp = vector_normalize(imp)
        
        self.feature_importance = {self.feature_names[i]: imp[i] for i in range(len(self.feature_names))}
        return self.feature_importance

    def plot_feature_importance(self, importance_type='auto', normalize=True, fit_params=None, 
                                sorted=True, width=5, save_to=None):
        imp_dict = self.get_feature_importance(importance_type, normalize, fit_params)
        columns, imps = np.array(list(imp_dict.keys())), np.array(list(imp_dict.values()))
        plt.figure(figsize=(width, len(columns) / 3))
        order = np.argsort(imps)
        colors = colormap.winter(np.arange(len(columns))/len(columns))
        plt.barh(columns[order], imps[order], color=colors)
        if save_to is not None:
            plt.savefig(save_to)
        plt.show()

    def plot_calibartion_curve(self, X, y, predict_params={}, width=4, save_to=None):
        approx = self.predict_proba(X, **predict_params)[:, 1]
        fig = plt.figure(figsize=(width, width*1.5), tight_layout=True)
        gs = fig.add_gridspec(3, 1)
        ax1 = fig.add_subplot(gs[0:2, 0])
        ax2 = fig.add_subplot(gs[2, 0])
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y, approx, n_bins=10)
        ax1.plot([0, 1], [0, 1], color='gray')
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-")
        ax1.set_xlabel('Fraction of positives')
        ax1.set_ylabel('Mean of prediction values')
        ax1.grid()
        ax1.set_xlim([0.0, 1.0])
        sns.distplot(approx, bins=10, ax=ax2)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylabel('Count')
        if save_to is not None:
            plt.savefig(save_to)
        plt.show()

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(
                (self.model_type, self.model_name, self.model, 
                 self.is_trained, self.best_iteration,
                 self.feature_names, self.n_features, self.n_classes, 
                 self.feature_importance), f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.model_type, self.model_name, self.model,\
            self.is_trained, self.best_iteration,\
            self.feature_names, self.n_features, self.n_classes,\
            self.feature_importance = pickle.load(f)
