import numpy as np
import pandas as pd

from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor
from catboost import Pool as CatPool
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor, LGBMRanker, LGBMModel
from lightgbm.compat import _LGBMLabelEncoder
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor, XGBRanker, XGBRFRegressor, XGBRFClassifier, XGBModel
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import seaborn as sns

from .utils import booster2sklearn
from ..utils import vector_normalize
from .logger import LGBMLogger, XGBLogger

try:
    import optuna
except ModuleNotFoundError:
    print('optuna not found.')

import pickle


MODEL_ZOO = {
    'xgb': ['XGBClassifier', 'XGBRegressor', 'XGBRanker', 'XGBRFRegressor', 'XGBRFClassifier', 'XGBModel'],
    'lgb': ['LGBMClassifier', 'LGBMRegressor', 'LGBMRanker', 'LGBMModel'],
    'cat': ['CatBoost', 'CatBoostClassifier', 'CatBoostRegressor']
}


class Trainer:
    '''
    Wrapper for sklearn like models

    Some useful features:
    - Get various kinds of feature importance
    - Plot calibration curve
    - Built-in probality calibration (WIP)
    - Automated parameter tuning using Optuna
    - Export training log
    - Save and load your model
    '''

    def __init__(self, model=None, path=None):
        if model is not None:
            self.model_type = model
            self.model_name = type(self.model_type()).__name__
            self.is_trained = False
        elif path is not None:
            self.load(path)
        else:
            raise ValueError('either model or path must be given.')

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
              params=None, fit_params=None, convert_to_sklearn=True, 
              # Probability calibration
              calibration=False, calibration_method='isotonic', calibration_cv=5,
              # Optuna
              optimize_params=False, 
              # Misc
              log_path=None, n_jobs=-1):

        self._data_check(train_data)
        self._data_check(valid_data)
        n_features = train_data[0].shape[1]
        n_classes = len(np.unique(train_data[1]))
        if isinstance(train_data[0], pd.DataFrame):
            self.feature_names = train_data[0].columns.tolist()
        else:
            self.feature_names = [f'f{i}' for i in range(n_features)]

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
            
            self.model = self.model_type(**params)
            self.model.fit(X=dtrain, eval_set=dvalid, **fit_params)
            self.best_iteration = self.model.get_best_iteration()

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

            logger = LGBMLogger(
                log_path if log_path is not None else '.trainer.log', params, fit_params)
            callbacks = [logger]

            self.model = lgb.train(
                params, train_set=dtrain, valid_sets=[dtrain, dvalid], callbacks=callbacks, **fit_params)
            self.best_iteration = self.model.best_iteration
            if convert_to_sklearn:
                self.model = booster2sklearn(self.model, self.model_type, n_features, n_classes)
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

            logger = XGBLogger(
                log_path if log_path is not None else '.trainer.log')
            callbacks = [logger]

            self.model = xgb.train(
                params, dtrain=dtrain, evals=[(dtrain, 'train'), (dvalid, 'valid')], 
                callbacks=callbacks, **fit_params)
            self.best_iteration = self.model.best_iteration
            if convert_to_sklearn:
                self.model = booster2sklearn(self.model, self.model_type, n_features, n_classes)

        else:
            self.model = self.model_type(**params)
            self.model.fit(train_data[0], train_data[1], **fit_params)
            self.best_iteration = -1

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

        return {self.feature_names[i]: imp[i] for i in range(len(self.feature_names))}

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
                self.is_trained, self.feature_names, self.best_iteration), f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.model_type, self.model_name, self.model, \
            self.is_trained, self.feature_names, self.best_iteration = pickle.load(f)
