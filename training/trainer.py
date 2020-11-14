import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import seaborn as sns

from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor
from catboost import Pool as CatPool
import lightgbm as lgb
import xgboost as xgb


class Trainer:
    '''
    Wrapper for sklearn like models

    # Input:
    train/valid/test_data: (X, y, w=None)
    '''

    MODEL_ZOO = {
        'cat': CatBoost,
        'lgb': lgb.train,
        'xgb': xgb.train,
    }

    def __init__(self, model) -> None:
        if isinstance(model, str):
            if model in self.MODEL_ZOO.keys():
                self.model = self.MODEL_ZOO[model]
                self.model_name = model
            else:
                raise ValueError(f'{model} not in {self.MODEL_ZOO.keys()}')
        else:
            self.model = model
            self.model_name = type(model).__name__
        
        self.is_trained = False
        self.null_importances = None
        self.perm_importances = None
        self.calibrate_model = None

    def _data_check(self, data) -> None:
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
              params=None, fit_params=None,
              calibration=False, calibration_params={'method': 'isotonic', 'cv': 3},
              # Misc
              log_path=None, n_jobs=-1):

        self._data_check(train_data)
        self._data_check(valid_data)


        if self.model_name in ['CatBoostRegressor', 'CatBoostClassifier', 'cat']:
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
            
            _params = params.copy()
            if self.model_name == 'cat':
                self.model = self.model(params)
            self.model.fit(X=dtrain, eval_set=dvalid, **fit_params)
            self.best_iteration = self.model.get_best_iteration()


        elif self.model_name == 'lgb':
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

            self.model = self.model(
                params, train_set=dtrain, valid_sets=[dtrain, dvalid], **fit_params)
            self.best_iteration = self.model.best_iteration

        elif self.model_name == 'xgb':
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

            self.model = self.model(
                params, dtrain=dtrain, evals=[(dtrain, 'train'), (dvalid, 'valid')], **fit_params)
            self.best_iteration = self.model.best_iteration

        else:
            self.model = self.model(**params)
            self.model.fit(train_data[0], train_data[1], **fit_params)
            self.best_iteration = -1

        self.is_trained = True

        # if calibration:
        #     self.cal_model = CalibratedClassifierCV(
        #         self.model, **calibration_params)
        #     self.cal_model.fit(X, y)

    def get_model(self):
        return self.model

    def get_best_iteration(self):
        return self.best_iteration

    # def predict(self, X, method='predict'):
    #     if X is None:
    #         print('No data to predict.')
    #         return None
    #     if self.cal_model is None:
    #         _model = self.model
    #     else:
    #         _model = self.cal_model

    #     if method == 'predict':
    #         return _model.predict(X)
    #     elif method == 'binary_proba':
    #         return _model.predict_proba(X)
    #     elif method == 'binary_proba_positive':
    #         return _model.predict_proba(X)[:, 1]
    #     else:
    #         raise ValueError(method)

    # def get_feature_importances(self, method='fast', importance_params={}):
    #     if method == 'auto':
    #         if self.model_type in ['CatBoostRegressor', 'CatBoostClassifier',
    #                                'LGBMRegressor', 'LGBMClassifier',
    #                                'RandomForestRegressor', 'RandomForestClassifier']:
    #             return self.model.feature_importances_
    #         else:
    #             if self.permutation_importances is None:
    #                 return self.get_permutation_importances(**importance_params)
    #             else:
    #                 return self.permutation_importances
    #     elif method == 'fast':
    #         if self.model_type in ['CatBoostRegressor', 'CatBoostClassifier',
    #                                'LGBMRegressor', 'LGBMClassifier',
    #                                'RandomForestRegressor', 'RandomForestClassifier']:
    #             return self.model.feature_importances_
    #         else:  # Gomennasai
    #             return np.zeros(self.input_shape[1])
    #     elif method == 'permutation':
    #         if self.permutation_importances is None:
    #             return self.get_permutation_importances(**importance_params)
    #         else:
    #             return self.permutation_importances
    #     elif method == 'null':
    #         if self.null_importances is None:
    #             return self.get_null_importances(**importance_params)
    #         else:
    #             return self.null_importances
    #     else:
    #         raise ValueError(method)

    # def get_null_importances(self, X, y, X_valid=None, y_valid=None,
    #                          cat_features=None, fit_params={}, prediction='predict',
    #                          eval_metric=None, iteration=10, verbose=False):
    #     assert self.model_type in ['CatBoostRegressor', 'CatBoostClassifier',
    #                                'LGBMRegressor', 'LGBMClassifier',
    #                                'RandomForestRegressor', 'RandomForestClassifier']
    #     assert self.is_trained

    #     if verbose:
    #         _iter = tqdm(range(iteration), desc='Calculating null importance')
    #     else:
    #         _iter = range(iteration)

    #     # Get actual importances
    #     actual_imps = self.model.feature_importances_
    #     null_imps = np.empty(
    #         (iteration, self.input_shape[1]), dtype=np.float16)

    #     # Get null importances
    #     for i in _iter:
    #         y_shuffled = np.random.permutation(y)
    #         self.train(X, y_shuffled, X_valid, y_valid,
    #                    cat_features, fit_params)
    #         null_imps[i] = self.model.feature_importances_

    #     # Calculation feature score
    #     null_imps75 = np.percentile(null_imps, 75, axis=0)
    #     null_importances = np.log(1e-10 + actual_imps / (1 + null_imps75))
    #     self.null_importances = null_importances
    #     return null_importances

    # def get_permutation_importances(self, X, y, X_valid=None, y_valid=None,
    #                                 cat_features=None, fit_params={}, pred_method='predict',
    #                                 eval_metric=RMSE(), iteration=None, verbose=False):
    #     assert self.is_trained

    #     if verbose:
    #         _iter = tqdm(
    #             range(self.input_shape[1]), desc='Calculating permutation importance')
    #     else:
    #         _iter = range(self.input_shape[1])

    #     # Get baseline score
    #     if X_valid is None:
    #         pred = self.predict(X, pred_method)
    #         baseline_score = eval_metric(y, pred)
    #     else:
    #         pred = self.predict(X_valid, pred_method)
    #         baseline_score = eval_metric(y_valid, pred)
    #     permutation_scores = np.empty(self.input_shape[1])

    #     # Get permutation importances
    #     for icol in _iter:
    #         X_shuffled = copy(X)
    #         X_shuffled[:, icol] = np.random.permutation(X_shuffled[:, icol])
    #         self.train(X_shuffled, y, X_valid, y_valid,
    #                    cat_features, fit_params)
    #         if X_valid is None:
    #             pred = self.predict(X, pred_method)
    #             permutation_scores[icol] = eval_metric(y, pred)
    #         else:
    #             pred = self.predict(X_valid, pred_method)
    #             permutation_scores[icol] = eval_metric(y_valid, pred)
    #         del X_shuffled, pred
    #         gc.collect()

    #     # Calculation feature score
    #     permutation_importances = permutation_scores - baseline_score
    #     self.permutation_importances = permutation_importances
    #     return permutation_importances

    # def get_coeffcients(self):
    #     if self.model_type in ['LinearRegression', 'LogisticRegression',
    #                            'Ridge', 'Lasso']:
    #         return self.model.coef_
    #     elif self.model_type in ['SVR', 'SVC'] and self.model.get_params()['kernel'] == 'linear':
    #         return self.model.coef_
    #     else:
    #         return np.zeros(self.input_shape[1])

    # def plot_feature_importances(self, columns=None, method='fast'):
    #     imps = self.get_feature_importances(method)

    #     if columns is None:
    #         columns = [f'feature_{i}' for i in range(len(imps))]

    #     plt.figure(figsize=(5, int(len(columns) / 3)))
    #     order = np.argsort(imps)
    #     colors = colormap.winter(np.arange(len(columns))/len(columns))
    #     plt.barh(np.array(columns)[order], imps[order], color=colors)
    #     plt.show()
