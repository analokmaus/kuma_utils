import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from copy import deepcopy
import optuna
import optuna.integration.lightgbm as lgb_tune
optuna.logging.set_verbosity(optuna.logging.FATAL)
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import seaborn as sns
from catboost import Pool as CatPool
import lightgbm as lgb
import xgboost as xgb
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold

try:
    import japanize_matplotlib
except ModuleNotFoundError:
    pass

from .utils import booster2sklearn, ModelExtractor, XGBModelExtractor, auc_metric, mse_metric
from .logger import LGBMLogger
from .optuna import PARAMS_ZOO
from kuma_utils.utils import vector_normalize


MODEL_ZOO = {
    'xgb': [
        'XGBClassifier', 'XGBRegressor', 'XGBRanker', 'XGBRFRegressor', 
        'XGBRFClassifier', 'XGBModel'],
    'lgb': [
        'LGBMClassifier', 'LGBMRegressor', 'LGBMRanker', 'LGBMModel'],
    'cat': [
        'CatBoost', 'CatBoostClassifier', 'CatBoostRegressor']
}


DIRECTION = {
    'maximize': [
        # CatBoost
        'Precision', 'Recall', 'F1', 'BalancedAccuracy', 'MCC', 'Accuracy', 
        'AUC', 'Kappa', 'WKappa', 'LogLikelihoodOfPrediction', 'R2', 'NDCG'
        # LightGBM
        'map', 'auc', 'average_precision', 'ndcg', 
        # XGBoost
        'auc', 'aucpr', 'map', 'ndcg'
    ],
    'minimize': [
        # CatBoost
        'Logloss', 'CrossEntropy', 'BalancedErrorRate', 'HingeLoss', 'HammingLoss', 
        'ZeroOneLoss', 'MAE', 'MAPE', 'Poisson', 'Quantile', 'RMSE', 'Lq', 'Huber',
        'FairLoss', 'SMAPE', 'MSLE', 'MultiRMSE', 'MultiClass', 'MultiClassOneVsAll', 
        # LightGBM
        'l1', 'mean_absolute_error', 'mae', 'l2', 'mean_squared_error', 'mse', 'regression', 
        'rmse', 'oot_mean_squared_error', 'quantile', 'mape', 'huber', 'fair', 'poisson', 
        'binary_logloss', 'binary', 'binary_error', 'multi-logloss', 'multiclass', 'softmax',
        'multi-error', 'cross_entropy', 'kullback_leibler', 
        # XGBoost
        'rmse', 'rmsle', 'mae', 'mape', 'mphe', 'logloss', 'error', 'merror', 'mlogloss',
        'poisson-nlogli'
    ]
}


class Trainer:
    '''
    Wrapper for sklearn API models

    Features:
    - Perform cross validation during training process (only for GBDT)
    - Get various kinds of feature importance
    - Plot calibration curve
    - Automated parameter tuning using Optuna
    - Export training log
    - Save and load your model
    '''
    def __init__(self, model=None, path=None, serial='trainer0'):
        self.serial = serial
        self.snapshot_items = [
            'serial', 'model', 'model_name', 'is_trained', 'best_iteration', 'best_score',
            'evals_result', 'feature_names', 'n_features', 'n_classes'
        ]
        if model is not None:
            self.model_type = model
            self.model_name = type(self.model_type()).__name__
            self.is_trained = False
            self.best_score = None
            self.best_iteration = None
            self.evals_result = None
            self.feature_names = None
            self.n_features = None
            self.n_classes = None
            self.feature_importance = None
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

    def _parse_eval_metric(self, params, maximize):
        for key in ['metric', 'eval_metric']:
            if key in params.keys():
                if isinstance(params[key], (list, tuple)):
                    print('setting multiple metrics are not recommended.')
                    main_metric = params[key][0]
                elif isinstance(params[key], str):
                    main_metric = params[key]
                else:
                    main_metric = params[key].__class__.__name__
                if not main_metric in DIRECTION['maximize'] + DIRECTION['minimize']:
                    print(f'specify optimization direction for metric {main_metric}.')
                _maximize = main_metric in DIRECTION['maximize']
                return main_metric, _maximize
        else:
            return None, maximize

    def train(self, 
              # Dataset
              train_data, valid_data=(), cat_features=None, 
              # Model
              params={}, fit_params={}, convert_to_sklearn=True, 
              # Probability calibration
              calibration=False, calibration_method='isotonic', calibration_cv=5,
              # Optuna
              tune_model=False, optuna_params=None, maximize=None, 
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

        main_metric, maximize = self._parse_eval_metric(params, maximize)
        if eval_metric is not None and maximize is None:
            raise ValueError('metric direction must be specified.')
    
        if isinstance(logger, (str, Path)):
            logger = LGBMLogger(logger, stdout=True, file=True)
        elif logger is None:
            logger = LGBMLogger(logger, stdout=True, file=False)
        assert isinstance(logger, LGBMLogger)

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
                self.best_score = None
                self.best_model = None

                def cat_objective(trial):
                    _params = params.copy()
                    if optuna_params is None:
                        _params = PARAMS_ZOO[self.model_name](trial, _params)
                    else:
                        _params = optuna_params(trial, _params)
                    
                    self.model = self.model_type(**_params)
                    self.model.fit(X=dtrain, eval_set=dvalid, **_fit_params)

                    if eval_metric is None:
                        score = self.model.get_best_score()['validation'][main_metric]
                    else:
                        score = eval_metric(self.model, valid_data)
                    
                    if self.best_score is None:
                        self.best_score = score
                        self.best_model = self.model.copy()
                        self.best_iteration = self.model.get_best_iteration()
                        self.evals_result = self.model.get_evals_result()
                    
                    if maximize and self.best_score < score:
                        self.best_model = self.model.copy()
                        self.best_score = score
                        self.best_iteration = self.model.get_best_iteration()
                        self.evals_result = self.model.get_evals_result()
                    elif not maximize and self.best_score > score:
                        self.best_model = self.model.copy()
                        self.best_score = score
                        self.best_iteration = self.model.get_best_iteration()
                        self.evals_result = self.model.get_evals_result()
                    
                    return score
                
                study = optuna.create_study(direction='maximize' if maximize else 'minimize')
                study.optimize(
                    cat_objective, n_trials=n_trials, timeout=timeout,
                    callbacks=[logger.optuna], n_jobs=1)
                self.model = self.best_model.copy()
                del self.best_model
                
            else:
                _params = params.copy()
                self.model = self.model_type(**_params)
                self.model.fit(X=dtrain, eval_set=dvalid, **fit_params)
                self.best_score = self.model.get_best_score()['validation'][main_metric]
                self.evals_result = self.model.get_evals_result()
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

            # Optuna intergration
            if tune_model:
                _fit_params = fit_params.copy()
                _params = params.copy()
                _params.update({'metric': main_metric})
                study = optuna.create_study(direction='maximize' if maximize else 'minimize')
                res = {}
                tuner = lgb_tune.LightGBMTuner(
                    _params, 
                    train_set=dtrain, valid_sets=[dtrain, dvalid], 
                    valid_names='valid_1', verbosity=0,
                    study=study, time_budget=timeout, show_progress_bar=False,
                    optuna_callbacks=[logger.optuna], **deepcopy(_fit_params))
                tuner.run()
                self.model = tuner.get_best_booster()
                self.best_score = tuner.best_score
                del tuner
            else:
                _params = params.copy()
                
                res = {}
                lgb_callbacks = [logger.lgbm, lgb.record_evaluation(res)]
                if 'callbacks' in fit_params.keys():
                    lgb_callbacks += fit_params['callbacks']
                    del fit_params['callbacks']
                self.model = lgb.train(
                    _params, train_set=dtrain, valid_sets=[dtrain, dvalid], 
                    callbacks=lgb_callbacks, **fit_params)
                if maximize:
                    self.best_score = np.max(res['valid_1'][main_metric])
                else:
                    self.best_score = np.min(res['valid_1'][main_metric])
                self.evals_result = res

            self.best_iteration = self.model.best_iteration
            
            if convert_to_sklearn:
                self.model = booster2sklearn(
                    self.model, self.model_type, train_data[0], train_data[1])                    

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
                self.best_score = None
                self.best_model = None

                def xgb_objective(trial):
                    _params = params.copy()
                    if optuna_params is None:
                        _params = PARAMS_ZOO[self.model_name](trial, _params)
                    else:
                        _params = optuna_params(trial, _params)
                    
                    res = {}
                    pruning_callback = optuna.integration.XGBoostPruningCallback(
                        trial, f'valid-{main_metric}')
                    self.model = xgb.train(
                        _params, dtrain=dtrain,
                        evals=[(dtrain, 'train'), (dvalid, 'valid')],
                        evals_result=res, callbacks=[pruning_callback], **_fit_params)

                    if eval_metric is None:
                        if maximize:
                            score = np.max(res['valid'][main_metric])
                        else:
                            score = np.min(res['valid'][main_metric])
                    else:
                        score = eval_metric(self.model, dvalid)
                    
                    if self.best_score is None:
                        self.best_score = score
                        self.best_model = self.model.copy()
                        self.evals_result = res
                    
                    if maximize and self.best_score < score:
                        self.best_score = score
                        self.best_model = self.model.copy()
                        self.evals_result = res
                    elif not maximize and self.best_score > score:
                        self.best_score = score
                        self.best_model = self.model.copy()
                        self.evals_result = res

                    return score

                study = optuna.create_study(direction='maximize' if maximize else 'minimize')
                study.optimize(
                    xgb_objective, n_trials=n_trials, timeout=timeout, 
                    callbacks=[logger.optuna], n_jobs=1)
                self.model = self.best_model.copy()
                del self.best_model
            else:
                _params = params.copy()
                
                res = {}
                self.model = xgb.train(
                    _params, dtrain=dtrain, evals=[(dtrain, 'train'), (dvalid, 'valid')], 
                    callbacks=[logger.lgbm], evals_result=res, **fit_params)
                if maximize:
                    self.best_score = np.max(res['valid'][main_metric])
                else:
                    self.best_score = np.min(res['valid'][main_metric])
                self.evals_result = res

            self.best_iteration = self.model.best_ntree_limit

            if convert_to_sklearn:
                self.model = booster2sklearn(
                    self.model, self.model_type, train_data[0], train_data[1])

        else:
            ''' Other skelarn models '''
            if eval_metric is None:
                if self.n_classes == 2:
                    eval_metric = auc_metric
                    maximize = True
                else:
                    eval_metric = mse_metric
                    maximize = False
                print('eval_metric automatically selected.')

            # Optuna integration
            if tune_model:
                self.best_score = None
                self.best_model = None

                def sklearn_objective(trial):
                    _params = params.copy()
                    if optuna_params is None:
                        _params = PARAMS_ZOO[self.model_name](trial, params)
                    else:
                        _params = optuna_params(trial, _params)
                    
                    self.model = self.model_type(**_params)
                    self.model.fit(train_data[0], train_data[1], **fit_params)
                    score = eval_metric(self.model, valid_data)

                    if self.best_score is None:
                        self.best_score = score
                        self.best_model = deepcopy(self.model)

                    if maximize and self.best_score < score:
                        self.best_model = deepcopy(self.model)
                        self.best_score = score
                    elif not maximize and self.best_score > score:
                        self.best_model = deepcopy(self.model)
                        self.best_score = score
                    
                    return score

                study = optuna.create_study(direction='maximize' if maximize else 'minimize')
                study.optimize(
                    sklearn_objective, n_trials=n_trials, timeout=timeout, 
                    callbacks=[logger.optuna], n_jobs=1)
                self.model = deepcopy(self.best_model)
                del self.best_model
            else:
                self.model = self.model_type(**params)
                self.model.fit(train_data[0], train_data[1], **fit_params)
                self.best_score = eval_metric(self.model, valid_data)
                logger(f'[None]\tbest score is {self.best_score:.6f}')

        self.is_trained = True

        if calibration:
            pass

    fit = train

    def cv(self,
           # Dataset
           data, cat_features=None, groups=None, folds=KFold(n_splits=5),
           # Model
           params={}, fit_params={}, convert_to_sklearn=True,
           # Optuna
           tune_model=False, optuna_params=None, maximize=None,
           eval_metric=None, n_trials=None, timeout=None,
           lgbm_n_trials=[7, 20, 10, 6, 20],
           # Misc
           logger=None, n_jobs=-1):
        
        self._data_check(data)
        self.n_features = data[0].shape[1]
        self.n_classes = len(np.unique(data[1]))
        if isinstance(data[0], pd.DataFrame):
            self.feature_names = data[0].columns.tolist()
        else:
            self.feature_names = [f'f{i}' for i in range(self.n_features)]

        main_metric, maximize = self._parse_eval_metric(params, maximize)
        if eval_metric is not None and maximize is None:
            raise ValueError('metric direction must be specified.')

        if isinstance(logger, (str, Path)):
            logger = LGBMLogger(logger, stdout=True, file=True)
        elif logger is None:
            logger = LGBMLogger(logger, stdout=True, file=False)
        assert isinstance(logger, LGBMLogger)

        if self.model_name in MODEL_ZOO['cat']:
            raise NotImplementedError('catboost is incompatible with .cv().')

        elif self.model_name in MODEL_ZOO['lgb']:
            ''' LightGBM '''
            dtrain = lgb.Dataset(
                data=data[0],
                label=data[1],
                weight=data[2] if len(data) == 3 else None,
                categorical_feature=cat_features)

            # Optuna intergration
            if tune_model:
                _fit_params = fit_params.copy()
                _params = params.copy()
                _params.update({'metric': main_metric})
                study = optuna.create_study(direction='maximize' if maximize else 'minimize')
                tuner = lgb_tune.LightGBMTunerCV(
                    _params, dtrain, folds=folds, time_budget=timeout,
                    study=study, return_cvbooster=True, optuna_callbacks=[logger.optuna],
                    show_progress_bar=False, **deepcopy(_fit_params))
                tuner.run()
                self.model = tuner.get_best_booster().boosters
                self.best_iteration = tuner.get_best_booster().best_iteration
                self.best_score = tuner.best_score
                del tuner
            else:
                _params = params.copy()
                model_extractor = ModelExtractor()
                res = {}
                lgb_callbacks = [
                    logger.lgbm, model_extractor, lgb.record_evaluation(res)]
                if 'callbacks' in fit_params.keys():
                    lgb_callbacks += fit_params['callbacks']
                    del fit_params['callbacks']
                res = lgb.cv(
                    _params, train_set=dtrain, folds=folds, 
                    callbacks=lgb_callbacks,  **fit_params)
                self.model = model_extractor.get_model().boosters
                self.best_iteration = model_extractor.get_best_iteration()
                self.evals_result = res
                if maximize:
                    self.best_score = np.max(res[f'valid {main_metric}-mean'])
                else:
                    self.best_score = np.min(res[f'valid {main_metric}-mean'])
                
            for i in range(len(self.model)):
                self.model[i].best_iteration = self.best_iteration

            logger(f'[{self.best_iteration}]\tbest score is {self.best_score:.6f}')

            if convert_to_sklearn:
                for i in range(len(self.model)):
                    self.model[i] = booster2sklearn(
                        self.model[i], self.model_type, data[0], data[1])

        elif self.model_name in MODEL_ZOO['xgb']:
            ''' XGBoost '''
            dtrain = xgb.DMatrix(
                data=data[0],
                label=data[1],
                weight=data[2] if len(data) == 3 else None,
                nthread=n_jobs)

            # Optuna integration
            if tune_model:
                _fit_params = fit_params.copy()
                if 'verbose_eval' in _fit_params.keys():
                    _fit_params.update({'verbose_eval': False})
                
                self.best_score = None
                self.best_model = None
                self.best_iteration = 0

                def xgb_objective(trial):
                    _params = params.copy()
                    if optuna_params is None:
                        _params = PARAMS_ZOO[self.model_name](trial, params)
                    else:
                        _params = optuna_params(trial, _params)
                    
                    models = []
                    model_extractor = XGBModelExtractor(models)
                    res = xgb.cv(
                        _params, dtrain=dtrain, folds=folds, maximize=maximize,
                        callbacks=[model_extractor], **_fit_params)
                    self.model = models

                    if eval_metric is None:
                        if maximize:
                            score = np.max(res[f'test-{main_metric}-mean'])
                            best_iteration = np.argmax(res[f'test-{main_metric}-mean'])
                        else:
                            score = np.min(res[f'test-{main_metric}-mean'])
                            best_iteration = np.argmin(res[f'test-{main_metric}-mean'])
                    else:
                        raise NotImplementedError('Do not use custom eval_metric for .cv() :(')

                    if self.best_score is None:
                        self.best_score = score
                        self.best_model = self.model.copy()
                        self.evals_result = res

                    if maximize and self.best_score < score:
                        self.best_score = score
                        self.best_model = self.model.copy()
                        self.best_iteration = best_iteration
                        self.evals_result = res
                    elif not maximize and self.best_score > score:
                        self.best_score = score
                        self.best_model = self.model.copy()
                        self.best_iteration = best_iteration
                        self.evals_result = res

                    return score

                study = optuna.create_study(
                    direction='maximize' if maximize else 'minimize')
                study.optimize(
                    xgb_objective, n_trials=n_trials, timeout=timeout, 
                    callbacks=[logger.optuna], n_jobs=1)
                self.model = self.best_model.copy()
                del self.best_model
            else:
                _params = params.copy()

                model_extractor = ModelExtractor()
                res = xgb.cv(
                    _params, dtrain=dtrain, folds=folds, maximize=maximize, 
                    callbacks=[logger.lgbm, model_extractor], **fit_params)
                self.model = model_extractor.get_model()

                if maximize:
                    self.best_score = np.max(
                        res[f'test-{main_metric}-mean'])
                    self.best_iteration = np.argmax(
                        res[f'test-{main_metric}-mean'])
                else:
                    self.best_score = np.min(
                        res[f'test-{main_metric}-mean'])
                    self.best_iteration = np.argmin(
                        res[f'test-{main_metric}-mean'])
                
                for i in range(len(self.model)):
                    self.model[i].best_ntree_limit = self.best_iteration
            
            logger(f'[{self.best_iteration}]\tbest score is {self.best_score:.6f}')

            if convert_to_sklearn:
                for i in range(len(self.model)):
                    self.model[i] = booster2sklearn(
                        self.model[i], self.model_type, data[0], data[1])
        
        else:
            raise NotImplementedError(f'{self.model_name} is incompatible with .cv().')

        self.is_trained = True

    def predict(self, X, **kwargs):
        assert self.is_trained
        if isinstance(self.model, list):
            predictions = []
            for i in range(len(self.model)):
                if self.model_name in MODEL_ZOO['xgb']:
                    predictions.append(
                        self.model[i].predict(
                            X, iteration_range=(0, self.get_best_iteration()), **kwargs))
                else:
                    predictions.append(self.model[i].predict(X, **kwargs))
            return predictions
        else:
            if self.model_name in MODEL_ZOO['xgb']:
                return self.model.predict(
                    X, iteration_range=(0, self.get_best_iteration()), **kwargs)
            else:
                return self.model.predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        assert self.is_trained
        if isinstance(self.model, list):
            predictions = []
            for i in range(len(self.model)):
                if self.model_name in MODEL_ZOO['xgb']:
                    predictions.append(
                        self.model[i].predict_proba(
                            X, iteration_range=(0, self.get_best_iteration()), **kwargs))
                else:
                    predictions.append(self.model[i].predict_proba(X, **kwargs))
            return predictions
        else:
            if self.model_name in MODEL_ZOO['xgb']:
                return self.model.predict_proba(
                    X, iteration_range=(0, self.get_best_iteration()), **kwargs)
            else:
                return self.model.predict_proba(X, **kwargs)

    def smart_predict(self, X, **kwargs):
        assert self.is_trained
        is_classifier = self.model_name[-10:] == 'Classifier' or \
            self.model_name in ['SVC', 'LogisticRegression']
        if is_classifier and self.n_classes == 2:
            prediction = self.predict_proba(X, **kwargs)
            if isinstance(prediction, list):
                return [p[:, 1] for p in prediction]
            else:
                return prediction[:, 1]
        else:
            return self.predict(X, **kwargs)

    def get_model(self):
        return self.model

    def get_best_iteration(self):
        return self.best_iteration

    def get_best_score(self):
        return self.best_score

    def _get_permutation_importance(self, model, fit_params):
        assert fit_params is not None
        res = permutation_importance(model, **fit_params)
        return res['importances_mean'], res['importances_std']

    def _get_feature_importance(self, model, importance_type='auto', normalize=True, fit_params=None):
        if self.model_name in MODEL_ZOO['cat']:
            ''' CatBoost '''
            if importance_type == 'auto':
                imp = model.get_feature_importance()
            elif importance_type == 'split':
                print(
                    f'model type {self.model_name} does not support importance type {importance_type}')
                imp = model.get_feature_importance()
            elif importance_type == 'gain':
                imp = model.get_feature_importance()
            elif importance_type == 'permutation':
                imp = model.get_permutation_importance(model, fit_params)[0]
            elif importance_type == 'null':
                # WIP
                pass
            else:
                raise ValueError(
                    f'Unsupported importance type {importance_type}')

        elif self.model_name in MODEL_ZOO['lgb']:
            ''' LightGBM '''
            if isinstance(self.model, lgb.Booster):
                _model = model
            else:
                _model = model._Booster

            if importance_type == 'auto':
                imp = _model.feature_importance(importance_type='split')
            elif importance_type == 'split':
                imp = _model.feature_importance(importance_type='split')
            elif importance_type == 'gain':
                imp = _model.feature_importance(importance_type='gain')
            elif importance_type == 'permutation':
                imp = self._get_permutation_importance(model, fit_params)[0]
            elif importance_type == 'null':
                # WIP
                pass
            else:
                raise ValueError(
                    f'Unsupported importance type {importance_type}')

        elif self.model_name in MODEL_ZOO['xgb']:
            ''' XGBoost '''
            if isinstance(self.model, xgb.Booster):
                _model = model
            else:
                _model = model.get_booster()

            if importance_type == 'auto':
                _imp = _model.get_score(importance_type='weight')
            elif importance_type == 'split':
                _imp = _model.get_score(importance_type='weight')
            elif importance_type == 'gain':
                _imp = _model.get_score(importance_type='gain')
            elif importance_type == 'permutation':
                imp = self._get_permutation_importance(model, fit_params)[0]
            elif importance_type == 'null':
                # WIP
                pass
            else:
                raise ValueError(
                    f'Unsupported importance type {importance_type}')
            try:
                imp
            except:
                imp = [_imp[f] if f in _imp.keys(
                ) else 0 for f in self.feature_names]

        else:
            ''' Others '''
            if importance_type in ['auto', 'split', 'gain']:
                try:
                    imp = model.feature_importances_
                except:
                    raise ValueError(
                        f'model type {self.model_name} does not support importance type {importance_type}')
            elif importance_type == 'permutation':
                imp = self._get_permutation_importance(model, fit_params)[0]
            elif importance_type == 'null':
                # WIP
                pass
            else:
                raise ValueError(
                    f'Unsupported importance type {importance_type}')

        if normalize:
            imp = vector_normalize(imp)

        return {self.feature_names[i]: imp[i] for i in range(len(self.feature_names))}

    def get_feature_importance(self, importance_type='auto', normalize=True, fit_params=None, 
                               as_pandas='auto'):
        if isinstance(self.model, list):
            self.feature_importance = [self._get_feature_importance(
                m, importance_type, normalize, fit_params) for m in self.model]
            if as_pandas in ['auto', True]:
                return pd.DataFrame(self.feature_importance)
            else:
                return self.feature_importance
        else:
            self.feature_importance = self._get_feature_importance(
                self.model, importance_type, normalize, fit_params)
            if as_pandas in ['auto', False]:
                return self.feature_importance
            else:
                return pd.DataFrame([self.feature_importance])

    def plot_feature_importance(self, importance_type='auto', normalize=True, fit_params=None, 
                                sort=True, size=5, save_to=None):
        imp_df = self.get_feature_importance(importance_type, normalize, fit_params, as_pandas=True)
        plt.figure(figsize=(size, imp_df.shape[1]/3))
        order = imp_df.mean().sort_values(ascending=False).index.tolist() \
            if sort else None
        sns.barplot(data=imp_df, orient='h', errorbar='sd',
                    order=order, palette="coolwarm")
        if save_to is not None:
            plt.savefig(save_to)
        plt.show()

    def plot_calibration_curve(self, data, predict_params={}, size=4, save_to=None):
        X, y = data[0], data[1]
        approx = self.smart_predict(X, **predict_params)
        fig = plt.figure(figsize=(size, size*1.5), tight_layout=True)
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
        sns.histplot(
            approx, bins=10, element="step", stat="density", common_norm=False, ax=ax2)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylabel('Density')
        if save_to is not None:
            plt.savefig(save_to)
        plt.show()

    def save(self, path):
        with open(path, 'wb') as f:
            snapshot = tuple([getattr(self, item) for item in self.snapshot_items])
            pickle.dump(snapshot, f)

    def load(self, path):
        with open(path, 'rb') as f:
            snapshot = pickle.load(f)
        for i, item in enumerate(self.snapshot_items):
            setattr(self, item, snapshot[i])

    def __repr__(self):
        desc = f'Trainer: {self.serial}\n'
        items = ['model', 'is_trained', 'best_iteration', 'best_score']
        for i in items:
            desc += f'{i}: {getattr(self, i)}\n'
        return desc

    def info(self):
        print(self.__repr__())
