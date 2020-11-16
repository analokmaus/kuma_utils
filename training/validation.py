import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .trainer import Trainer
from .logger import LGBMLogger

try:
    import optuna
    import optuna.integration.lightgbm as lgb_tune
except ModuleNotFoundError:
    print('optuna not found.')

from pathlib import Path


class CrossValidator:
    '''
    Amazing cross validation wrapper for sklearn models

    Some useful features:
    - Automated parameter tuning using Optuna
    '''

    def __init__(self, model=None, path=None, serial='cv0'):
        self.model = model
        self.serial = serial

    def _data_check(self, data):
        assert isinstance(data, (list, tuple))
        assert len(data) >= 2
        for i, d in enumerate(data):
            assert isinstance(d, (pd.Series, pd.DataFrame, np.ndarray))
            if i == 0:
                dshape = len(d)
            assert dshape == len(d)
            dshape = len(d)

    def _split_data(self, data, idx):
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.iloc[idx]
        else:
            return data[idx]

    def train(self,
              # Dataset
              data, cat_features=None, groups=None, folds=KFold(n_splits=5), 
              # Model
              params={}, fit_params={},
              # Optuna
              tune_model=False, optuna_params=None, maximize=True,
              eval_metric=None, n_trials=None, timeout=None, 
              # Misc
              logger=None, n_jobs=-1):

        self.nfold = folds.get_n_splits()
        self.models = []

        if isinstance(logger, (str, Path)):
            _logger = LGBMLogger(logger, params, fit_params)
        elif isinstance(logger, LGBMLogger):
            _logger = logger
        elif logger is None:
            _logger = None
        else:
            raise ValueError('invalid logger.')

        _fit_params = fit_params.copy()
        if 'verbose_eval' in fit_params.keys():
            _fit_params.update({'verbose_eval': False})

        self.scores = []

        for fold_i, (train_idx, valid_idx) in enumerate(
            folds.split(X=data[0], y=data[1], groups=groups)):
            
            print(f'[{self.serial}] Starting fold {fold_i}')
            if _logger is not None:
                _logger.write(f'[{self.serial}] Starting fold {fold_i}\n')
            
            train_data = [self._split_data(d, train_idx) for d in data]
            valid_data = [self._split_data(d, valid_idx) for d in data]

            trn = Trainer(self.model, serial=f'{self.serial}_fold{fold_i}')
            trn.train(
                train_data=train_data, valid_data=valid_data, cat_features=cat_features,
                params=params, fit_params=_fit_params,
                tune_model=False, # do not tune model internally
                maximize=maximize, eval_metric=eval_metric,
                logger=_logger, n_jobs=n_jobs
            )
            self.models.append(trn)

            best_score = trn.get_best_score()
            best_iter = trn.get_best_iteration()
            print(f'[{self.serial}] Fold {fold_i}: metric={best_score:.6f} | iter={best_iter}')
            if _logger is not None:
                _logger.write(
                    f'[{self.serial}] Fold {fold_i}: metric={best_score:.6f} | iter={best_iter}\n')
            self.scores.append(best_score)

        mean_score = np.mean(self.scores)
        se_score = np.std(self.scores) / np.sqrt(self.nfold)
        print(f'[{self.serial}] Overall metric: {mean_score:.6f} +- {se_score:.6f}')
        if _logger is not None:
            _logger.write(
                f'[{self.serial}] Overall metric: {mean_score:.6f} +- {se_score:.6f}\n')
