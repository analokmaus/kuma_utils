import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm.auto import tqdm
from .base import PreprocessingTemplate
from .utils import analyze_column


class LGBMImputer(PreprocessingTemplate):
    '''
    Regression imputer using LightGBM
    '''

    def __init__(self, cat_features=[], n_iter=100, verbose=False):
        self.n_iter = n_iter
        self.cat_features = cat_features
        self.verbose = verbose
        self.n_features = None
        self.feature_names = None
        self.feature_with_missing = None
        self.imputers = {}
        self.offsets = {}
        self.objectives = {}

    def _analyze_feature(self, X, icol, col):
        if icol in self.cat_features:
            nuni = X[col].dropna().nunique()
            if nuni == 2:
                params = {
                    'objective': 'binary'
                }
            elif nuni > 2:
                params = {
                    'objective': 'multiclass',
                    'num_class': nuni + 1
                }
        else:
            if analyze_column(X[col]) == 'numerical':
                params = {
                    'objective': 'regression'
                }
            else:
                nuni = X[col].dropna().nunique()
                if nuni == 2:
                    params = {
                        'objective': 'binary'
                    }
                elif nuni > 2:
                    params = {
                        'objective': 'multiclass',
                        'num_class': nuni + 1
                    }
                else:
                    return None
        return params
    
    def _fit_lgb(self, X, col, params):
        null_idx = X[col].isnull()
        x_train = X.loc[~null_idx].drop(col, axis=1)
        y_offset = X[col].min()
        y_train = X.loc[~null_idx, col].astype(int) - y_offset
        dtrain = lgb.Dataset(
            data=x_train,
            label=y_train
        )
        model = lgb.train(
            params, dtrain,
            num_boost_round=self.n_iter,
        )
        return model, y_offset, null_idx

    def fit(self, X: pd.DataFrame, y=None):
        self.n_features = X.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f'f{i}' for i in range(self.n_features)]
            X = pd.DataFrame(X, columns=self.feature_names)
        self.feature_with_missing = [
            col for col in self.feature_names if X[col].isnull().sum() > 0]
        
        if self.verbose:
            pbar = tqdm(self.feature_with_missing)
            iterator = enumerate(pbar)
        else:
            iterator = enumerate(self.feature_with_missing)

        for icol, col in iterator:
            params = self._analyze_feature(X, icol, col)
            if params is None:
                continue
            params['verbosity'] = -1
            model, y_offset, null_idx = self._fit_lgb(X, col, params)
            # x_test = X.loc[null_idx].drop(col, axis=1)

            self.imputers[col] = model
            self.offsets[col] = y_offset
            self.objectives[col] = params['objective']
            if self.verbose:
                pbar.set_description(
                    f'{col}:\t{self.objectives[col]}...iter{model.best_iteration}/{self.n_iter}')

    def transform(self, X: pd.DataFrame):
        output_X = X.copy()

        for col in self.feature_with_missing:
            model = self.imputers[col]
            y_offset = self.offsets[col]
            objective = self.objectives[col]

            null_idx = X[col].isnull()
            x_test = X.loc[null_idx].drop(col, axis=1)

            y_test = model.predict(x_test)
            if objective == 'multiclass':
                y_test = np.argmax(y_test, axis=1).astype(float)
            elif objective == 'binary':
                y_test = (y_test > 0.5).astype(float)
            y_test += y_offset
            output_X.loc[null_idx, col] = y_test
            if objective in ['multiclass', 'binary']:
                output_X[col] = output_X[col].astype(int)

        return output_X
        
    def fit_transform(self, X: pd.DataFrame, y=None):
        self.n_features = X.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f'f{i}' for i in range(self.n_features)]
            X = pd.DataFrame(X, columns=self.feature_names)

        output_X = X.copy()
        self.feature_with_missing = [col for col in self.feature_names if X[col].isnull().sum() > 0]

        if self.verbose:
            pbar = tqdm(self.feature_with_missing)
            iterator = enumerate(pbar)
        else:
            iterator = enumerate(self.feature_with_missing)

        for icol, col in iterator:
            params = self._analyze_feature(X, icol, col)
            if params is None:
                continue
            params['verbosity'] = -1
            
            model, y_offset, null_idx = self._fit_lgb(X, col, params)
            x_test = X.loc[null_idx].drop(col, axis=1)
            y_test = model.predict(x_test)
            if params['objective'] == 'multiclass':
                y_test = np.argmax(y_test, axis=1).astype(float)
            elif params['objective'] == 'binary':
                y_test = (y_test > 0.5).astype(float)
            y_test += y_offset
            output_X.loc[null_idx, col] = y_test
            if params['objective'] in ['multiclass', 'binary']:
                output_X[col] = output_X[col].astype(int)
            self.imputers[col] = model
            self.offsets[col] = y_offset
            self.objectives[col] = params['objective']
            if self.verbose:
                pbar.set_description(f'{col}:\t{self.objectives[col]}...iter{model.best_iteration}/{self.n_iter}')
        
        return output_X
