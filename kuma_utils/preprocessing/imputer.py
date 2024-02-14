import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from tqdm.auto import tqdm
from .base import PreprocessingTemplate
from .utils import analyze_column


class LGBMImputer(PreprocessingTemplate):
    '''
    Regression imputer using LightGBM

    ## Arguments
    target_cols: default = []

    Specify additional columns to impute for cases where you want to include features without missing values,
    such as when missing values exist only in the test data.

    cat_features: default = []

    By default, the algorithm will detect feature type (categorical or numerical) based on the data type.
    Columns specified in this argument will be considered categorical regardless of its data type.

    fit_params: default = {'num_boost_round': 100}

    Parameters for `lightgbm.train()`, such as callbacks, can be added.

    fit_method: default = 'fit'

    In case you want to run cross validation, set this to 'cv'.

    verbose: default = False
    
    Turning on verbose allows you to visualize the fitting process, providing a sense of reassurance.
    '''
    def __init__(
            self,
            target_cols: list[str] = [],
            cat_features: list[str, int] = [],
            fit_params: dict = {'num_boost_round': 100},
            fit_method: str = 'fit',
            verbose: bool = False):
        self.target_cols = target_cols
        self.cat_features = cat_features
        self.fit_params = fit_params
        self.fit_method = fit_method
        self.verbose = verbose
        self.n_features = None
        self.feature_names = None
        self._feature_scanned = False
        self.imputers = {}
        self.cat_encoder = OrdinalEncoder()

    def _transform_dataframe(self, X: pd.DataFrame, fit: bool = False):
        if fit:
            self.n_features = X.shape[1]
            if isinstance(X, pd.DataFrame):
                self.feature_names = X.columns.tolist()
            else:
                self.feature_names = [f'f{i}' for i in range(self.n_features)]
                X = pd.DataFrame(X, columns=self.feature_names)
            if len(self.cat_features) > 0 and isinstance(self.cat_features[0], int):
                self.cat_features = [self.feature_names[i] for i in self.cat_features]
        else:
            if isinstance(X, pd.DataFrame):
                pass
            else:
                assert X.shape[1] == self.n_features
                X = pd.DataFrame(X, columns=self.feature_names)
        return X.copy()

    def _scan_features(self, X: pd.DataFrame):
        self.col_dict = {}
        self.target_columns = []
        self.categorical_columns = []

        for col in X.columns:
            self.col_dict[col] = {}
            col_arr = X[col]
            col_type = analyze_column(col_arr)
            if col in self.cat_features:  # override
                col_type = 'categorical'
            self.col_dict[col]['col_type'] = col_type

            if col_type == 'categorical':
                num_class = col_arr.dropna().nunique()
                self.col_dict[col]['num_class'] = num_class
                if num_class == 2:
                    self.col_dict[col]['params'] = {
                        'objective': 'binary'
                    }
                elif num_class > 2:
                    self.col_dict[col]['params'] = {
                        'objective': 'multiclass',
                        'num_class': num_class
                    }
                elif num_class == 1:
                    self.col_dict[col]['params'] = None
                self.categorical_columns.append(col)
            else:  # numerical features
                self.col_dict[col]['params'] = {
                    'objective': 'regression'
                }

            null_mask = col_arr.isnull()
            is_target = null_mask.sum() > 0
            if col in self.target_cols:  # override
                is_target = True
            self.col_dict[col]['null_mask'] = null_mask
            if is_target:
                self.target_columns.append(col)

        self._feature_scanned = True
    
    def _fit_lgb(self, X: pd.DataFrame, col: str):
        col_info = self.col_dict[col]
        null_mask = col_info['null_mask']
        _categorical_columns = self.categorical_columns.copy()
        if col in _categorical_columns:
            _categorical_columns.remove(col)
        if col_info['params'] is None:  # Single class
            model = SimpleImputer()
            model.fit(X[col])
        else:
            params = col_info['params']
            params['verbose'] = -1
            x_train = X.loc[~null_mask].drop(col, axis=1)
            y_train = X.loc[~null_mask, col].copy()
            dtrain = lgb.Dataset(data=x_train, label=y_train)
            if self.fit_method == 'fit':
                model = lgb.train(
                    params, dtrain, valid_sets=[dtrain],
                    categorical_feature=_categorical_columns, **self.fit_params)
            elif self.fit_method == 'cv':
                res = lgb.cv(
                    params, dtrain, return_cvbooster=True,
                    stratified=False if col_info['params']['objective'] == 'regression' else True,
                    categorical_feature=_categorical_columns,
                    **self.fit_params)
                model = res['cvbooster']
        return model

    def fit(self, X: pd.DataFrame, y=None):
        X = self._transform_dataframe(X, fit=True)
        self._scan_features(X)
        X[self.categorical_columns] = self.cat_encoder.fit_transform(X[self.categorical_columns])
        X[self.categorical_columns] = X[self.categorical_columns].infer_objects(copy=False)
        
        if self.verbose:
            pbar = tqdm(self.target_columns)
            iterator = enumerate(pbar)
        else:
            iterator = enumerate(self.target_columns)

        for _, col in iterator:
            model = self._fit_lgb(X, col)
            self.imputers[col] = model
            if self.verbose:
                pbar.set_description(col)

    def transform(self, X: pd.DataFrame):
        assert self._feature_scanned
        output_X = self._transform_dataframe(X, fit=False)
        output_X[self.categorical_columns] = self.cat_encoder.transform(output_X[self.categorical_columns])
        
        for col in self.target_columns:
            if self.col_dict[col]['params'] is None:
                model = self.imputers[col]
                output_X[col] = model.transform(output_X[col])
            else:
                objective = self.col_dict[col]['params']['objective']
                model = self.imputers[col]
                null_mask = output_X[col].isnull()
                x_test = output_X.loc[null_mask].drop(col, axis=1)
                y_test = model.predict(x_test)
                if self.fit_method == 'cv':
                    y_test = np.mean(y_test, axis=0)
                if objective == 'multiclass':
                    y_test = np.argmax(y_test, axis=1).astype(float)
                elif objective == 'binary':
                    y_test = (y_test > 0.5).astype(float)
                output_X.loc[null_mask, col] = y_test

        output_X[self.categorical_columns] = self.cat_encoder.inverse_transform(output_X[self.categorical_columns])
        return output_X
        
    def fit_transform(self, X: pd.DataFrame, y=None):
        self.fit(X)
        return self.transform(X)
