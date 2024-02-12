import pandas as pd
from .base import PreprocessingTemplate
from .utils import analyze_column


class SelectNumerical(PreprocessingTemplate):
    '''
    '''
    def __init__(self, include_cols: list = [], exclude_cols: list = []):
        self.include_cols = include_cols
        self.exclude_cols = exclude_cols

    def fit(self, X: pd.DataFrame, y=None) -> None:
        raise RuntimeError("fit() is not supported.")
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if len(self.include_cols) == 0:
            select_cols = [col for col in X.columns if \
                           (analyze_column(X[col]) == 'numerical') and (col not in self.exclude_cols)]
        else:
            select_cols = self.include_cols
        return X[select_cols].copy()

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.transform(X)


class SelectCategorical(PreprocessingTemplate):
    '''
    '''
    def __init__(self, include_cols: list = [], exclude_cols: list = []):
        self.include_cols = include_cols
        self.exclude_cols = exclude_cols
        
    def fit(self, X: pd.DataFrame, y=None) -> None:
        raise RuntimeError("fit() is not supported.")
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if len(self.include_cols) == 0:
            select_cols = [col for col in X.columns if \
                           (analyze_column(X[col]) == 'categorical') and (col not in self.exclude_cols)]
        else:
            select_cols = self.include_cols
        return X[select_cols].copy()

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.transform(X)


class DummyVarible(PreprocessingTemplate):

    def __init__(self, include_cols: list = [], exclude_cols: list = [], dummy_na: bool = False):
        self.include_cols = include_cols
        self.exclude_cols = exclude_cols
        self.dummy_na = dummy_na

    def fit(self, X: pd.DataFrame, y=None) -> None:
        raise RuntimeError("fit() is not supported.")
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_new = X.copy()
        if len(self.include_cols) == 0:
            target_cols = [col for col in X.columns if col not in self.exclude_cols]
        else:
            target_cols = self.include_cols
        X_new = pd.concat([
            X, pd.get_dummies(
                X[target_cols], dummy_na=self.dummy_na, prefix=target_cols, prefix_sep='=').astype(int)], axis=1)
        X_new.drop(target_cols, axis=1, inplace=True)
        return X_new

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.transform(X)
