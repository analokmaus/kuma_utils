import pandas as pd


class PreprocessingTemplate:

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series):
        pass

    def transform(self, X: pd.DataFrame):
        pass

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
        pass
    
    def __repr__(self):
        return self.__class__.__name__
    