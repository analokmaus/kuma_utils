from .base import PreprocessingTemplate


class Cast(PreprocessingTemplate):
    '''
    '''
    def __init__(self, dtype):
        self.dtype = dtype
        
    def fit(self, X, y=None) -> None:
        raise RuntimeError("fit() is not supported.")
        
    def transform(self, X):
        return X.copy().astype(self.dtype)

    def fit_transform(self, X, y=None):
        return X.copy().astype(self.dtype)
