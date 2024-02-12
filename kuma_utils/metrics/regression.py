import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from .base import MetricTemplate


class RMSE(MetricTemplate):
    '''
    Root mean square error
    '''
    def __init__(self):
        super().__init__(maximize=False)

    def _test(self, target, approx):
        return mean_squared_error(target, approx, squared=False)


class PearsonCorr(MetricTemplate):
    '''
    Pearson product-moment correlation coefficients
    '''
    def __init__(self):
        super().__init__(maximize=False)

    def _test(self, target, approx):
        return np.corrcoef(target, approx)[0, 1]


class R2Score(MetricTemplate):
    '''
    Coefficient of determination score
    '''
    def __init__(self):
        super().__init__(maximize=False)

    def _test(self, target, approx):
        return r2_score(target, approx)
