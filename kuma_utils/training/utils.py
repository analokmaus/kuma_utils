import numpy as np
import lightgbm as lgb
from lightgbm.compat import _LGBMLabelEncoder
from xgboost.compat import XGBoostLabelEncoder
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, mean_squared_error, mean_absolute_error)


def booster2sklearn(booster, model, X, y):
    assert isinstance(booster, (lgb.Booster, xgb.Booster))
    new_model = model()
    new_model._Booster = booster
    new_model._n_features = X.shape[1]
    new_model.fitted_ = True
    if new_model.__class__.__name__ == 'LGBMClassifier':
        new_model._le = _LGBMLabelEncoder().fit(y)
        new_model._class_map = dict(zip(
            new_model._le.classes_,
            new_model._le.transform(new_model._le.classes_)))
        new_model._classes = new_model._le.classes_
        new_model._n_classes = len(new_model._classes)
    elif new_model.__class__.__name__ == 'XGBClassifier':
        new_model._le = XGBoostLabelEncoder().fit(y)
        new_model.classes_ = new_model._le.classes_
    return new_model


def acc_metric(model, data):
    if isinstance(model, (lgb.Booster, xgb.Booster)):
        target = data.get_label()
        approx = (model.predict(data) >= 0.5).astype(int)
    else:
        target = data[1]
        approx = model.predict(data[0])
    return np.mean(target == approx)


def auc_metric(model, data):
    if isinstance(model, (lgb.Booster, xgb.Booster)):
        target = data.get_label()
        approx = model.predict(data)
    else:
        target = data[1]
        approx = model.predict_proba(data[0])[:, 1]
    return roc_auc_score(target, approx)


def mae_metric(model, data):
    if isinstance(model, (lgb.Booster, xgb.Booster)):
        target = data.get_label()
        approx = model.predict(data)
    else:
        target = data[1]
        approx = model.predict(data[0])
    return mean_absolute_error(target, approx)


def mse_metric(model, data):
    if isinstance(model, (lgb.Booster, xgb.Booster)):
        target = data.get_label()
        approx = model.predict(data)
    else:
        target = data[1]
        approx = model.predict(data[0])
    return mean_squared_error(target, approx)


def rmse_metric(model, data):
    if isinstance(model, (lgb.Booster, xgb.Booster)):
        target = data.get_label()
        approx = model.predict(data)
    else:
        target = data[1]
        approx = model.predict(data[0])
    return np.sqrt(mean_squared_error(target, approx))


class ModelExtractor:
    '''
    Model extractor for lightgbm and xgboost .cv()
    '''

    def __init__(self):
        self.model = None

    def __call__(self, env):
        if env.model is not None:
            # lightgbm
            self.model = env.model
        else:
            # xgboost
            self.model = [cvpack.bst for cvpack in env.cvfolds]

    def get_model(self):
        return self.model

    def get_best_iteration(self):
        return self.model.best_iteration


class XGBModelExtractor(xgb.callback.TrainingCallback):
    def __init__(self, cvboosters):
        self._cvboosters = cvboosters
    
    def after_training(self, model):
        self._cvboosters[:] = [cvpack.bst for cvpack in model.cvfolds]
        return model
