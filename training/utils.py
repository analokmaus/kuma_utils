import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error


def booster2sklearn(booster, model, n_features, n_classes):
    assert isinstance(booster, (lgb.Booster, xgb.Booster))
    new_model = model()
    new_model._Booster = booster
    new_model._n_features = n_features
    new_model._n_classes = n_classes
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
