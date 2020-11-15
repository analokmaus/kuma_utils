import lightgbm as lgb
import xgboost as xgb


def booster2sklearn(booster, model, n_features, n_classes):
    assert isinstance(booster, (lgb.Booster, xgb.Booster))
    new_model = model()
    new_model._Booster = booster
    new_model._n_features = n_features
    new_model._n_classes = n_classes
    return new_model
