import torch
from sklearn.metrics import roc_auc_score


def auc(_y, y):
    _y = np.squeeze(_y.data.cpu().numpy())
    y = np.squeeze(y.data.cpu().numpy())
    return roc_auc_score(y, _y)


def accuracy(_y, y):
    _y = _y.data.cpu().numpy()
    _y = np.argmax(_y, axis=1)
    y = y.data.cpu().numpy()
    return np.mean(_y == y)
