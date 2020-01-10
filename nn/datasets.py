import numpy as np
import pandas as pd
from pathlib import Path
from copy import copy
import random
from collections import Counter, defaultdict

from PIL import Image

import torch
import torch.utils.data as D

from ..common import KumaNumpy as kn


def category2embedding(df, categorical_features, dim='auto'):
    df = kn.to_numpy(df)
    cat_dims = [int(kn.nunique(df[:, col])) for col in categorical_features]
    if dim == 'auto':
        emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]
    elif isinstance(dim, int):
        emb_dims = [(x, dim) for x in cat_dims]
    else:
        raise ValueError(dim)
    return emb_dims
