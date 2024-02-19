import os
import random
import numpy as np
from sklearn import metrics

# metric define
def MAE(true, pred):
    score = np.mean(np.abs(true-pred))
    return score
def RMSE(true, pred):
    score = metrics.mean_squared_error(true, pred, squared=False)
    return score
def NMAE(true, pred):
    mae = np.mean(np.abs(true-pred))
    score = mae / np.mean(np.abs(true))
    return score
def R2(true, pred):
    score = metrics.r2_score(true, pred)
    return score

# seed setting
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)