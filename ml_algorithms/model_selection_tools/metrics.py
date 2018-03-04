import numpy as np
from math import sqrt

def accuracy_score(y_true, y_predict):
    # calculate the accuracy of y_predict
    # y_true长度应该和y_predict一致
    assert y_true.shape[0] == y_predict.shape[0], \
        "the size of y_true must be the same as y_predict"
    
    return sum(y_predict == y_true) / len(y_true)

def mean_squared_error(y_true, y_predict):
    # 计算y_true, y_predict之间的MSE
    assert len(y_true) == len(y_predict), \
        "the size of y_true must equal to y_predict"

    return np.sum((y_true - y_predict) ** 2) / len(y_true)

def root_mean_squared_error(y_true, y_predict):
    #calculate RMSE between y_true and y_predict
    return sqrt(mean_squared_error(y_true, y_predict))

def mean_absolute_error(y_true, y_predict):
    #calculate MAE between y_ture and y_predict
    assert len(y_true) == len(y_predict), \
        "the size of y_true must equal to y_predict"
    
    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)

def r2_score(y_true, y_predict):
    # calculate r square between y_ture and y_predict
    
    return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)