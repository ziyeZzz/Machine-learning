import numpy as np

def accuracy_score(y_true, y_predict):
    # calculate the accuracy of y_predict
    # y_true长度应该和y_predict一致
    assert y_true.shape[0] == y_predict.shape[0], \
        "the size of y_true must be the same as y_predict"
    
    return sum(y_predict == y_true) / len(y_true)