import numpy as np
from math import sqrt
from collections import Counter

class KNNClassifier:
    def __init__(self, k):
        # 初始化KNN分类器
        assert k>=1, "k must be valid"
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        #根据训练数据集训练KNN分类器
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert self.k <= X_train.shape[0], \
            "k must be valid"
        
        self._X_train = X_train
        self._y_train = y_train
        return self
    
    def predict(self, X_predict):
        # 给定待预测数据集X_predict, 返回X_predict的结果向量
        assert self._X_train is not None and self._y_train is not None, \
            "must fit before predict"
        assert X_predict.shape[1] == self._X_train.shape[1], \
            "the features number must be the same"
        
        y_predict = [self._predict(x) for x in X_predict]
        return y_predict
    
    def _predict(self, x):
        # 给定单个待预测数据x，返回x的预测结果值
        assert x.shape[0] == self._X_train.shape[1],\
            "the feature number of x must equal to X_train"# 其实前面已经检查过了
        distances = [sqrt(np.sum((x_train - x)**2)) for x_train in self._X_train]
        nearest = np.argsort(distances)
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]

    def __repr__(self):
        # 显示k等于多少
        return "KNN(K=%d)" % self.k