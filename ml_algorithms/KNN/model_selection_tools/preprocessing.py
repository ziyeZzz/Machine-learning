import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, X):
        # 根据训练数据X计算出X每个维度上的均值和方差
        assert X.ndim == 2, "the dimension of X must be 2" # 这里只算二维

        self.mean_ = np.array([np.mean(X[:,i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:,i]) for i in range(X.shape[1])])

        return self

    def transform(self, X):
        # 将X 进行方差均值归一化
        assert X.ndim == 2, "the dimension of X must be 2"
        assert self.mean_ is not None and self.scale_ is not None, \
            "must fit before transform!"
        
        resX = np.empty(shape = X.shape, dtype= float)
        for col in range(X.shape[1]):
            resX[:,col] = (X[:,col] - self.mean_[col]) / self.scale_[col]
        
        return resX