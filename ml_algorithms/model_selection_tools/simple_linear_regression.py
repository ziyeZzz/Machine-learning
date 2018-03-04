import numpy as np
from .metrics import r2_score

class SimpleLinearRegression1:
    
    def __init__(self):
        # initialize simple linear regression model
        self.a_ = None # '_'means the variable is not given by user, but calculate by program
        self.b_ = None
    
    def fit(self, x_train, y_train):
        # train simple linear regression model by x_train, y_train
        assert x_train.ndim == 1, \
            "Simple linear regression can only solve single feature training data"
        assert len(x_train) == len(y_train), \
            "the size of x_train should be equal to y_train"
        
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0

        for x,y in zip(x_train, y_train):
            num += (x - x_mean)*(y - y_mean)
            d += (x - x_mean) ** 2
        
        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self
    
    def predict(self, x_predict):
        #return result vector according to x_predict
        assert x_predict.ndim == 1, \
            "simple linear regression can only solve single feature training data"
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict"
        
        return np.array([self._predict(x) for x in x_predict])
    
    def _predict(self, x_single):
        return self.a_ * x_single + self.b_
    
    def __repr__(self):
        return "SimpleLinearRegression1()"

#simple linear regression 2
class SimpleLinearRegression:
    
    def __init__(self):
        # initialize simple linear regression model
        self.a_ = None # '_'means the variable is not given by user, but calculate by program
        self.b_ = None
    
    def fit(self, x_train, y_train):
        # train simple linear regression model by x_train, y_train
        assert x_train.ndim == 1, \
            "Simple linear regression can only solve single feature training data"
        assert len(x_train) == len(y_train), \
            "the size of x_train should be equal to y_train"
        
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self
    
    def predict(self, x_predict):
        #return result vector according to x_predict
        assert x_predict.ndim == 1, \
            "simple linear regression can only solve single feature training data"
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict"
        
        return np.array([self._predict(x) for x in x_predict])
    
    def _predict(self, x_single):
        return self.a_ * x_single + self.b_
    
    def score(self, x_test, y_test):
        #calculate accuracy according to test dataset

        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)
    
    def __repr__(self):
        return "SimpleLinearRegression2()"
