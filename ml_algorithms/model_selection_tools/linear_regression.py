import numpy as np
from .metrics import r2_score

class LinearRegression:

    def __init__(self):
        #initialize linear regression model
        self.coef_ = None
        self.interception_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        X_b = np.hstack([np.ones((len(X_train),1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self
    
    def predict(self, X_predict):
        # return y_predict according to X_predict
        assert self.interception_ is not None and self.coef_ is not None, \
            "must fit before predict"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature numbers of X_predict must be equal to the X_train"
        X_b = np.hstack([np.ones((len(X_predict),1)), X_predict])
        return X_b.dot(self._theta)
    
    def score(self, X_test, y_test):
        #calculate accuracy according to X_test and y_test
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)
    
    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        
        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')
        
        def dJ(theta, X_b, y):
            # res = np.empty(len(theta))
            # res[0] = np.sum(X_b.dot(theta)-y)
            # for i in range(1, len(theta)):
            #     res[i] = (X_b.dot(theta) - y).dot(X_b[:,i])
            # return res * 2 /len(X_b)
            return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(X_b) # /len(y)也是一样的
        
        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if(abs(J(theta, X_b, y) - J(last_theta, X_b, y))< epsilon):
                    break
                cur_iter += 1
            
            return theta
        
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def __repr__(self):
        return "LinearRegression()"