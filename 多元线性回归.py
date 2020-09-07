import numpy as np


# 封装自己的多元线性回归方法
class MyLinearRegression:
    # 初始化
    def __init__(self):
        self.theta = None
        self.interception_ = None
        self.theta_ = None

    def fit(self, X, y):
        # 合成X_b
        X_ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((X_ones, X))
        self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.interception_ = self.theta[0]
        self.theta_ = self.theta[1:]
        return self

    def predict(self, X_test):
        return X_test.dot(self.theta_) + self.interception_

    def coef_(self):
        return self.theta_

    def intercept_(self):
        return self.interception_

    def r2_score(self, X_test, y_test):
        mes = np.sum((self.predict(X_test) - y_test) ** 2) / X_test.shape[0]
        va = np.sum((y_test - np.mean(y)) ** 2) / X_test.shape[0]
        re = 1 - (mes / va)
        return re
