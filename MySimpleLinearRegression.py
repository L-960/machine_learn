import numpy as np


class MySimpleLinearRegression:
    def __init__(self):
        self.a = None
        self.b = None
        self.y_predict = None

    def fit(self, x_train, y_train):
        '''
        训练
        :param x_train:训练集（拥有一个特征值，多行的矩阵）
        :param y_train:训练集对应的标签
        :return: 返回本身
        '''
        # a向量化
        self.a = (x_train - np.mean(x_train)).dot(y_train - np.mean(y_train)) / \
                 (x_train - np.mean(x_train)).dot(x_train - np.mean(x_train))

        # b
        self.b = np.mean(y_train) - self.a * np.mean(x_train)

        return self

    def predict(self, x_test):
        '''
        返回预测数据集合
        :param x_test:测试数据集
        :return: 返回预测结果集
        '''
        res = np.array([self.mysingle(i) for i in x_test])
        self.y_predict = res
        return self.y_predict

    def mysingle(self, x):
        return self.a * x + self.b

    # 均方误差
    def mse(self, y_test):
        return np.sum((np.array(y_test) - self.y_predict) ** 2) / len(y_test)

    # 均方根误差
    def rmse(self, y_test):
        return np.sqrt(np.sum((np.array(y_test) - self.y_predict) ** 2) / len(y_test))

    # 平均绝对误差
    def mae(self, y_test):
        return np.sum(np.absolute(np.array(y_test) - self.y_predict)) / len(y_test)
