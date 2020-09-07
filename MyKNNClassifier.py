import numpy as np
from collections import Counter



class MyKNNClassifier:
    def __init__(self, k=3):
        assert k >= 1, 'k must >=1'
        self.k = k
        self.x_train = None
        self.y_train = None
        self.predict_data = None

    def fit(self, x_train, y_train):
        '''
        提供训练方法
        '''
        assert x_train.shape[0] == y_train.shape[0], '数据维度必须相同'
        assert self.k <= y_train.shape[0], 'k值必须小于等于数据长度'
        self.x_train = x_train
        self.y_train = y_train
        return self

    def predict(self, x):
        # self._predict(x)
        '''
        获取分类向量并返回
        :return:
        '''
        y_predict = [self._predict(p) for p in x]
        self.predict_data = np.array(y_predict)
        return self.predict_data

    # 私有方法
    def _predict(self, x):
        # 按照待预测x点与已知的所属分类y去获取欧拉距离的列表值
        olist = [np.sqrt(np.sum((p - x) ** 2)) for p in self.x_train]
        # 获取排好序的索引 # 找最近的k个点
        sindex = np.argsort(olist)[:self.k]
        data = self.y_train[sindex]
        # 获取预测值,返回最多的一个
        return Counter(data).most_common(1)[0][0]

    # 获取预测完的数据集与传入结果集比对，生成得分
    def score(self, x_test, y_test):
        return np.sum(y_test == self.predict_data) / y_test.shape[0]


# 封装自己的train_test_split算法
def my_train_test_split(x, y, test_ratio=0.2):
    # permutation(n) 给出从0到n-1的一个不重复随机排列
    shuffle_indexes = np.random.permutation(x.shape[0])
    # 测试集数量
    test_size = int(x.shape[0] * test_ratio)
    # 训练集数量
    train_size = int(x.shape[0] * (1 - test_ratio))

    # 测试集 训练集 索引
    test_indexes = shuffle_indexes[:test_size]
    train_indexes = shuffle_indexes[train_size:]

    # 根据索引取数据
    x_train = x[train_indexes]
    y_train = y[train_indexes]
    x_test = x[test_indexes]
    y_test = y[test_indexes]
    return x_train, x_test, y_train, y_test
