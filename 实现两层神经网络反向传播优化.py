class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 损失
        self.y = None  # softmax的输出
        self.t = None  # 监督数据（one-hot vector）

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


import numpy as np


# 接受一个np.array
def one_hot(y):
    # 创建独热编码的存储对象
    re = np.zeros((y.shape[0], 10))
    for i in range(y.shape[0]):
        re[i, y[i]] = 1

    return re


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 激活函数
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


# 交叉熵误差：损失函数
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


# 对接收函数求偏导：梯度
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    # numpy中对矩阵进行遍历;
    # flags=['multi_index']:表示对矩阵x进行多重索引
    # op_flags=['readwrite']:表示不仅可以对矩阵x进行read，还可以进行write；
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    # 迭代,it.finished属性判断迭代器是否已经结束
    while not it.finished:
        # 返回当前矩阵元素的索引(行,列)
        idx = it.multi_index
        temp_val = x[idx]  # 当前索引元素

        x[idx] = float(temp_val) + h
        f1 = f(x)

        x[idx] = float(temp_val) - h
        f2 = f(x)

        grad[idx] = (f1 - f2) / (2 * h)
        # 还原idx对应x位置的值
        x[idx] = temp_val
        # 进行下一次迭代
        it.iternext()
    return grad


import os
import gzip
import pickle


def load_mnist(is_one_hot=True, flat=True, normalize=True):
    """
    :param one_host: 是否对标签进行独热编码
    :param flat: 是否对图像数据扁平化处理，把64*64压扁784
    :param normalize:是否将图像数据进行正规化处理，输入图像是像素是否保持0-255
    :return:
    """
    # 定位文件路径
    # 定位文件路径
    base_path = os.path.realpath('')

    base_path = base_path + '/MNIST/'

    flist = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz',
    ]

    # 读取压缩文件 压缩文件需要使用gzip读取
    with gzip.open(base_path + flist[0], 'rb') as handle:
        # 读取二进制
        # offset移位
        # MNIST 的图像数据是 28 像素 × 28 像素的灰度图像 所有列是784
        train_x = np.frombuffer(handle.read(), dtype=np.uint8, offset=16).reshape(-1, 784)

    with gzip.open(base_path + flist[1], 'rb') as handle:
        train_y = np.frombuffer(handle.read(), dtype=np.uint8, offset=8)

    with gzip.open(base_path + flist[2], 'rb') as handle:
        test_x = np.frombuffer(handle.read(), dtype=np.uint8, offset=16).reshape(-1, 784)

    with gzip.open(base_path + flist[3], 'rb') as handle:
        test_y = np.frombuffer(handle.read(), dtype=np.uint8, offset=8)

    if is_one_hot:
        train_y = one_hot(train_y)
        test_y = one_hot(test_y)

    if not flat:
        train_x = train_x.reshape(-1, 28, 28)
        test_x = test_x.reshape(-1, 28, 28)

    if normalize:
        train_x = train_x / 255
        test_x = test_x / 255

    return (train_x, train_y), (test_x, test_y)


import sys, os

sys.path.append(os.pardir)
import numpy as np
from collections import OrderedDict


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict()
        # 第一个仿射层
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()

        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        # 输出层
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # x:输入数据, t:监督数据
    # 损失函数：通过最后一层得到交叉熵误差
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x:输入数据, t:监督数据
    # 基于数值计算批量梯度下降法
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    # 反向传播
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        # 在反向传播中 获取修正后存储的W和B。
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads


if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = load_mnist()

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    iters_num = 10000
    train_size = train_x.shape[0]
    batch_size = 100
    learning_rate = 0.1
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = train_x[batch_mask]
        t_batch = train_y[batch_mask]

        # 通过误差反向传播法求梯度
        grad = network.gradient(x_batch, t_batch)

        # 更新
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        print(loss)

    # 随机取一张图片
    index = np.random.randint(test_x.shape[0], size=1)
    img = test_x[index]
    y_true = np.argmax(test_y[index])
    print(y_true)
    p_ = network.predict(img)
    np.argmax(softmax(p_))
