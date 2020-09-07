import numpy as np

# 此类的目的是求斜率和截距 单属性
class Linear_model(object):
    def __init__(self):
        self.w = np.random.randn(1)[0]
        self.b = np.random.randn(1)[0]
        print('起始随机生成的斜率和截距---------', self.w, self.b)

    # model就是方程f(x) = xw + b
    def model(self, x):
        return self.w * x + self.b

    # 线性问题，原理都是最小二乘法
    def loss(self, x, y):
        # 此方程中2个未知数：w 和 b
        cost = (y - self.model(x)) ** 2
        # 求偏导数 导数是偏导数的一种特殊形式（只有一个未知数的时候）
        # 求偏导数，把其他的都当做已知数，求一个未知数的导数
        # 对w求偏导
        g_w = 2 * (y - self.model(x)) * (-x)
        # 对b求偏导
        g_b = 2 * (y - self.model(x)) * (-1)
        return g_w, g_b

    # 梯度下降
    def gradient_descent(self, g_w, g_b, step=0.01):
        # 更新新的斜率和截距
        # 现在的位置-梯度*步长
        self.w = self.w - g_w * step
        self.b = self.b - g_b * step
        print('更新后w和b---------', self.w, self.b)

    def fit(self, X, y, precision=0.001):
        w_last = self.w + 1
        b_last = self.b + 1
        # 精度
        # precision=0.001
        while True:

            if (np.abs(self.w - w_last) < precision) and (np.abs(self.b - b_last) < precision):
                break

            # 斜率的均值
            g_w = 0
            # 截距的均值
            g_b = 0
            size = X.shape[0]

            # 根据当前w，b求得偏导的均值
            for xi, yi in zip(X, y):
                # 求均值
                g_w += self.loss(xi, yi)[0] / size
                g_b += self.loss(xi, yi)[1] / size

            # 更新前给_last重新赋值
            w_last = self.w
            b_last = self.b

            # 将偏导的均值传给gradient_descent 执行梯度下降法更新
            self.gradient_descent(g_w, g_b)

    def coef_(self):
        return self.w

    def intercept(self):
        return self.b
