import numpy as np


# 数值微分求导
# f是一个函数, x是自变量, 传入x是一个标量
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / 2 * h


# 数值微分求梯度向量, 传入x是一个向量, 底层函数
def _numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    # 遍历x中的特征, xi
    for i in range(x.size):
        tmp = x[i]
        x[i] = tmp + h
        fxh1 = f(x)  # 其他纬度都不变, x[i]加了一个微小量
        x[i] = tmp - h
        fxh2 = f(x)  # 其他纬度都不变, x[i]减了一个微小量, 满足求偏导数的概念
        # 利用中心差分公式计算偏导数
        grad[i] = (fxh1 - fxh2) / (2 * h)
        # 恢复x[i]的值
        x[i] = tmp
    return grad


# 传入X是一个矩阵
def numerical_gradient(f, X):
    # 判断维度
    if X.ndim == 1:
        return _numerical_gradient(f, X)
    else:
        grad = np.zeros_like(X)
        # 遍历X中的每一行数据，分别求梯度
        for i, x in enumerate(X):
            grad[i] = numerical_gradient(f, x)
        return grad
