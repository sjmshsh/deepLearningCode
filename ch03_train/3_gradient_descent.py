import numpy as np
import matplotlib.pyplot as plt

from common.gradient import numerical_gradient


# 定义梯度下降法的函数
def gradient_descent(f, init_x, lr=0.01, num_iter=1000):
    x = init_x
    # 定义列表保存x的变化
    x_history = []
    # 循环迭代的过程
    for i in range(num_iter):
        x_history.append(x.copy())
        # 计算梯度
        grad = numerical_gradient(f, x)
        # 更新参数
        x -= lr * grad
    return x, np.array(x_history)


# 定义目标函数 f(x1, x2) = x1^2 + x2^2
def f(x):
    return x[0] ** 2 + x[1] ** 2


if __name__ == '__main__':
    # 1. 定义初始值
    init_x = np.array([-3.0, 4.0])
    # 2. 定义超参数
    lr = 0.1
    num_iter = 20
    # 3. 使用梯度下降法, 计算最小值点
    x, x_history = gradient_descent(f, init_x, lr, num_iter)
    print('最小值点为: ', x)

    # 画图
    plt.plot([-5, 5], [0, 0], '--')
    plt.plot([0, 0], [-5, 5], '--b')
    plt.scatter(x_history[:, 0], x_history[:, 1])
    plt.xlim([-3.5, 3.5])
    plt.ylim([-4.5, 4.5])
    plt.xlabel('X[0]')
    plt.ylabel('X[1]')
    plt.show()
