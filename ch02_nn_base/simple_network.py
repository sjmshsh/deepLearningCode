import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identity(x):
    return x


# 初始化网络
def init_network():
    network = {}
    # 第一层参数
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])  # 2*3的矩阵
    network['b1'] = np.array([0.1, 0.2, 0.3])  # 向量
    # 第二层参数
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])  # 3*2的矩阵
    network['b2'] = np.array([0.1, 0.2])  # 向量
    # 第三层参数
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])  # 2*2的矩阵
    network['b3'] = np.array([0.1, 0.2])  # 向量
    return network


# 前向传播
def forward(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    # 逐层进行计算传递
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = identity(a3)
    return y


# 测试主流程
network = init_network()

# 定义数据
x = np.array([1.0, 0.5])

# 前向转播
y = forward(network, x)

print(y)
