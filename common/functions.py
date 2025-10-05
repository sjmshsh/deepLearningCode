import numpy as np

# 阶跃函数
def step_function0(x):
    if x > 0:
        return 1
    else:
        return 0

def step_function(x):
    return np.array(x > 0, dtype=int)

# sigmoid函数
def sigmoid(x):
    return 1/(1+np.exp(-x))

# ReLU函数
def relu(x):
    return np.maximum(0, x)

# softMax函数
def softmax(x):
    # 如果是二维矩阵
    if x.ndim == 2:
        y = np.exp(x) / np.sum(np.exp(x), axis=1)
        return y
    # 溢出处理策略
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
