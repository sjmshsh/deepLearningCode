# 数值微分求导
# f是一个函数, x是自变量
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / 2 * h
