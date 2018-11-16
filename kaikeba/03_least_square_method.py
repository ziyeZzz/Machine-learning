# -*- coding: utf-8 -*-
# 最小二乘法求解
# Q: y = a * x + b
import numpy as np
import matplotlib.pyplot as plt

def lsf(x, y):
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(x * y)
    sum_x2 = sum(x**2)
    n = len(x)
    a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    b = (sum_x2 * sum_y - sum_x * sum_xy) / (n * sum_x2 - sum_x**2)
    return a,b


if __name__ == '__main__':
    y = np.array((1.00, 0.90, 0.90, 0.81, 0.60, 0.56, 0.35))
    x = np.arange(3.6, 4.2, 0.1)
    a,b = lsf(x, y)
    print('y = {} * x + {}'.format(a, b))
    plt.plot(x, y, 'k*')
    x1 = np.arange(3.5, 4.3, 0.01)
    y1 = a * x1 + b
    plt.plot(x1, y1, 'r-')
    plt.show()
