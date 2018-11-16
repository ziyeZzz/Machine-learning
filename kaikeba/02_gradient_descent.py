# -*- coding: utf-8 -*-
# 梯度下降和随机梯度下降法求解
# Q:argmin1/2*[(x1+x2-4)^2 + (2x1+3x2-7)^2 + (4x1+x2-9)^2]

import random

# 要最小化的函数
def f(x1, x2):
    return 1/2 * ((x1 + x2 - 4) ** 2 + (2 * x1 + 3 * x2 - 7) ** 2 + (4 * x1 + x2 - 9) ** 2)
# x1的偏导
def f_x1(x1, x2):
    return (x1+x2-4)*1 + (2*x1+3*x2-7)*2 + (4*x1+x2-9)*4
# x2的偏导
def f_x2(x1, x2):
    return (x1+x2-4)*1 + (2*x1+3*x2-7)*3 + (4*x1+x2-9)*1

# 梯度下降
def gradient_descent(x1, x2, a):
    f_best = f(x1, x2) # 当前最优
    x1_update = x1 - a * f_x1(x1, x2)
    x2_update = x2 - a * f_x2(x1, x2)
    f_update = f(x1_update, x2_update)
    if f_best - f_update > 0.01:
        return(gradient_descent(x1_update, x2_update, a))
    else:
        return x1, x2, f_best

# 随机取一个数据求x1,x2偏导, 并更新x1, x2 => 相当于以偏概全
def f_random(x1, x2, a):
    dice = random.randint(1,3)
    if dice == 1:
        x1_update = x1 - a * ((x1+x2-4)*1)
        x2_update = x2 - a * ((x1+x2-4)*1)
    elif dice == 2:
        x1_update = x1 - a * ((2*x1+3*x2-7)*2)
        x2_update = x2 - a * ((2*x1+3*x2-7)*3)
    else:
        x1_update = x1 - a * ((4*x1+x2-9)*4)
        x2_update = x2 - a * ((4*x1+x2-9)*1)
    return x1_update, x2_update

# 随机梯度下降
def random_gradient_descent(x1, x2, a):
    f_best =f(x1, x2)
    x1_update, x2_update = f_random(x1, x2, a)
    f_update = f(x1_update, x2_update)
    if f_best - f_update > 0.01:
        return(random_gradient_descent(x1_update, x2_update, a))
    else:
        return x1, x2, f_best

if __name__ == '__main__':
    # a is the step length. x1_0 and x2_0 are initial value of x1 and x2.
    a = 0.01
    x1_0, x2_0 = 0, 0
    # 经过梯度下降后，最好的x1, x2 和 f
    x1, x2, f_best = gradient_descent(x1_0, x2_0, a)
    print('x1:{}, x2:{}, minimal_f:{}'.format(x1, x2, f_best))
    # 经随机梯度下降后，算出的最好的x1, x2 和 f
    a = 0.05
    x1, x2, f_best = random_gradient_descent(x1_0, x2_0, a)
    print('x1:{}, x2:{}, minimal_f:{}'.format(x1, x2, f_best))