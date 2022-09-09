# # solve函数可以给出全部解，
# from sympy import *
# import numpy as np
#
# x, y = symbols('x y')
# eqs = [Eq(x-0.195, 0), Eq(x**2+y**2-0.125*(x**2+y**2)**0.5-0.001591*(x**2+y**2)**0.5*atan2(y, x), 0)]
#
# print(nsolve(eqs, [x, y], (0.195, 0)))
import numpy as np
from scipy.optimize import root


def f1(x):
    return [x[0]-0.195, x[0]**2+x[1]**2-0.125*(x[0]**2+x[1]**2)**0.5-0.001591*(x[0]**2+x[1]**2)**0.5*np.arctan(x[1], x[0])]


print(root(f1, [0, -1]).x)  # 初始猜测值[0,-1]
print(root(f1, [0, 0]).x)  # 初始猜测值[0,0]

# from scipy.optimize import *
# import numpy as np
#
# def f(X):
#     x = X[0]
#     y = X[1]
#     return np.array([x-0.2,
#             x**2+y**2-0.125*(x**2+y**2)**0.5-0.001591*(x**2+y**2)**0.5*np.arctan(y/x)])
#
# guess = np.array([[0.2, 0]])
# s_fscolve = fsolve(f, guess)
# print(s_fscolve)


