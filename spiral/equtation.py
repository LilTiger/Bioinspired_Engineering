# # solve函数可以给出全部解，
# from sympy import *
# import numpy as np
#
# x, y = symbols('x y')
# eqs = [Eq(1.732*x - 0.20784 - y, 0), Eq(sqrt(x+y)-0.1-0.03183*(atan2(y, x)), 0)]
#
# print(nsolve(eqs, [x, y]))

# # 利用gekko求解非线性方程组
# from gekko import GEKKO
# import numpy as np
#
# m = GEKKO()
# x = m.Var(value=0.12)  # 给定初值为0
# y = m.Var(value=0)  # 给定初值为0
#
# m.Equations([1.732*x - 0.20784 - y == 0,
#              ((x+y)**0.5) - 0.1 - 0.03183 * (GEKKO.atan(y, x)) == 0])
# m.solve(disp=False)
# x, y = x.value, y.value
# print(x, y)


from scipy.optimize import *
import numpy as np

def f(X):
    x = X[0]
    y = X[1]
    return np.array([1.732*x - 0.20784 - y,
            ((x + y)**0.5) - 0.1 - 0.03183 * (np.arctan(y/x))])

guess = np.array([[0.1, 1]])
s_fscolve = fsolve(f, guess)
print(s_fscolve)


