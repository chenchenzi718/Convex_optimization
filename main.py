"""
    主函数文件，主要实现了凸优化算法中的 primal-dual search direction并在具体的函数上实践
"""

import matplotlib.pyplot as plt
from TestFuncClass import *
from scipy.optimize import minimize
from MyConvexSolver import *

if __name__ == "__main__":
    # test_func_name 可以取 "test_1" 或者 “test_2” 或者 “test_3”
    test_func_name = "test_3"

    # 迭代初值
    x0 = np.array([0., 0.])
    if test_func_name == "test_1":
        x0 = np.array([0.1, 0.1])
    elif test_func_name == "test_2":
        x0 = np.array([0.1, 1.0])
    elif test_func_name == "test_3":
        x0 = np.array([0.9, 2.0])

    # 建立TestFunc类，根据输入的不同测试函数选择不同的约束条件
    test_func_class = TestFunc(test_func_str=test_func_name)
    cons, bounds = test_func_class.test_func_constraint()
    test_func = test_func_class.test_func_val

    # 使用内置的minimize求出“精确解”
    res = minimize(test_func, x0, method='SLSQP', bounds=bounds, constraints=cons)
    print(f"result of optimize.minimize function:{res.x}")
    print(f"val of optimize.minimize function:{res.fun}")

    solver = MyConvexSolver(test_func_class)
    x0, _lambda, _gamma = solver.primal_dual_convex_algorithm(x0)
    print(x0)
