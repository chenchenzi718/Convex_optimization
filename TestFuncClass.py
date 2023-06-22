import numpy as np
import sympy as sy

"""
    本文件完成了类TestFunc的编写，根据输入的test_func_str的不同决定不同的测试函数
    test_func_val函数计算对应的值
    同时也在本类中给出了对应于测试函数的约束条件设置test_func_constraint
    与 SQP 时不同，在这里我们直接给出等式约束的Ax=b的 A，b
"""


class TestFunc:
    def __init__(self, test_func_str="test_1"):
        self.test_func_str = test_func_str

        self.cons = None
        self.bounds = None

        # 凸优化要求等式约束，这里直接给出A,b
        self.A = None
        self.b = None

    def test_func_val(self, input_val):
        # define a function named Rosenbrock
        if self.test_func_str == "test_1":    # 返回一个f(x,y)=0.5x^2+y^2-xy-2x-6y
            return 0.5 * input_val[0]**2.0 + input_val[1]**2.0 - input_val[0] * input_val[1] - 2.0 * input_val[0] - \
                6.0 * input_val[1]
        elif self.test_func_str == "test_2":  # 返回一个函数f(x,y)=x^2+9y^2
            return input_val[0]**2.0 + 9.0 * input_val[1]**2.0
        elif self.test_func_str == "test_3":  # 返回一个函数 f(x,y) = 3x^2+y^4+exp(x+y)-2x-3y
            return 3.0 * input_val[0]**2 + input_val[1]**4 + sy.exp(input_val[0] + input_val[1]) - 2.0 * input_val[0] \
                - 3.0 * input_val[1]

    # 不等式约束均为大于等于的约束
    def test_func_constraint(self):
        if self.test_func_str == "test_1":
            cons_ineq_1 = {'type': 'eq', 'fun': lambda x: -x[0] - x[1] + 2}
            cons_ineq_2 = {'type': 'ineq', 'fun': lambda x: x[0] - 2.0 * x[1] + 2}
            cons_ineq_3 = {'type': 'ineq', 'fun': lambda x: -2.0 * x[0] - x[1] + 3}
            cons_ineq_4 = {'type': 'ineq', 'fun': lambda x: x[0]}
            cons_ineq_5 = {'type': 'ineq', 'fun': lambda x: x[1]}
            cons = (cons_ineq_1, cons_ineq_2, cons_ineq_3, cons_ineq_4, cons_ineq_5)
            self.cons = cons
            self.A = np.array([[1., 1.]])
            self.b = np.array(2.)
            return cons, None
        elif self.test_func_str == "test_2":
            cons_eq = {'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1}
            cons_ineq_1 = {'type': 'ineq', 'fun': lambda x: x[0] + 3.0 * x[1] - 3}
            cons_ineq_2 = {'type': 'ineq', 'fun': lambda x: -x[0] + x[1]}
            cons_ineq_3 = {'type': 'ineq', 'fun': lambda x: x[0]}
            cons_ineq_4 = {'type': 'ineq', 'fun': lambda x: x[1]}
            cons = (cons_eq, cons_ineq_1, cons_ineq_2, cons_ineq_3, cons_ineq_4)
            self.cons = cons
            self.A = np.array([[1.0, 1.0]])
            self.b = np.array(1.0)
            return cons, None
        elif self.test_func_str == "test_3":
            cons_eq = {'type': 'eq', 'fun': lambda x: 2.0 * x[0] + x[1] - 3}
            cons_ineq_1 = {'type': 'ineq', 'fun': lambda x: -x[0]**2 - x[1]**2 + 5}
            cons_ineq_2 = {'type': 'ineq', 'fun': lambda x: x[0] + 2 * x[1] - 4}
            cons = (cons_eq, cons_ineq_1, cons_ineq_2)
            self.cons = cons
            self.A = np.array([[2.0, 1.0]])
            self.b = np.array(3.0)
            return cons, None
