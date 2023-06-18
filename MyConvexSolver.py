from JacobianEval import *


class MyConvexSolver:
    def __init__(self, test_func: TestFunc, epi=1e-5):
        self.func = test_func.test_func_val
        self.cons, self.bounds = test_func.test_func_constraint()
        # 等式约束Ax=b
        self.A = test_func.A
        self.b = test_func.b
        self.epi = epi

        # 将bounds转化为cons存储为列表
        if self.bounds is not None:
            self.bounds_to_cons = []
            self.__change_bounds_to_cons()
            self.cons_with_bounds = list(self.cons) + self.bounds_to_cons
        else:
            self.cons_with_bounds = self.cons

        self.test_func_jac = None
        self.test_func_hes = None
        self.cons_func_jac = []
        self.cons_func_hes = []
        self.__build_sy_jacobi_hessian()

        # 这个变量用来记录sqp的x_k中间结果
        self.myconvex_intermedium_result = []

    # 给出需要计算的函数以及 不等式约束函数 的jacobi的符号计算结果
    def __build_sy_jacobi_hessian(self):
        y1, y2 = sy.symbols("y1, y2")
        test_func = self.func([y1, y2])

        # 封装成sympy的符号矩阵
        funcs = sy.Matrix([test_func])
        args = sy.Matrix([y1, y2])

        # 实例化类并计算雅克比矩阵
        h = JacobianEval(funcs, args)
        self.test_func_jac = h.jacobi()
        self.test_func_hes = h.hessian()

        for cons in self.cons_with_bounds:
            if cons['type'] == 'ineq':
                # 凸优化中要求不等式约束为小于等于,cons中存储的是大于等于（因为要方便scipy 中的minimize计算）
                funcs = -sy.Matrix([cons['fun']([y1, y2])])
                h = JacobianEval(funcs, args)
                self.cons_func_jac.append(h.jacobi())
                self.cons_func_hes.append(h.hessian())

    # 想要把两个变量的区间信息直接作为一个函数不等式约束
    def __change_bounds_to_cons(self):
        x0_range = self.bounds[0]
        x1_range = self.bounds[1]
        if x0_range[0] is not None:
            self.bounds_to_cons.append({'type': 'ineq', 'fun': lambda x: x[0] - x0_range[0]})
        if x0_range[1] is not None:
            self.bounds_to_cons.append({'type': 'ineq', 'fun': lambda x: -x[0] + x0_range[1]})
        if x1_range[0] is not None:
            self.bounds_to_cons.append({'type': 'ineq', 'fun': lambda x: x[1] - x1_range[0]})
        if x1_range[1] is not None:
            self.bounds_to_cons.append({'type': 'ineq', 'fun': lambda x: -x[1] + x1_range[1]})

    # 计算jacobi函数，sym_jacobi为符号函数, 返回的为一个行向量
    def n_jacobi(self, sym_jacobi, x):
        y1, y2 = sy.symbols("y1, y2")
        f_jac = sy.lambdify([y1, y2], sym_jacobi, 'numpy')  # 通过这段话转化为可以计算的函数表达式
        return f_jac(x[0], x[1])

    # 计算hessian函数，sym_hessian为符号函数
    def n_hessian(self, sym_hessian, x):
        y1, y2 = sy.symbols("y1, y2")
        f_his = sy.lambdify([y1, y2], sym_hessian, 'numpy')
        return f_his(x[0], x[1])

    # 计算 dual residual delta(f0(x))+J(f(x)) @ lambda + AT @ gamma
    def dual_residual(self, x, _lambda, _gamma):
        n_jac_f0 = self.n_jacobi(self.test_func_jac, x).T
        for i in range(len(self.cons_func_jac)):
            n_jac_fi = self.n_jacobi(self.cons_func_jac[i], x)
            n_jac_f0 += _lambda[i] * n_jac_fi.T
        n_jac_f0 += _gamma
