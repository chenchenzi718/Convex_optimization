import numpy as np

from JacobianEval import *


class MyConvexSolver:
    def __init__(self, test_func: TestFunc, epi=1e-5):
        self.func = test_func.test_func_val
        self.cons, self.bounds = test_func.test_func_constraint()
        # 等式约束Ax=b
        self.A = test_func.A
        self.b = test_func.b
        self.epi = epi

        # 只保留cons中的不等式约束
        self.cons = [cons for cons in self.cons if cons['type']=='ineq']

        # 将bounds转化为cons存储为列表
        if self.bounds is not None:
            self.bounds_to_cons = []
            self.__change_bounds_to_cons()
            self.cons_with_bounds = self.cons + self.bounds_to_cons
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

    # 计算不等式约束 f(x)<=0 ，其中 f(x)=(f1(x),...,fm(x))T 均为满足 fi(x)<=0 的约束, 返回 m*1 矩阵，m为不等式约束数目
    def ineq_cons_val(self, x):
        val = []
        for cons in self.cons_with_bounds:
            # 凸优化中要求不等式约束为小于等于,cons中存储的是大于等于（因为要方便scipy 中的minimize计算）
            val.append(-cons['fun'](x))
        return np.array(val)

    # 计算不等式约束 f(x) 的 jacobi 矩阵, 返回一个 m*n 矩阵，m为cons数，n为x维数
    def ineq_cons_jac(self, x):
        val = []
        for jac in self.cons_func_jac:
            val.append(self.n_jacobi(jac, x)[0])
        return np.array(val)

    # 计算 dual residual delta(f0(x))+J(f(x)) @ lambda + AT @ gamma ，返回一个 n*1 矩阵
    def dual_residual(self, x, _lambda, _gamma):
        res = self.n_jacobi(self.test_func_jac, x).T + self.ineq_cons_jac(x).T @ _lambda.reshape(-1, 1) + \
            self.A.T @ _gamma.reshape(-1, 1)
        return res

    # 计算 centrality residual -diag(\lambda)f(x)-1/t I， 返回一个 m*1 矩阵 ，m为不等式约束数目
    def central_residual(self, x, _lambda, t):
        res = -np.diag(_lambda) @ self.ineq_cons_val(x) - 1.0/t * np.eye(len(_lambda))
        return res

    # 计算 primal residue Ax-b
    def primal_residual(self, x):
        return self.A @ x.reshape(-1, 1) - self.b.reshape(-1, 1)

    # 返回 residual 矩阵的二阶模
    def total_res(self, x, _lambda, _gamma, t):
        dual_res = self.dual_residual(x, _lambda, _gamma)
        central_res = self.central_residual(x, _lambda, t)
        primal_res = self.primal_residual(x)
        total_res = np.concatenate((dual_res, central_res, primal_res), axis=0)
        return total_res

    # 给出surrogate duality gap
    def surrogate_duality_gap(self, x, _lambda):
        return -np.dot(self.ineq_cons_val(x), _lambda)

    # 完成 primal_dual 方法中的牛顿迭代法
    def newton_iteration(self, x, _lambda, _gamma, t):
        n = 2
        m = len(self.cons_with_bounds)
        p = len(self.A)
        total_dim = n + m + p
        jac_total_res = np.zeros((total_dim, total_dim))

        jac_11 = self.n_hessian(self.test_func_hes, x)
        for i in range(m):
            jac_11 += _lambda[i] * self.n_hessian(self.cons_func_hes[i], x)

        jac_12 = self.ineq_cons_jac(x).T
        jac_13 = self.A.T
        jac_21 = -np.diag(_lambda) @ self.ineq_cons_jac(x)
        jac_22 = -np.diag(self.ineq_cons_val(x))
        jac_31 = self.A

        jac_total_res[:n, :n] = jac_11
        jac_total_res[:n, n:(n+m)] = jac_12
        jac_total_res[:n, (n+m):] = jac_13
        jac_total_res[n:(n+m), :n] = jac_21
        jac_total_res[n:(n+m), n:(n+m)] = jac_22
        jac_total_res[(n+m):, :n] = jac_31

        # 接下来只需求解方程组 jac_total_res @ delta_(x,lambda,gamma) = - total_res
        return np.linalg.solve(jac_total_res, -self.total_res(x, _lambda, _gamma, t))

    # 完成 primal-dual 内点算法
    def primal_dual_convex_algorithm(self, x0):
        m = len(self.cons_with_bounds)

