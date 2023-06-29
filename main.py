"""
    主函数文件，主要实现了凸优化算法中的 primal-dual search direction并在具体的函数上实践
"""

from PyQt5.QtWidgets import QGraphicsScene
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.optimize import minimize
from PyQt5.QtWidgets import QApplication, QMainWindow

from functools import partial
import sys
from ui_convex import *

from MyConvexSolver import *


# 画出最终的结果图
def pic_my_convex(func_class: TestFunc, actual_x, my_convex_medium_x, test_func_name="test_1"):
    func = func_class.test_func_val
    dx = 0.01
    dy = 0.01

    if test_func_name == "test_1":
        x = np.arange(-2., 2., dx)
        y = np.arange(-2., 2., dy)
        X, Y = np.meshgrid(x, y)
    elif test_func_name == "test_2":
        x = np.arange(-0.5, 0.5, dx)
        y = np.arange(0.5, 1.5, dy)
        X, Y = np.meshgrid(x, y)
    else:
        x = np.arange(-0.5, 1.5, dx)
        y = np.arange(0, 2.5, dy)
        X, Y = np.meshgrid(x, y)

    y1, y2 = sy.symbols("y1, y2")
    func_sym = func([y1, y2])
    func_val = sy.lambdify([y1, y2], func_sym, 'numpy')

    contour = plt.contour(X, Y, func_val(X, Y), 20)  # 生成等值线图
    plt.contourf(X, Y, func_val(X, Y), 20)
    # plt.clabel(contour, inline=1, fontsize=10)
    plt.colorbar()

    # 一个初值进行绘制
    # 在图上画出最终收敛点
    plt.scatter(actual_x[0], actual_x[1], marker="*", c="y", s=100)

    # 在图上画出在内点算法迭代过程中的点
    plt_x = []
    plt_y = []
    for _array in my_convex_medium_x:
        plt_x.append(_array[0])
        plt_y.append(_array[1])
    plt.plot(plt_x, plt_y, "w--")
    plt.title("Primary value x0 iteration in method " + str(test_func_name))

    return plt


# 定义一个UI
class MyConvexUI(QMainWindow, Ui_MyResult):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(partial(self.test_func, "test_1"))
        self.pushButton_2.clicked.connect(partial(self.test_func, "test_2"))
        self.pushButton_3.clicked.connect(partial(self.test_func, "test_3"))

    def test_func(self, test_func_str):
        if test_func_str == "test_1":
            x0 = np.array([0.1, 0.1])
        elif test_func_str == "test_2":
            x0 = np.array([0.1, 1.0])
        elif test_func_str == "test_3":
            x0 = np.array([0.9, 2.0])
        else:
            x0 = np.array([0., 0.])

        # 建立TestFunc类，根据输入的不同测试函数选择不同的约束条件
        test_func_class = TestFunc(test_func_str=test_func_str)
        cons, bounds = test_func_class.test_func_constraint()
        test_func = test_func_class.test_func_val

        # 使用内置的minimize求出“精确解”
        res = minimize(test_func, x0, method='SLSQP', bounds=bounds, constraints=cons)
        print(f"result of optimize.minimize function:{res.x}")
        print(f"val of optimize.minimize function:{res.fun}")

        solver = MyConvexSolver(test_func_class)
        x1, _lambda, _gamma = solver.primal_dual_convex_algorithm(x0)
        print(f"result of my convex solver:{x1}")
        print(f"val of my convex solver function:{test_func(x1)}")

        plt = pic_my_convex(test_func_class, res.x, solver.myconvex_intermedium_result, test_func_str)
        # 获取当前的Figure对象
        canvas = plt.gcf().canvas

        # 将pyplot图形转换为Figure对象
        figure_object = canvas.figure

        # 关闭pyplot图形
        plt.close()

        self.scene = QGraphicsScene()  # 创建一个场景
        self.canvas = FigureCanvas(figure_object)
        self.scene.addWidget(self.canvas)
        self.graphicsView.setScene(self.scene)  # 将创建添加到图形视图显示窗口


if __name__ == "__main__":

    use_ui = True

    # 如果要使用ui，如果出现没安装pyqt跑不起来，可以修改 use_ui 为False进行测试
    if use_ui:
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = MyConvexUI()  # 创建PyQt设计的窗体对象
        MainWindow.show()  # 显示窗体
        sys.exit(app.exec_())  # 程序关闭时退出进程
    else:
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

        # 使用我编写的primal-dual 内点法计算结果
        solver = MyConvexSolver(test_func_class)
        x1, _lambda, _gamma = solver.primal_dual_convex_algorithm(x0)
        print(f"result of my convex solver:{x1}")
        print(f"val of my convex solver function:{test_func(x1)}")
        print(f"the iteration num:{len(solver.myconvex_intermedium_result)-1}")

        # 仅画出在x0条件下上述迭代的图示
        plt = pic_my_convex(test_func_class, res.x, solver.myconvex_intermedium_result, test_func_name=test_func_name)
        plt.show()
