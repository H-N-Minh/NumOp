""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the existing interface and return values of the task functions.
- Prior to your submission, check that the pdf showing your plots is generated.
"""
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.optimize as opt
from typing import Callable


# Modify the function bodies below to be used for function value and gradient computation
def func_1a(x: np.ndarray) -> float:
    """Computes and returns the function value for function 1d) at a given point x
    @param x Vector of size (2,)
    """

    """ Start of your code
    """
    f = 2.0 * (x[0] ** 3) - 6 * (x[1] ** 2) + 3 * (x[0] ** 2) * x[1]
    """ End of your code
    """
    return f


def grad_1a(x: np.ndarray) -> np.ndarray:
    """Computes and returns the analytical gradient result for function 1d) at a given point x
    @param x Vector of size (2,)
    """

    """ Start of your code
    """
    grad_x1 = 6 * (x[0] ** 2) + 6 * x[0] * x[1]
    grad_x2 = -12 * x[1] + 3 * (x[0] ** 2)
    grad = np.array([grad_x1, grad_x2])
    """ End of your code
    """
    return grad


def func_1b(x: np.ndarray) -> float:
    """Computes and returns the function value for function 1d) at a given point x
    @param x Vector of size (2,)
    """

    """ Start of your code
    """
    x_squared_l2_norm = x[0] ** 2 + x[1] ** 2
    f = (x[0] ** 2) + x[0] * x_squared_l2_norm + x_squared_l2_norm
    """ End of your code
    """
    return f


def grad_1b(x: np.ndarray) -> np.ndarray:
    """Computes and returns the analytical gradient result for function 1d) at a given point x
    @param x Vector of size (2,)
    """

    """ Start of your code
    """
    grad_x1 = 4 * x[0] + 3 * (x[0] ** 2) + x[1] ** 2
    grad_x2 = 2 * x[0] * x[1] + 2 * x[1]
    grad = np.array([grad_x1, grad_x2])
    """ End of your code
    """
    return grad


def func_1c(x: np.ndarray) -> float:
    """Computes and returns the function value for function 1d) at a given point x
    @param x Vector of size (2,)
    """

    """ Start of your code
    """
    f = np.log(1 + 0.5 * ((x[0] ** 2) + 3 * (x[1] ** 3)))
    """ End of your code
    """
    return f


def grad_1c(x: np.ndarray) -> np.ndarray:
    """Computes and returns the analytical gradient result for function 1d) at a given point x
    @param x Vector of size (2,)
    """

    """ Start of your code
    """
    denom = x[0] ** 2 + 3 * x[1] ** 3 + 2
    grad_x1 = (2 * x[0]) / denom
    grad_x2 = (9 * x[1] ** 2) / denom
    grad = np.array([grad_x1, grad_x2])
    """ End of your code
    """
    return grad


def func_1d(x: np.ndarray) -> float:
    """Computes and returns the function value for function 1d) at a given point x
    @param x Vector of size (2,)
    """

    """ Start of your code
    """
    f = (x[0] - 2) ** 2 + x[0] * (x[1] ** 2) - 2
    """ End of your code
    """
    return f


def grad_1d(x: np.ndarray) -> np.ndarray:
    """Computes and returns the analytical gradient result for function 1d) at a given point x
    @param x Vector of size (2,)
    """

    """ Start of your code
    """
    grad_x1 = 2 * x[0] + x[1] ** 2 - 4
    grad_x2 = 2 * x[0] * x[1]
    grad = np.array([grad_x1, grad_x2])
    """ End of your code
    """
    return grad


def task1():
    """Characterization of Functions

    Requirements for the plots:
        - ax[0] Contour plot for a)
        - ax[1] Contour plot for b)
        - ax[2] Contour plot for c)
    Choose the number of contour lines such that the stationary points and the function can be well characterized.
    """
    print("\nTask 1")

    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("Task 1 - Contour plots of functions", fontsize=16)

    ax[0].set_title("a)")
    ax[0].set_xlabel("$x_1$")
    ax[0].set_ylabel("$x_2$")

    ax[1].set_title("b)")
    ax[1].set_xlabel("$x_1$")
    ax[1].set_ylabel("$x_2$")

    ax[2].set_title("c)")
    ax[2].set_xlabel("$x_1$")
    ax[2].set_ylabel("$x_2$")

    ax[3].set_title("d)")
    ax[3].set_xlabel("$x_1$")
    ax[3].set_ylabel("$x_2$")


    """ Start of your code
    """
    function_list = [func_1a, func_1b, func_1c, func_1d]
    grid_range_list = [[[-5, 2], [-2, 5]],
                       [[-1.8, 1], [-1.5, 1.5]],
                       [[-1, 1], [-0.75, 1]],
                       [[-1, 3], [-2.5, 2.5]]]
    stationary_point_list = [[[[0, 0], 'saddle point'], [[-4, 4], 'saddle point']],
                             [[[0, 0], 'strict local minimum'], [[-4 / 3, 0], 'strict local maximum'],
                              [[-1, 1], 'saddle point'], [[-1, -1], 'saddle point']],
                             [[[0, 0], 'saddle point']],
                             [[[0, -2], 'saddle point'], [[0, 2], 'saddle point'], [[2, 0], 'strict local minimum']]]
    for (func, grid_range, i, stationary_points) in zip(function_list, grid_range_list, range(4),
                                                        stationary_point_list):
        x1, x2 = np.meshgrid(np.linspace(grid_range[0][0], grid_range[0][1], 100),
                             np.linspace(grid_range[1][0], grid_range[1][1], 100))
        cs = ax[i].contour(x1, x2, func(np.array([x1, x2])), levels=30)
        ax[i].clabel(cs, inline=1, fontsize=10)
        for (point, point_type) in stationary_points:
            ax[i].plot(point[0], point[1], 'o', label=point_type)
        ax[i].legend()
    """ End of your code
    """
    return fig


def task2():
    """Numerical Gradient Verification

    Implement the numerical gradient approximation using central differences in function approx_grad_task1. This function takes the function to be evaluated at point x as argument and returns the gradient approximation at that point.

    Pass the functions from task1 and compare the analytical and numerical gradient results for a given point x with np.allclose.

    Output the result of the comparison to the console.
    """
    print("\nTask 2")

    def approx_grad_task1(
        func: Callable, x: np.ndarray, eps: float, *args
    ) -> np.ndarray:
        """Numerical Gradient Computation
        @param x Vector of size (2,)
        @param eps float for numerical finite difference computation
        This function shall compute the gradient approximation for a given point 'x', 'eps' and a function 'func'
        using the given central differences formulation for 2D functions. (Task1 functions)
        @return The gradient approximation
        """

        """ Start of your code
        """
        grad_x1 = (func(np.array([x[0] + eps, x[1]])) - func(np.array([x[0] - eps, x[1]]))) / (2 * eps)
        grad_x2 = (func(np.array([x[0], x[1] + eps])) - func(np.array([x[0], x[1] - eps]))) / (2 * eps)
        grad = np.array([grad_x1, grad_x2])
        """ End of your code
        """

        return grad

    """ Start of your code
    """
    np.random.seed(1)
    x = np.random.rand(2, 3) * 10
    function_list = [func_1a, func_1b, func_1c, func_1d]
    gradient_function_list = [grad_1a, grad_1b, grad_1c, grad_1d]
    task_list = ['a', 'b', 'c', 'd']

    for (f_x, grad_x, task_id) in zip(function_list, gradient_function_list, task_list):
        print(f'---------- function 1{task_id} ----------')
        for i in range(3):
            x_i = x[:, i]
            print(f'Selected random vector x: {x_i}')
            approx_gradient = approx_grad_task1(func=f_x, x=x_i, eps=1e-5)
            analytic_gradient = grad_x(x_i)
            print(f'approximated gradient 1{task_id}: {approx_gradient}')
            print(f'analytic     gradient 1{task_id}: {analytic_gradient}')
            if np.allclose(approx_gradient, analytic_gradient):
                print(f'The numeric gradient approximates the analytic gradient for function 1{task_id}')
            else:
                print(f'The numeric gradient does not approximate the analytic gradient for function 1{task_id}')
    """ End of your code
    """


# Modify the function bodies below to be used for function value and gradient computation
def func_3a(x: np.ndarray, A: np.ndarray, B: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes and returns the function value for function 3a) at a given point x
    @param x Vector of size (n,)
    @param A Matrix of size (m,o)
    @param B Matrix of size (o,n)
    @param b Vector of size (m,)
    """

    """ Start of your code
    """
    x = np.reshape(x, (len(x),))
    b = np.reshape(b, (len(b),))
    f_x = np.array([0.5 * np.linalg.norm((A @ B) @ x - b)**2])
    """ End of your code
    """
    return f_x


def grad_3a(x: np.ndarray, A: np.ndarray, B: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes and returns the gradient value for function 3a) at a given point x
    @param x Vector of size (n,)
    @param A Matrix of size (m,o)
    @param B Matrix of size (o,n)
    @param b Vector of size (m,)
    """

    """ Start of your code
    """
    x = np.reshape(x, (len(x),))
    b = np.reshape(b, (len(b),))
    grad_x = np.transpose(A @ B) @ (A @ B @ x - b)
    """ End of your code
    """
    return grad_x


def hessian_3a(x: np.ndarray, A: np.ndarray, B: np.ndarray, b: np.ndarray):
    """Computes and returns the Hessian value for function 3a) at a given point x
    @param x Vector of size (n,)
    @param A Matrix of size (m,o)
    @param B Matrix of size (o,n)
    @param b Vector of size (m,)
    """

    """ Start of your code
    """
    hessian_x = np.transpose(A @ B) @ A @ B
    """ End of your code
    """
    return hessian_x


def func_3b(x: np.ndarray, K: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Computes and returns the function value for function 3b) at a given point x
    @param x Vector of size (n,)
    @param K Matrix of size (n, n)
    @param t Vector of size (n,)
    """

    """ Start of your code
    """
    x = np.reshape(x, (len(x),))
    t = np.reshape(t, (len(t),))
    f_x = np.array([np.sum(x) - 0.5 * np.transpose(x * t) @ K @ (x * t)])
    """ End of your code
    """
    return f_x


def grad_3b(x: np.ndarray, K: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Computes and returns the gradient value for function 3b) at a given point x
    @param x Vector of size (n,)
    @param K Matrix of size (n, n)
    @param t Vector of size (n,)
    """

    """ Start of your code
    """
    x = np.reshape(x, (len(x),))
    t = np.reshape(t, (len(t),))
    grad_x = 1 - K @ (x * t) * t
    """ End of your code
    """
    return grad_x


def hessian_3b(x: np.ndarray, K: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Computes and returns the Hessian value for function 3b) at a given point x
    @param x Vector of size (n,)
    @param K Matrix of size (n, n)
    @param t Vector of size (n,)
    """

    """ Start of your code
    """
    t = np.reshape(t, (len(t),))
    hessian_x = - np.diag(t) @ K.T @ np.diag(t)
    """ End of your code
    """
    return hessian_x


def func_3c(
    alpha: np.ndarray, A: np.ndarray, x: np.ndarray, y: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Computes and returns the function value for function 3c) at a given point x
    @param a scalar of size (1,)
    @param A Matrix of size (m,n)
    @param x Vector of size (n,)
    @param y Matrix of size (n,)
    @param b Vector of size (m,)
    """

    """ Start of your code
    """
    alpha = np.reshape(alpha, (len(alpha),))
    x = np.reshape(x, (len(x),))
    y = np.reshape(y, (len(y),))
    b = np.reshape(b, (len(b),))
    f_x = np.array([0.5 * np.linalg.norm(A @ (x + alpha * y) - b)**2])
    """ End of your code
    """
    return f_x


def grad_3c(
    alpha: np.ndarray, A: np.ndarray, x: np.ndarray, y: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Computes and returns the gradient value for function 3c) at a given point x
    @param a scalar of size (1,)
    @param A Matrix of size (m,n)
    @param x Vector of size (n,)
    @param y Matrix of size (n,)
    @param b Vector of size (m,)
    """

    """ Start of your code
    """
    alpha = np.reshape(alpha, (len(alpha),))
    x = np.reshape(x, (len(x),))
    y = np.reshape(y, (len(y),))
    b = np.reshape(b, (len(b),))
    grad_alpha = np.array([np.transpose(A @ (x + alpha * y) - b) @ A @ y])
    """ End of your code
    """
    return grad_alpha


def hessian_3c(
    alpha: np.ndarray, A: np.ndarray, x: np.ndarray, y: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Computes and returns the Hessian value for function 3c) at a given point x
    @param a scalar of size (1,)
    @param A Matrix of size (m,n)
    @param x Vector of size (n,)
    @param y Matrix of size (n,)
    @param b Vector of size (m,)
    """

    """ Start of your code
    """
    y = np.reshape(y, (len(y),))
    hessian_alpha = np.array([np.transpose(y) @ np.transpose(A) @ A @ y])
    """ End of your code
    """
    return hessian_alpha


def task3():
    """Matrix Calculus: Numerical Gradient Verification

    Utilize the function scipy.optimize.approx_fprime to numerically check the correctness of your analytic results. To this end, implement the functions func_3a, grad_3a, hessian_3a, func_3b, grad_3b, hessian_3b, func_3c, grad_3c, hessian_3c and compare them to the approximations.

    Check the correctness of your results by comparing the analytical and numerical results for three random points x with np.allclose. Also stick to the provided values for the present variables.

    Output the result of the comparison to the console.
    """
    print("\nTask 3")

    A = np.array([[0, 1], [2, 3]])  # do not change
    B = np.array([[3, 2], [1, 0]])  # do not change
    K = np.array([[1, 2], [2, 1]])  # do not change
    b = np.array([[4], [0.5]])  # do not change
    y = np.array([[1], [1]])  # do not change
    x = np.array([[0.5], [0.75]])  # do not change
    t = np.array([[7.5], [-3]])  # do not change

    """ Start of your code
    """
    task_list = ['a', 'a', 'b', 'b', 'c', 'c']
    print_text_list = ['gradient', 'hessian', 'gradient', 'hessian', 'gradient', 'hessian']
    function_list = [[func_3a, grad_3a], [grad_3a, hessian_3a],
                     [func_3b, grad_3b], [grad_3b, hessian_3b],
                     [func_3c, grad_3c], [grad_3c, hessian_3c]]
    for (fun, task_id, print_text) in zip(function_list, task_list, print_text_list):
        print(f'------------ function 3{task_id} ------------')
        if task_id == 'a':
            x_rand = np.random.rand(2,) * 10
            print(f'Selected random vector x: {x_rand}')
            true_function_return = fun[1](x_rand, A, B, b)
            approx_function_return = opt.approx_fprime(x_rand, fun[0], 1.49e-08, A, B, b)
        elif task_id == 'b':
            x_rand = np.random.rand(2,) * 10
            print(f'Selected random vector x: {x_rand}')
            true_function_return = fun[1](x_rand, K, t)
            approx_function_return = opt.approx_fprime(x_rand, fun[0], 1.49e-08, K, t)
        else:
            alpha = np.random.rand(1)
            print(f'Selected random scalar alpha: {alpha}')
            true_function_return = fun[1](alpha, A, x, y, b)
            approx_function_return = opt.approx_fprime(alpha, fun[0], 1.49e-08,  A, x, y, b)
        print(f'approximated {print_text} 1{task_id}: {approx_function_return}')
        print(f'analytic     {print_text} 1{task_id}: {true_function_return}')
        if np.allclose(true_function_return, approx_function_return):
            print(f'The numeric {print_text} approximates the analytic {print_text} for function 1{task_id}')
        else:
            print(f'The numeric {print_text} does not approximate the analytic {print_text} for function 1{task_id}')
    """ End of your code
    """


def task4():
    """Linear Program

    Implement LP for student task selection and solve using scipy.optimize.linprog's solver.

    """
    print("\nTask 4")

    """ Start of your code
    """
    I = 2  # number of students
    J = 15  # number of tasks
    task_timing_student_1 = (np.array([0.5, 0.25, 0.25, 0.25, 1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 1.5, 2.5, 1.0, 2.5, 3.5])
                             .reshape((J, 1)))
    task_timing_student_2 = (np.array([0.75, 1.0, 0.75, 0.5, 0.5, 0.25, 0.25, 0.25, 2.0, 1.25, 1.0, 4.0, 2.5, 3.0, 2.0])
                             .reshape((J, 1)))
    time_budget_student_1 = 9
    time_budget_student_2 = 6

    task_hours_student_1 = np.sum(task_timing_student_1)
    task_hours_student_2 = np.sum(task_timing_student_2)
    print('------- Hours each student would take to solve the entire assignment on its own -------\n'
          f'Student 1: {task_hours_student_1}h\n'
          f'Student 2: {task_hours_student_2}h')

    c = np.vstack([task_timing_student_1, task_timing_student_2])

    A_ub = np.vstack([np.hstack([np.tile(task_timing_student_1.transpose(), (J, 1)), np.zeros((J, J))]),
                      np.hstack([np.zeros((J, J)), np.tile(task_timing_student_2.transpose(), (J, 1))]),
                      np.hstack([np.identity(J), np.identity(J)]),
                      np.hstack([-np.identity(J), -np.identity(J)])])

    b_ub = np.vstack([np.full((J, 1), time_budget_student_1),
                      np.full((J, 1), time_budget_student_2),
                      np.full((J, 1), 1),
                      np.full((J, 1), -1)])

    res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub)

    print('------- Results of linear optimization -------')
    student_1_time = c.transpose() @ (np.vstack([np.ones((J, 1)), np.zeros((J, 1))]) * res.x.reshape((I * J, 1)))
    student_2_time = c.transpose() @ (np.vstack([np.zeros((J, 1)), np.ones((J, 1))]) * res.x.reshape((I * J, 1)))

    print(f'Task assignment x: {res.x}\n'
          f'Time of Student 1: {student_1_time.flatten()[0]}h\n'
          f'Time of Student 2: {student_2_time.flatten()[0]}h\n'
          f'Total time:        {res.fun}h')
    """ End of your code
    """


if __name__ == "__main__":
    pdf = PdfPages("figures.pdf")

    # tasks = [task1, task2, task3, task4]
    tasks = [task2]
    for t in tasks:
        fig = t()

        if fig is not None:
            pdf.savefig(fig)

    pdf.close()
