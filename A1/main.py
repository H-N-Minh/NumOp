#!/usr/bin/env python
"""Python code submission file.
IMPORTANT:
- Do not include any additional python packages.
- Do not change the existing interface and return values of the task functions.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import approx_fprime
from typing import Callable

# Modify the following global variables to be used in your functions
""" Start of your code
"""
alpha = 69
beta = 420
d = 2.5
b = np.array([10, -20, 30, -40, 50])
D = np.array([
    [  45,   -2,  350,  -40,   50],
    [ -60,   70,   -7,   35,   45],
    [  11, -120, -130,   -4,    0],
    [  -8,  -45,  -87,  190,  206],
    [  10,   20,  451,  -10,  -50]
])
A = np.array([
    [ 157, -146,  134,  160, -144],
    [ 133,  111, -207,  -37,  245],
    [-247,    0, -109,  -89,   99],
    [ -65,  -12,  -67,  134,  -11],
    [ -85,   72,  -82,  211,  -66]
])
""" End of your code
"""


def task1():
    """Characterization of Functions
    Requirements for the plots:
        - ax[0, 0] Contour plot for a)
        - ax[0, 1] Contour plot for b)
        - ax[1, 0] Contour plot for c)
        - ax[1, 1] Contour plot for d)
    """
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle("Task 1 - Contour plots of functions", fontsize=16)
    ax[0, 0].set_title("a)")
    ax[0, 0].set_xlabel("$x_1$")
    ax[0, 0].set_ylabel("$x_2$")

    ax[0, 1].set_title("b)")
    ax[0, 1].set_xlabel("$x_1$")
    ax[0, 1].set_ylabel("$x_2$")

    ax[1, 0].set_title("c)")
    ax[1, 0].set_xlabel("$x_1$")
    ax[1, 0].set_ylabel("$x_2$")

    ax[1, 1].set_title("d)")
    ax[1, 1].set_xlabel("$x_1$")
    ax[1, 1].set_ylabel("$x_2$")

    """ Start of your code
    """
    # controller (manually adjust inputs here)
    grid_range_min = -5     # min/max value of grid to plot
    grid_range_max = 5
    
    x1_a = np.linspace(-5, 5, 100)      # for a, stationary points is a straight line, so we create a linspace
    x2_a = (x1_a + 2.5) / 3             # func of the straight line
    stationary_points = [[(x1_a, x2_a, "Global Minimum: x1 = 3x2 - 2.5")],                                      # a) [(1. stationary point with coord and type), (2. stationary point with coord and type), ...]
                         
                         [(0, 2, "Saddle Point"), (0, -2, "Saddle Point"), (2, 0, "Strict Local Minimum")],     # b)

                         [(0, 0, "Strict Local Minimum"), (-4/3, 0, "Strict Local Maximum"),                    # c)
                          (-1, 1, "Saddle Point"), (-1, -1, "Saddle Point")],

                         [(1/69, 0, "Strict Global Minimum")]]                                                                      # d)

    # Loop through each function to generate contour and plot stationary points
    func_list = [func_1a, func_1b, func_1c, func_1d]
    for i in range(4):
        ax_i = ax[i//2, i%2]
        contour_points_plot(ax_i, func_list[i], grid_range_min, grid_range_max, stationary_points[i])
    
    """ End of your code
    """
    return fig


def contour_points_plot(ax, func, grid_range_min, grid_range_max, stationary_points):
    """Helper function for task1 to plot contour and stationary points for each axis
    @param ax: axis to plot
    @param func: function to plot
    @param x1_min: lowest value of x1 to show on graph
    @param x1_max: highest value of x1 to show on graph
    @param stationary_points: list of stationary points to plot
    """
    # convert x1, x2 to meshgrid for easier calculation, then calculate z
    x1, x2 = np.meshgrid(np.linspace(grid_range_min, grid_range_max), np.linspace(grid_range_min, grid_range_max))
    z = func(np.array([x1, x2]))

    # plot contour with 50 levels
    ax.contour(x1, x2, z, 50)

    # plot stationary points
    for (x1, x2, point_type) in stationary_points:
        ax.plot(x1, x2, 'o', label=point_type)
    ax.legend()

# Modify the function bodies below to be used for function value and gradient computation


def approx_grad_task1(
    func: Callable[[np.ndarray], float], x: np.ndarray, eps: float
) -> np.ndarray:
    """Numerical Gradient Computation
    @param func function that takes a vector
    @param x Vector of size (2,)
    @param eps small value for numerical gradient computation
    This function shall compute the gradient approximation for a given point 'x' and a function 'func'
    using the given central differences formulation for 2D functions. (Task1 functions)
    @return The gradient approximation
    """
    assert len(x) == 2
    # Calculate partial derivative with respect to x1 and x2, then put into an array as gradient
    grad_x1 = (func(np.array([x[0] + eps, x[1]])) - func(np.array([x[0] - eps, x[1]]))) / (2 * eps)
    grad_x2 = (func(np.array([x[0], x[1] + eps])) - func(np.array([x[0], x[1] - eps]))) / (2 * eps)
    grad = np.array([grad_x1, grad_x2])
    return grad


def approx_grad_task2(
    func: Callable[[np.ndarray], float], x: np.ndarray, eps: float
) -> np.ndarray:
    """Numerical Gradient Computation
    @param func function that takes a vector
    @param x Vector of size (n,)
    @param eps small value for numerical gradient computation
    This function shall compute the gradient approximation for a given point 'x' and a function 'func'
    using scipy.optimize.approx_fprime(). (Task2 functions)
    @return The gradient approximation
    """
    return approx_fprime(x, func, eps)


def func_1a(x: np.ndarray) -> float:
    """Computes and returns the function value for function 1a) at a given point x
    @param x Vector of size (2,)
    """
    x1 = x[0]
    x2 = x[1]
    f = x1**2 - 6*x1*x2 + 9*x2**2 + 5*x1 - 15*x2 + 6.25
    return f


def grad_1a(x: np.ndarray) -> np.ndarray:
    """Computes and returns the analytical gradient result for function 1a) at a given point x
    @param x Vector of size (2,)
    """
    x1 = x[0]
    x2 = x[1]
    f_derivative_x1 = 2*x1 - 6*x2 + 2*d
    f_derivative_x2 = -6*x1 + 18*x2 - 6*d
    return np.array([f_derivative_x1, f_derivative_x2])


def func_1b(x: np.ndarray) -> float:
    """Computes and returns the function value for function 1b) at a given point x
    @param x Vector of size (2,)
    """
    x1 = x[0]
    x2 = x[1]
    f = (x1-2)**2 + x1*(x2**2) - 2
    return f


def grad_1b(x: np.ndarray) -> np.ndarray:
    """Computes and returns the analytical gradient result for function 1b) at a given point x
    @param x Vector of size (2,)
    """
    x1 = x[0]
    x2 = x[1]
    f_derivative_x1 = 2*x1 - 4 + x2**2
    f_derivative_x2 = 2*x1*x2
    return np.array([f_derivative_x1, f_derivative_x2])


def func_1c(x: np.ndarray) -> float:
    """Computes and returns the function value for function 1c) at a given point x
    @param x Vector of size (2,)
    """
    x1 = x[0]
    x2 = x[1]
    f = 2*(x1**2) + x1**3 + x1*(x2**2) + x2**2
    return f


def grad_1c(x: np.ndarray) -> np.ndarray:
    """Computes and returns the analytical gradient result for function 1c) at a given point x
    @param x Vector of size (2,)
    """
    x1 = x[0]
    x2 = x[1]
    f_derivative_x1 = 4*x1 + 3*(x1**2) + x2**2
    f_derivative_x2 = 2*x1*x2 + 2*x2
    return np.array([f_derivative_x1, f_derivative_x2])


def func_1d(x: np.ndarray) -> float:
    """Computes and returns the function value for function 1d) at a given point x
    @param x Vector of size (2,)
    """
    x1 = x[0]
    x2 = x[1]
    f = alpha*x1**2 + beta*x2**2 - 2*x1
    return f


def grad_1d(x: np.ndarray) -> np.ndarray:
    """Computes and returns the analytical gradient result for function 1d) at a given point x
    @param x Vector of size (2,)
    """
    x1 = x[0]
    x2 = x[1]
    f_derivative_x1 = 2*alpha*x1 - 2
    f_derivative_x2 = 2*beta*x2
    return np.array([f_derivative_x1, f_derivative_x2])


def func_2a(x: np.ndarray) -> float:
    """Computes and returns the function value for function 2a) at a given point x
    @param x Vector of size (n,)
    """
    return (1 / 4) * np.linalg.norm(x - b)**4

def grad_2a(x: np.ndarray) -> np.ndarray:
    """Computes and returns the analytical gradient result for function 2a) at a given point x
    @param x Vector of size (n,)
    """
    # return np.power(np.linalg.norm(x - b), 3)                 # Version 1
    return np.power(np.linalg.norm(x - b), 2) * (x - b)       # Version 2/3



def func_2b(x: np.ndarray) -> float:
    """Computes and returns the function value for function 2b) at a given point x
    @param x Vector of size (n,)
    """
    # def g(z):
    #     return (1/2) * (z**2) + z
    Ax = A @ x
    return np.sum(0.5 * Ax**2 + Ax)


def grad_2b(x: np.ndarray) -> np.ndarray:
    """Computes and returns the analytical gradient result for function 2b) at a given point x
    @param x Vector of size (n,)
    """
    Ax = A @ x
    return A.T @ (Ax + 1)


def func_2c(x: np.ndarray) -> float:
    """Computes and returns the function value for function 2c) at a given point x
    @param x Vector of size (n,)
    """
    return (x / b).T @ D @ (x / b)


def grad_2c(x: np.ndarray) -> np.ndarray:
    """Computes and returns the analytical gradient result for function 2c) at a given point x
    @param x Vector of size (n,)
    """
    return D @ (x / b) / b + D.T @ (x / b) / b


def task3():
    """Numerical Gradient Verification
    ax[0] to ax[3] Bar plot comparison, analytical vs numerical gradient for Task 1
    ax[4] to ax[6] Bar plot comparison, analytical vs numerical gradient for Task 2

    """
    fig, ax = plt.subplot_mosaic(
        [
            3 * ["1a)"] + 3 * ["1b)"] + 3 * ["1c)"] + 3 * ["1d)"],
            4 * ["2a)"] + 4 * ["2b)"] + 4 * ["2c)"],
        ],
        figsize=(15, 10),
        constrained_layout=True,
    )
    fig.suptitle("Task 3 - Numerical vs analytical", fontsize=16)
    keys = ["1a)", "1b)", "1c)", "1d)", "2a)", "2b)", "2c)"]
    for k in keys:
        ax[k].set_title(k)
        ax[k].set_xlabel(r"$\epsilon$")
        ax[k].set_ylabel("Error")

    """ Start of your code
    """
    eps = np.logspace(-7, 0, 100)

    task1_funcs = {
        "1a)": (func_1a, grad_1a),
        "1b)": (func_1b, grad_1b),
        "1c)": (func_1c, grad_1c),
        "1d)": (func_1d, grad_1d)
    }

    # Go through each func, then compare analytical and numerical gradient
    for key, (func, grad) in task1_funcs.items():
        # Generate a random test point for x range [-5, 5]
        x = np.random.rand(2) * 10 - 5

        # Calculate the analytical gradient
        analytical_grad = grad(x)

        # Calculate errors for each epsilon
        errors_l1 = []      # sum of absolute difference / total amont of error
        errors_l2 = []      # overall difference  between the analytical and numerical gradients.
        errors_linf = []    # worst case error
        
        # Loop through all epsilon, from very small up to 1, to show difference in error
        for e in eps:
            # Compute the numerical gradient approximation
            numerical_grad = approx_grad_task1(func, x, e)

            # Compute errors using different norms (l1, l2, linf)
            error = numerical_grad - analytical_grad
            errors_l1.append(np.linalg.norm(error, 1))  
            errors_l2.append(np.linalg.norm(error, 2))
            errors_linf.append(np.linalg.norm(error, np.inf))

        # Plot errors on a logarithmic scale for each norm
        ax[key].semilogx(eps, errors_l1, label=r"$\ell_1$")
        ax[key].semilogx(eps, errors_l2, label=r"$\ell_2$")
        ax[key].semilogx(eps, errors_linf, label=r"$\ell_\infty$")
        ax[key].legend()


    ############################### The same code, but for task2
    task2_funcs = {
        "2a)": (func_2a, grad_2a),
        "2b)": (func_2b, grad_2b),
        "2c)": (func_2c, grad_2c)
    }

    # Go through each func, then compare analytical and numerical gradient
    for key, (func, grad) in task2_funcs.items():
        # Generate a random test point for x range [-5, 5]
        x = np.random.rand(5) * 10 - 5

        # Calculate the analytical gradient
        analytical_grad = grad(x)

        # Calculate errors for each epsilon
        errors_l1 = []      # sum of absolute difference / total amont of error
        errors_l2 = []      # overall difference  between the analytical and numerical gradients.
        errors_linf = []    # worst case error
        
        # Loop through all epsilon, from very small up to 1, to show difference in error
        for e in eps:
            # Compute the numerical gradient approximation
            numerical_grad = approx_grad_task2(func, x, e)

            # Compute errors using different norms (l1, l2, linf)
            error = numerical_grad - analytical_grad
            errors_l1.append(np.linalg.norm(error, 1))  
            errors_l2.append(np.linalg.norm(error, 2))
            errors_linf.append(np.linalg.norm(error, np.inf))

        # Plot errors on a logarithmic scale for each norm
        ax[key].semilogx(eps, errors_l1, label=r"$\ell_1$")
        ax[key].semilogx(eps, errors_l2, label=r"$\ell_2$")
        ax[key].semilogx(eps, errors_linf, label=r"$\ell_\infty$")
        ax[key].legend()

    """ End of your code
    """
    return fig


def task4():
    """Diet Problem
    Print optimal solution (total cost, amount of food and nutrients) to stdout
    """
    food_names = np.array(
        [
            "milk",
            "tomatoes",
            "bananas",
            "apples",
            "noodles",
            "lettuce",
            "bread",
            "eggs",
            "meat",
            "fish",
        ]
    )

    nutrient_names = np.array(
        ["energy", "lipids", "carbs", "proteins", "fiber", "vitamin c", "salt"]
    )

    """ Start of your code
    """

    """ End of your code
    """
    return


if __name__ == "__main__":
    tasks = [task1, task3]

    pdf = PdfPages("figures.pdf")
    for task in tasks:
        retval = task()
        pdf.savefig(retval)
    pdf.close()

    task4()
