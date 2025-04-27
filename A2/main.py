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
from matplotlib import patheffects
from scipy.optimize import minimize


def task1():
    """Lagrange Multiplier Problem

    Requirements for the plots:
        - ax[0] Contour plot for a)
        - ax[1] Contour plot for b)
        - ax[2] Contour plot for c)
    """

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Task 1 - Contour plots + Constraints", fontsize=16)

    ax[0].set_title("a)")
    ax[0].set_xlabel("$x_1$")
    ax[0].set_ylabel("$x_2$")
    ax[0].set_aspect("equal")

    ax[1].set_title("b)")
    ax[1].set_xlabel("$x_1$")
    ax[1].set_ylabel("$x_2$")
    ax[1].set_aspect("equal")

    ax[2].set_title("c)")
    ax[2].set_xlabel("$x_1$")
    ax[2].set_ylabel("$x_2$")
    ax[2].set_aspect("equal")

    """ Start of your code
    """
    ############################ a)
    def a_f(x_1, x_2):      # func of a)
        return x_2 - x_1
    
    def a_g1(x_1, x_2):     # inequality constraint g1(x) <= 0
        return 4*x_2 - x_1
    
    def a_g2(x_1, x_2):     # equality constraint g2(x) = 0
        return x_2 - (x_1**2)/10 + 3
    
    x_1, x_2 = np.meshgrid(np.linspace(-6, 6), np.linspace(-6, 6))

    # plot contours of function
    ax[0].contourf(x_1, x_2, a_f(x_1, x_2), 100, cmap="coolwarm")

    # plot equality constraint
    ax[0].contour(x_1, x_2, a_g2(x_1, x_2), [0], colors="blue")

    # plot inequality constraint
    cg1 = ax[0].contour(x_1, x_2, a_g1(x_1, x_2), [0], colors="red")
    cg1.set(path_effects=[patheffects.withTickedStroke(angle=90)])

    # plot special point
    ax[0].scatter(5, -1/2, label="optimal solution (5, -0.5)", zorder=3, color="cyan")

    ax[0].legend()


    
    # ########################## b)
    def b_f(x_1, x_2):
        return x_1**2 + x_2**2
    
    def b_g1(x_1, x_2):     # inequality constraint g1(x) <= 0
        return -x_1 - x_2 + 3

    def b_g2(x_1, x_2):     # inequality constraint g2(x) <= 0
        return 2 - x_2
    
    x_1, x_2 = np.meshgrid(np.linspace(-6, 6), np.linspace(-6, 6))

    # plot contours of function
    ax[1].contourf(x_1, x_2, b_f(x_1, x_2), 100, cmap="coolwarm")

    # plot inequality constraint
    cg1 = ax[1].contour(x_1, x_2, b_g1(x_1, x_2), [0], colors="red")
    cg1.set(path_effects=[patheffects.withTickedStroke(angle=90)])

    cg1 = ax[1].contour(x_1, x_2, b_g2(x_1, x_2), [0], colors="blue")
    cg1.set(path_effects=[patheffects.withTickedStroke(angle=90)])

    # plot special point
    ax[1].scatter(1, 2, label="optimal solution (1, 2)", zorder=3, color="cyan")

    ax[1].legend()
    
    #################################### c)
    def c_f(x_1, x_2):
        return (x_1-1)**2 + x_1 * (x_2**2) - 2
    
    def c_g(x_1, x_2):      # inequality constraint g(x) <= 0
        return x_1**2 + x_2**2 - 4

    x_1, x_2 = np.meshgrid(np.linspace(-6, 6), np.linspace(-6, 6))

    # plot contours of function
    ax[2].contourf(x_1, x_2, c_f(x_1, x_2), 100, cmap="coolwarm")

    # plot inequality constraint
    cg1 = ax[2].contour(x_1, x_2, c_g(x_1, x_2), [0], colors="red")
    cg1.set(path_effects=[patheffects.withTickedStroke(angle=90)])

    # plot special point
    ax[2].scatter(0,  np.sqrt(2), label=f"valid point (0,  √2)", zorder=3, color="darkgreen")
    ax[2].scatter(0, -np.sqrt(2), label=f"valid point (0, -√2)", zorder=3, color="darkgreen")
    ax[2].scatter(-0.548,  1.923, label="valid point (-0.548,  1.923)", zorder=3, color="green")
    ax[2].scatter(-0.548, -1.923, label="valid point (-0.548, -1.923)", zorder=3, color="green")

    ax[2].scatter(1, 0, label="optimal solution (1, 0)", zorder=3, color="cyan")

    ax[2].legend()

    # x_1, x_2 = np.meshgrid(np.linspace(-5, 5), np.linspace(-5, 5))

    # # plot contours of function
    # ax[0].contourf(x_1, x_2, f(x_1, x_2), 100, cmap="coolwarm")

    # # plot equality constraint
    # ax[0].contour(x_1, x_2, h(x_1, x_2), [0], colors="blue")

    # # plot inequality constraint
    # cg1 = ax[0].contour(x_1, x_2, g(x_1, x_2), [0], colors="red")
    # cg1.set(path_effects=[patheffects.withTickedStroke(angle=90)])

    # # plot special point
    # ax[0].scatter(-1 / 2, 1 / 2, label=r"$x^\ast$", zorder=3)

    # ax[0].legend()

    # task a)

    # task b)

    # task c)

    """ End of your code
    """
    return fig



def dx_lagrange(x1, x2, lam):
    return np.array([2 * (x1 - 1) - x2 - lam, -x1 - lam])


def dl_lagrange(x1, x2, lam):
    return -x1 - x2 + 4

def grad_lagrangian(x, lam):
    x1, x2 = x
    grad_x1 = 2 * (x1 - 1) - x2 - lam
    grad_x2 = -x1 - lam
    grad_lambda = -x1 + 4 - x2
    return np.array([grad_x1, grad_x2, grad_lambda])


def dx_aug_lagrange(x1, x2, lam, mu):
    return np.array([2 * (x1 - 1) - x2 - lam + mu * (x1 + x2 - 4), -x1 - lam - mu * (x1 + x2 - 4)])


def dl_aug_lagrange(x1, x2, lam, mu):
    return mu * (x1 + x2 - 4)


def lagrangian(x1, x2, lam):
    return (x1 - 1) ** 2 - x1 * x2 + lam * (-x1 - x2 + 4)


def aug_lagrangian(x1, x2, lam, mu):
    return (x1 - 1) ** 2 - x1 * x2 + lam * (-x1 - x2 + 4) + (mu / 2) * (-x1 - x2 + 4) ** 2


def g(x1, x2):
    return -x1 + 4 - x2


# Augmented Lagrangian function
def augmented_lagrangian(x, lambda_val, rho):
    x1, x2 = x
    return f(x1, x2) + lambda_val * g(x1, x2) + (rho / 2) * (g(x1, x2)) ** 2


# Function to optimize using the augmented Lagrangian method
def augmented_lagrangian_method(initial_guess, lambda_init=0.0, rho_init=1.0, max_iter=100, tol=1e-6):
    x = np.array(initial_guess)
    lambda_val = lambda_init
    rho = rho_init

    # Optimization loop
    for i in range(max_iter):
        # Minimize the augmented Lagrangian with respect to x
        result = minimize(lambda x: augmented_lagrangian(x, lambda_val, rho), x, method='BFGS')
        x = result.x

        # Compute the constraint violation
        constraint_violation = g(x[0], x[1])

        # Update the Lagrange multiplier and penalty parameter
        lambda_val += rho * constraint_violation
        rho *= 2  # Increase penalty parameter

        # Check for convergence (both the constraint violation and the optimization result)
        if abs(constraint_violation) < tol and result.success:
            print(f"Converged in {i + 1} iterations.")
            break

    return x, f(x[0], x[1])

def f(x1, x2):
    return (x1 - 1) ** 2 - x1 * x2


def task2():
    """Augmented Lagrangian Method

    Requirements for the plots:
        - ax[0] Contour plot
        - ax[1] Lagrangian over iterations
        - ax[2] Convergence to optimal solution over iterations
    """
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))
    ax[0].set_title(r"Trajectories")
    ax[1].set_title(r"Energy")
    ax[2].set_title(r"Convergence $\Vert x_k - x^\ast \Vert$")

    ax[0].set_xlabel("$x_1$")
    ax[0].set_ylabel("$x_2$")
    ax[0].set_aspect("equal")

    ax[1].set_xlabel("Iteration")
    ax[1].set_aspect("equal")

    ax[2].set_xlabel("Iteration")
    ax[2].set_aspect("equal")

    ax[0].set_xlim([-8, 8])
    ax[0].set_ylim([-8, 8])
    x1_ = np.linspace(-8, 8, 100)
    x2_ = np.linspace(-8, 8, 100)
    x1, x2 = np.meshgrid(x1_, x2_)

    fig.suptitle(
        "Task 2 - Contour plots + Constraints + Iterations over k", fontsize=16
    )
    """ Start of your code
    """
    # Plot example (remove in submission)
    z = f(x1, x2)

    lagrange_iter = []
    iter_normal = []
    aug_lagrangian_iter = []
    iter_aug = []

    x_normal = []
    x_aug = []

    # Plot example (remove in submission)
    ax[0].contourf(x1, x2, z, 100, cmap="coolwarm")
    ax[0].plot(x1[0], -x1[0] + 4, "-k")
    ax[0].scatter(3 / 2, 5 / 2, marker="+", color="r", s=100.0)

    tau = 0.01
    x1_0, x2_0 = np.random.uniform(-8, 8), np.random.uniform(-8, 8)
    lm_0 = np.random.uniform(0, 1)
    ax[0].scatter(x1_0, x2_0, marker="o", color="k", s=1000.0)
    max_iterations = 500

    learning_rate = 0.01
    tolerance = 1e-4
    x = np.array([x1_0, x2_0])  # Initial guess for x1 and x2
    lam = lm_0  # Initial guess for lambda

    for iteration in range(max_iterations):
        # Compute the gradient of the Lagrangian
        grad = grad_lagrangian(x, lam)
        # Update the variables x1, x2, and lambda
        x_new = x - learning_rate * grad[:2]  # Update for x1 and x2
        lam_new = lam + learning_rate * grad[2]  # Update for lambda
        ax[0].scatter(x[0], x[1], marker="+", color="r", s=200.0)
        iter_normal.append(iteration)
        lagrange_iter.append(lagrangian(x[0], x[1], lam))
        x_normal.append(np.sqrt((x[0] - 3 / 2) ** 2 + (x[1] - 5 / 2) ** 2))

        # Check for convergence (if the change in the variables is very small)
        if np.linalg.norm(x_new - x) < tolerance and abs(lam_new - lam) < tolerance:
            print(f"Converged in {iteration + 1} iterations.")
            break
        # Update the values for the next iteration
        x = x_new
        lam = lam_new

    x = np.array([x1_0, x2_0])
    lambda_val = lm_0
    rho = np.random.uniform(0, 1)

    # Optimization loop
    for i in range(max_iterations):
        # Minimize the augmented Lagrangian with respect to x
        result = minimize(lambda x: augmented_lagrangian(x, lambda_val, rho), x, method='BFGS')
        x = result.x
        ax[0].scatter(x[0], x[1], marker="x", color="b", s=200.0)

        iter_aug.append(i)
        aug_lagrangian_iter.append(aug_lagrangian(x[0], x[1], lambda_val, rho))
        x_aug.append(np.sqrt((x[0] - 3 / 2) ** 2 + (x[1] - 5 / 2) ** 2))
        # Compute the constraint violation
        constraint_violation = g(x[0], x[1])
        lambda_val += rho * constraint_violation
        rho *= 2  # Increase penalty parameter

        # Check for convergence (both the constraint violation and the optimization result)
        if abs(constraint_violation) < 1e-6 and result.success:
            print(f"Converged in {i + 1} iterations.")
            break

    print(len(aug_lagrangian_iter))
    aug_lagrangian_iter = aug_lagrangian_iter + [0.0] * (max_iterations - len(aug_lagrangian_iter))
    x_aug = x_aug + [0.0] * (max_iterations - len(x_aug))
    ax[1].plot(iter_normal, lagrange_iter)
    ax[2].plot(iter_normal, x_normal)
    ax[1].plot(iter_normal, aug_lagrangian_iter)
    ax[2].plot(iter_normal, x_aug)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    """ End of your code
    """
    return fig


if __name__ == "__main__":
    tasks = [task1, task2]

    pdf = PdfPages("figures.pdf")
    for task in tasks:
        retval = task()
        fig = retval[0] if type(retval) is tuple else retval
        pdf.savefig(fig)
    pdf.close()
