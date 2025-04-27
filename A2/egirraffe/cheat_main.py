#!/usr/bin/env python

""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the existing interface and return values of the task functions.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.core.fromnumeric import shape
from scipy.linalg import inv
from matplotlib.backends.backend_pdf import PdfPages
from typing import Callable
from matplotlib import patheffects

import numpy as np


def task1():
    """Lagrange Multiplier Problem

    Requirements for the plots:
        - ax[0] Contour plot for a)
        - ax[1] Contour plot for b)
        - ax[2] Contour plot for c)
    """

    lim = 5

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Task 1 - Contour plots + Constraints", fontsize=16)

    for title, a in zip(["a)", "b)", "c)"], ax):
        a.set_title(title)
        a.set_xlabel("$x_1$")
        a.set_ylabel("$x_2$")
        a.set_aspect("equal")
        a.set_xlim([-lim, lim])
        a.set_ylim([-lim, lim])

    """ Start of your code
    """
    def objective_1_a(x1, x2):
        return -x1 + 2 * x2

    def equality_constraint_1_a(x1, x2):
        return x1 + (x2 ** 2) / 2 - 3

    def inequality_constraint_1_a(x1, x2):
        return -3 * x1 - x2 / 2 + 5

    def objective_1_b(x1, x2):
        return (1 / 3) * x1 ** 2 + 3 * x2

    def equality_constraint_1_b(x1, x2):
        return ((x1 ** 2) - 3) - x2

    def inequality_constraint_1_b(x1, x2):
        return x1 / 2 - x2 - 1

    def objective_1_c(x1, x2):
        return (2 * (x1 ** 2) + x1 * x2 + 4 * (x2 ** 2)) / 2

    def equality_constraint_1_c(x1, x2):
        return x1 / 2 - x2 - 1

    def inequality_constraint_1_c(x1, x2):
        return x1 ** 2 + x2 - 3

    def create_contour_plot(axes, objective, equality_constraint, inequality_constraint, angle,
                            candidate_point_list):
        contour_levels = 10
        x1, x2 = np.meshgrid(np.linspace(-lim, lim), np.linspace(-lim, lim))

        # plot the level lines of the objective function and add labels to the lines
        contours_objective = axes.contour(x1, x2, objective(x1, x2), contour_levels)
        axes.clabel(contours_objective, fmt="%2.1f", use_clabeltext=True)

        # plot the equality constraint by using ax.contour with the level line at 0
        constraint_color = "orangered"
        axes.contour(x1, x2, equality_constraint(x1, x2), [0], colors=constraint_color)

        # plot the inequality constraint in the same way but also add indicator for the feasible region
        feasible_region_indicator = patheffects.withTickedStroke(angle=angle, length=1)
        contours_inequality = axes.contour(
            x1, x2, inequality_constraint(x1, x2), [0], colors=constraint_color
        )
        contours_inequality.set(path_effects=[feasible_region_indicator])

        # plot some (arbitrary) candidate points
        for candidate_point in candidate_point_list:
            if candidate_point[2] == 'optimal':
                color = "green"
                marker = "*"
            elif candidate_point[2] == 'valid':
                color = "black"
                marker = "o"
            else:
                # invalid
                color = "red"
                marker = "x"
            axes.scatter(candidate_point[0], candidate_point[1], c=color, marker=marker, zorder=2)

    objective_list = [objective_1_a, objective_1_b, objective_1_c]
    equality_constraint_list = [equality_constraint_1_a, equality_constraint_1_b, equality_constraint_1_c]
    inequality_constraint_list = [inequality_constraint_1_a, inequality_constraint_1_b, inequality_constraint_1_c]
    inequality_constraint_angle_list = [-90, -90, -90]
    candidate_points_list = [[[1.365, 1.808, 'invalid'], [1.913, -1.475, 'optimal'], [1, -2, 'invalid']],
                             [[-1.186, -1.593, 'optimal'], [1.686, -0.157, 'valid'], [0, -3, 'invalid']],
                             [[1.766, -0.117, 'invalid'], [-2.266, -2.133, 'invalid'], [10 / 14, -9 / 14, 'optimal']]]
    for (objective_function, equality, inequality, inequality_angle, candidate_points, i) in (
            zip(objective_list, equality_constraint_list, inequality_constraint_list, inequality_constraint_angle_list,
                candidate_points_list, range(3))):
        create_contour_plot(ax[i], objective_function, equality, inequality, inequality_angle, candidate_points)
    """ End of your code
    """

    return fig


def task2():
    """Glider Trajectory Problem

    Requirements for the plot (only main ax):
        - filled contour plot for objective function
        - contour plot for constraint at level set 0
        - mark graphically estimated optimum
        - mark analytically determined optimum
    """

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    fig.suptitle("Task 2 - Glider's Trajectory", fontsize=16)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    a = 1.0
    b = 0.5
    c = 0.0

    """ Start of your code
    """
    def objective(x1, x2):
        return x2 / x1

    def equality_constraint(x1, x2):
        return a + b * ((x2 - c) ** 2) - x1

    def inequality_constraint_1(x1, x2):
        return -x1

    def inequality_constraint_2(x1, x2):
        return -x2

    contour_levels = np.array([-2.5, -1, -0.7, -0.5, -0.2, 0.2, 0.5, 0.7, 1, 2.5])
    x1, x2 = np.meshgrid(np.linspace(0.8, 5), np.linspace(-2, 2))

    # plot the level lines of the objective function and add labels to the lines
    contours_objective = ax.contourf(x1, x2, objective(x1, x2), len(contour_levels),
                                     levels=contour_levels)
    ax.clabel(contours_objective, fmt="%2.1f", use_clabeltext=True)

    # plot the equality constraint by using ax.contour with the level line at 0
    constraint_color = "orangered"
    ax.contour(x1, x2, equality_constraint(x1, x2), [0], colors=constraint_color)

    feasible_region_indicator = patheffects.withTickedStroke(angle=-90, length=1)
    contours_inequality_1 = ax.contour(
        x1, x2, inequality_constraint_1(x1, x2), [0], colors=constraint_color
    )
    contours_inequality_1.set(path_effects=[feasible_region_indicator])

    contours_inequality_2 = ax.contour(
        x1, x2, inequality_constraint_2(x1, x2), [0], colors=constraint_color
    )
    contours_inequality_2.set(path_effects=[feasible_region_indicator])

    c_bar = fig.colorbar(contours_objective)
    c_bar.ax.set_ylabel('contour levels')
    # plot some (arbitrary) candidate points
    ax.scatter(2, np.sqrt(2), c="green", marker="*", zorder=2)
    ax.scatter(2, - np.sqrt(2), c="red", marker="x", zorder=2)

    """ End of your code
    """

    plt.legend()
    return fig


if __name__ == "__main__":
    tasks = [task1, task2]

    pdf = PdfPages("figures.pdf")
    for task in tasks:
        retval = task()
        fig = retval[0] if type(retval) is tuple else retval
        pdf.savefig(fig)
    pdf.close()
