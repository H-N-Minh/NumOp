# Nonlinear Optimization Course Assignments W24

This repository contains the solutions for a series of assignments from a Nonlinear Optimization course. The project explores both the theoretical foundations and practical applications of optimization algorithms, from basic function analysis to training neural networks.

### Technical Overview

*   **Programming Language:** Python
*   **Key Libraries:** NumPy, Matplotlib, SciPy
*   **Core Concepts Covered:** Mathematical Optimization, Matrix Calculus, Linear Programming, Constrained Optimization (Lagrange Multipliers, KKT Conditions), Gradient-Based Methods, and Neural Network Implementation.

---

## Assignment 1: Function Analysis and Linear Programming

This assignment focused on the fundamentals of analyzing functions and solving a classic optimization problem.

*   **Characterization of Functions:** Analyzed various mathematical functions by computing their gradients and Hessians to identify and classify stationary points (minima, maxima, or saddle points).
*   **Matrix Calculus & Gradient Verification:** Practiced computing derivatives for matrix-based functions and verified the analytical gradients against numerical approximations to ensure correctness.
*   **The Diet Problem:** Formulated a real-world nutrition problem as a Linear Program (LP) and used SciPy's solver to find the most cost-effective diet that meets specific health constraints.

## Assignment 2: Constrained Optimization Techniques

This assignment delved into methods for solving optimization problems with equality and inequality constraints.

*   **Lagrange Multiplier Problems:** Solved several constrained optimization problems analytically using the method of Lagrange multipliers and the Karush-Kuhn-Tucker (KKT) conditions. The solutions were visualized by plotting the function's level sets against the constraint boundaries.
*   **Augmented Lagrangian Method:** Implemented and compared different iterative algorithms to solve a constrained problem. This included a naive gradient descent/ascent approach and the more robust Augmented Lagrangian method, analyzing their convergence behavior.

## Assignment 3: Species Classification with a Neural Network

This final assignment involved building, training, and evaluating a neural network from scratch to solve a classification problem.

*   **Neural Network Implementation:** Built a feed-forward neural network with one hidden layer to classify penguin species based on body measurements. This included implementing the forward pass, backward pass (backpropagation), and activation functions (SiLU, Softmax).
*   **Gradient-Based Optimization:** Trained the network using two different optimization algorithms: the standard **Steepest Descent** and the more advanced **Nesterov Accelerated Gradient (NAG)** method.
*   **Model Evaluation:** Compared the performance, training loss, and convergence rates of the two optimization methods, and evaluated the final accuracy of the trained models on a test dataset.
