import numpy as np
import matplotlib.pyplot as plt

# # Define the functions and their analytical gradients
# def f_a(x):
#     a = np.array([-1, 3])
#     d = 2.5
#     return (np.dot(a, x) - d) ** 2

# def grad_f_a(x):
#     a = np.array([-1, 3])
#     d = 2.5
#     return 2 * (np.dot(a, x) - d) * a

# def f_b(x):
#     return (x[0] - 2) ** 2 + x[0] * (x[1] ** 2) - 2

# def grad_f_b(x):
#     return np.array([
#         2 * (x[0] - 2) + x[1] ** 2,
#         2 * x[0] * x[1]
#     ])

# def f_c(x):
#     return x[0] ** 2 + x[0] * np.linalg.norm(x) + np.linalg.norm(x) ** 2

# def grad_f_c(x):
#     norm_x = np.linalg.norm(x)
#     return np.array([
#         2 * x[0] + norm_x + x[0] / norm_x * (x[0] + x[1]),
#         x[0] / norm_x * (x[0] + x[1]) + 2 * x[1]
#     ])

# def f_d(x, alpha=1.0, beta=1.0):
#     return alpha * x[0] ** 2 - 2 * x[0] + beta * x[1] ** 2

# def grad_f_d(x, alpha=1.0, beta=1.0):
#     return np.array([
#         2 * alpha * x[0] - 2,
#         2 * beta * x[1]
#     ])

# # Central difference approximation for the gradient
# def numerical_gradient(f, x, epsilon):
#     grad_approx = np.zeros_like(x)
#     for i in range(len(x)):
#         x_step_forward = np.array(x, dtype=float)
#         x_step_backward = np.array(x, dtype=float)
#         x_step_forward[i] += epsilon
#         x_step_backward[i] -= epsilon
#         grad_approx[i] = (f(x_step_forward) - f(x_step_backward)) / (2 * epsilon)
#     return grad_approx

# # Function to calculate and plot the gradient approximation error
# def gradient_check(f, grad_f, x, epsilon_values, norm_type=np.linalg.norm):
#     errors = []
#     for epsilon in epsilon_values:
#         numerical_grad = numerical_gradient(f, x, epsilon)
#         analytical_grad = grad_f(x)
#         error = norm_type(numerical_grad - analytical_grad)
#         errors.append(error)
#     return errors

# # Generate a random point for x
# x = np.random.rand(2)

# # Define the range of epsilon values
# epsilon_values = np.logspace(-7, 0, 100)

# # Calculate errors for each function
# errors_a = gradient_check(f_a, grad_f_a, x, epsilon_values)
# errors_b = gradient_check(f_b, grad_f_b, x, epsilon_values)
# errors_c = gradient_check(f_c, grad_f_c, x, epsilon_values)
# errors_d = gradient_check(lambda x: f_d(x, alpha=1.0, beta=1.0), 
#                           lambda x: grad_f_d(x, alpha=1.0, beta=1.0), x, epsilon_values)

# # Plotting the errors for each function
# plt.figure(figsize=(10, 8))
# plt.plot(epsilon_values, errors_a, label='f_a Error', marker='o', markersize=3)
# plt.plot(epsilon_values, errors_b, label='f_b Error', marker='s', markersize=3)
# plt.plot(epsilon_values, errors_c, label='f_c Error', marker='^', markersize=3)
# plt.plot(epsilon_values, errors_d, label='f_d Error', marker='x', markersize=3)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Epsilon (log scale)')
# plt.ylabel('Gradient Approximation Error (log scale)')
# plt.title('Gradient Approximation Error vs Epsilon')
# plt.legend()
# plt.grid(True, which="both", ls="--")
# plt.show()



eps = np.random.randint(-250, 251, size=(5, 5))
print(eps)