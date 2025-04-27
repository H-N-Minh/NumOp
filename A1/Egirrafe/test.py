import numpy as np
import matplotlib.pyplot as plt

# Define all functions and their analytical gradients from Task 1
def f1(x):
    """f(x) = (a^T x - d)^2 where a = [-1, 3]^T and d = 2.5"""
    a = np.array([-1, 3])
    d = 2.5
    return (np.dot(a, x) - d) ** 2

def grad_f1(x):
    """Analytical gradient of f1"""
    a = np.array([-1, 3])
    return 2 * (np.dot(a, x) - 2.5) * a

def f2(x):
    """f(x) = (x1 - 2)^2 + x1 * x2^2 - 2"""
    x1, x2 = x
    return (x1 - 2) ** 2 + x1 * x2 ** 2 - 2

def grad_f2(x):
    """Analytical gradient of f2"""
    x1, x2 = x
    return np.array([2 * (x1 - 2) + x2 ** 2, 2 * x1 * x2])

def f3(x):
    """f(x) = x1^2 + x1 * ||x||^2 + ||x||^2"""
    x1, x2 = x
    norm_x_squared = x1 ** 2 + x2 ** 2
    return x1 ** 2 + x1 * norm_x_squared + norm_x_squared

def grad_f3(x):
    """Analytical gradient of f3"""
    x1, x2 = x
    norm_x_squared = x1 ** 2 + x2 ** 2
    grad_x1 = 2 * x1 + 3 * x1 ** 2 + x2 ** 2
    grad_x2 = 2 * x2 * (1 + x1)
    return np.array([grad_x1, grad_x2])

def f4(x, alpha=1, beta=1):
    """f(x) = alpha * x1^2 - 2 * x1 + beta * x2^2"""
    x1, x2 = x
    return alpha * x1 ** 2 - 2 * x1 + beta * x2 ** 2

def grad_f4(x, alpha=1, beta=1):
    """Analytical gradient of f4"""
    x1, x2 = x
    return np.array([2 * alpha * x1 - 2, 2 * beta * x2])

# Central difference approximation for gradient
def numerical_gradient(f, x, epsilon):
    grad_approx = np.zeros_like(x)
    for i in range(len(x)):
        x_eps_plus = np.copy(x)
        x_eps_minus = np.copy(x)
        x_eps_plus[i] += epsilon
        x_eps_minus[i] -= epsilon
        grad_approx[i] = (f(x_eps_plus) - f(x_eps_minus)) / (2 * epsilon)
    return grad_approx

# Function to calculate and plot the error for gradient approximation
def check_gradient_approximation(f, grad_f, x):
    epsilons = np.logspace(-7, 0, 100)  # Range of epsilon values from 1e-7 to 1
    l1_errors = []
    l2_errors = []
    l_inf_errors = []

    analytical_grad = grad_f(x)

    for epsilon in epsilons:
        numerical_grad = numerical_gradient(f, x, epsilon)
        error = numerical_grad - analytical_grad
        
        # Calculate error norms
        l1_errors.append(np.linalg.norm(error, 1))
        l2_errors.append(np.linalg.norm(error, 2))
        l_inf_errors.append(np.linalg.norm(error, np.inf))

    # Plotting the errors
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, l1_errors, label=r'$\ell_1$ norm error')
    plt.plot(epsilons, l2_errors, label=r'$\ell_2$ norm error')
    plt.plot(epsilons, l_inf_errors, label=r'$\ell_\infty$ norm error')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\epsilon$ (log scale)')
    plt.ylabel('Error (log scale)')
    plt.title(f'Gradient Approximation Error for function {f.__name__}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Choose a random point x for testing
np.random.seed(42)
x_random = np.random.rand(2)

# Checking and plotting gradient approximation errors for each function
print("Checking gradient for f1...")
check_gradient_approximation(f1, grad_f1, x_random)

print("Checking gradient for f2...")
check_gradient_approximation(f2, grad_f2, x_random)

print("Checking gradient for f3...")
check_gradient_approximation(f3, grad_f3, x_random)

print("Checking gradient for f4...")
check_gradient_approximation(lambda x: f4(x, alpha=1, beta=1), lambda x: grad_f4(x, alpha=1, beta=1), x_random)
