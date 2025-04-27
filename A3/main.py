import numpy as np
from scipy.optimize import approx_fprime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import json


# number of neurons in each layer
INPUT_SIZE = 4
HIDDEN_SIZE = 5     # no particaular reason for choosing 5, this relies on tweaking to find the right size
OUTPUT_SIZE = 3

class NN(object):
    def __init__(self, num_input: int, num_hidden: int, num_output: int, gradient_method: str, dtype=np.float32):
        self.num_input = num_input      # number of neurons in input layer, which is 4
        self.num_hidden = num_hidden    # number of neurons in hidden layer
        self.num_output = num_output    # number of neurons in output layer, which is 3
        self.dtype = dtype              # data type for parameters of the model (W0 W1, b0 b1), default type is float
        self.gradient_method = gradient_method      # gradient method, either 'Steepest Descent' or 'Nesterov Accelerated Gradient aka NAG'

        self.init_params()

    def init_params(self):
        # TASK 1: Initialize the parameters using a normal distribution with μ = 0 and σ = 0.05

        mean, sd = 0, 0.05

        self.theta = {}
        # weight is a matrix of size (num_hidden x num_input) (5 rows x 4 columns)
        self.theta['W1'] = np.random.normal(mean, sd, (self.num_hidden, self.num_input)).astype(self.dtype)
        self.theta['W2'] = np.random.normal(mean, sd, (self.num_output, self.num_hidden)).astype(self.dtype)
        # bias is a vector of size (num_hidden x 1) (5 rows x 1 column)
        self.theta['b1'] = np.random.normal(mean, sd, (self.num_hidden, 1)).astype(self.dtype)
        self.theta['b2'] = np.random.normal(mean, sd, (self.num_output, 1)).astype(self.dtype)

    def export_model(self):
        # export the final value of the network's parameters to a json file
        with open(f'model_{self.gradient_method}.json', 'w') as fp:
            json.dump({key: value.tolist() for key, value in self.theta.items()}, fp)

    # TASK 2: Implement the forward pass of the neural network
    def forwardPass(self, x):
        """
        Forward pass of the neural network: (formulars are given in the assignment sheet)
        1. Compute the pre-activation for the hidden layer (z1) using formula z1 = W1 * x + b1
        2. Activate the hidden layer using SiLU: a1 = z1 * sigmoid(z1)
        3. Compute the pre-activation for the output layer (z2) using formula z2 = W2 * a1 + b2
        4. Activate the output layer using softmax: a2 = exp(z2) / sum(exp(z2))
        Args:
            x: Input data. (4 x 280)
        
        Returns:
            a2: Output probabilities. (3 x 280)
        """
        # Compute the pre-activation for the hidden layer (z1)
        z1 = np.dot(self.theta['W1'], x) + self.theta['b1']
        
        # Apply the SiLU activation function to z1
        a1 = z1 * (1 / (1 + np.exp(-z1)))  # sigmoid(z1) = 1 / (1 + exp(-z1))
        
        # Compute the pre-activation for the output layer (vector z2)
        z2 = np.dot(self.theta['W2'], a1) + self.theta['b2']
        
        # Apply the softmax activation function to z2
        e_z2 = np.exp(z2)      # Exponentials of each element of the vector z2
        a2 = e_z2 / np.sum(e_z2, axis=0, keepdims=True)   # axis=0 means sum along the column, since each collumn represents a sample
    
        return a1, z1, a2
    
    # TASK 3: Implement the backward pass of the neural network
    def backwardPass(self, x, y, a1, z1, a2):
        """
        Backward pass of the neural network:
        1. Compute dz2 = a2 - y
        2. Compute gradients of Cost func w.r.t W2 and b2
        3. Compute dz1 using SiLU derivative
        4. Compute gradients of Cost func w.r.t W1 and b1
        Args:
            x: Input data (4 x 280)
            y: correct answer (3 x 280)
            a1: Activated hidden layer (Nh x 280)
            z1: Pre-activation of hidden layer (Nh x 280)
            a2: Output probabilities (3 x 280)
        
        Returns:
            gradients: Dictionary containing dW1, db1, dW2, db2.
        """
        m = x.shape[1]  # Number of samples in the batch

        # Compute dz2 = a2 - y
        dz2 = a2 - y
        
        # Compute gradients for W1 and b1
        dw2 = np.dot(dz2, a1.T) / m
        # axis=1 means sum along the row, since each collumn represents a sample, and we want to find average of all samples
        db2 = np.sum(dz2, axis=1, keepdims=True) / m        
        
        # Compute dz1 using the derivative of SiLU
        sigmoid_z1 = 1 / (1 + np.exp(-z1))
        silu_derivative = sigmoid_z1 * (1 + z1 * (1 - sigmoid_z1))
        dz1 = np.dot(self.theta['W2'].T, dz2) * silu_derivative
        
        # Compute gradients for W1 and b1
        dW1 = np.dot(dz1, x.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m

        gradients = {'dw1': dW1, 'db1': db1, 'dw2': dw2, 'db2': db2}
        return gradients

    def steepestDescent(self, x_batch, y_batch, learning_rate):
        """
        Perform one step of training: forward pass, backward pass, and update parameters using gradients from backward pass.
        Args:
            x_batch: Input data (4 x 280)
            y_batch: correct answer (3 x 280)
            learning_rate: how big of a step do we take in the direction of the gradient
        """
        # Forward pass
        a1, z1, a2 = self.forwardPass(x_batch)
        
        # Backward pass
        gradients = self.backwardPass(x_batch, y_batch, a1, z1, a2)
        
        # Update weights and biases using gradient descent
        self.theta['W1'] -= learning_rate * gradients['dw1']
        self.theta['b1'] -= learning_rate * gradients['db1']
        self.theta['W2'] -= learning_rate * gradients['dw2']
        self.theta['b2'] -= learning_rate * gradients['db2']
    
    def NAG(self, x_batch, y_batch, learning_rate, momentum, velocity):
        """
        Perform one step of training using Nesterov Accelerated Gradient (NAG).
        Args:
            x_batch: Input data (4 x 280)
            y_batch: Correct labels (3 x 280)
            learning_rate: how big of a step do we take in the direction of the gradient
            velocity: Dictionary containing the current velocity for each parameter
        Returns:
            Updated velocity for the next iteration.
        """
        # Look-ahead step: Compute the temporary parameters
        temp_theta = {
            key: self.theta[key] - momentum * velocity[key]
            for key in self.theta.keys()
        }

        # Forward pass using the temporary parameters
        z1 = np.dot(temp_theta['W1'], x_batch) + temp_theta['b1']
        a1 = z1 * (1 / (1 + np.exp(-z1)))  # SiLU activation
        z2 = np.dot(temp_theta['W2'], a1) + temp_theta['b2']
        e_z2 = np.exp(z2)
        a2 = e_z2 / np.sum(e_z2, axis=0, keepdims=True)

        # Backward pass
        gradients = self.backwardPass(x_batch, y_batch, a1, z1, a2)

        # Update velocity and parameters
        for key in self.theta.keys():
            velocity[key] = momentum * velocity[key] - learning_rate * gradients[f'd{key.lower()}']
            self.theta[key] += velocity[key]

        return velocity





def task1():
    """ Neural Network

        Requirements for the plots:
            - ax[0] Plot showing the training loss for both variants
            - ax[1] Plot showing the training and test accuracy for both variants
    """


    # Create the models
    # Model using steepest descent
    net_GD = NN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, gradient_method='GD')

    # Model using Nesterovs method
    net_NAG = NN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, gradient_method='NAG')

    net_GD.export_model()
    net_NAG.export_model()

    # Configure plot
    fig = plt.figure(figsize=[12,6])
    axs = []
    axs.append(fig.add_subplot(121))
    axs.append(fig.add_subplot(122))

    axs[0].set_title('Loss')
    axs[0].grid()

    axs[1].set_title('Accuracy')
    axs[1].grid()
    return fig

if __name__ == '__main__':

    # load the data set
    with np.load('data_train.npz') as data_set:
        # get the training data
        x_train_g = data_set['x']
        y_train_g = data_set['y']

    with np.load('data_test.npz') as data_set:
        # get the test data
        x_test_g = data_set['x']
        y_test_g = data_set['y']

    print(f'Training set with {x_train_g.shape[0]} data samples.')
    print(f'Test set with {x_test_g.shape[0]} data samples.')

    tasks = [task1]

    pdf = PdfPages('figures.pdf')
    for task in tasks:
        retval = task()
        fig = retval[0] if type(retval) is tuple else retval
        pdf.savefig(fig)
    pdf.close()

    
