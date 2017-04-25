#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 23:44:54 2017

@author: Alex Andonian

This module implements a simple, feed foward neural network class that provides
the basic functionality to complete the Coding Problem in `coding_task.pdf`.
A Jupyter notebook `coding_task.ipynb` accompanies this module and demonstrates
basic usage of the `FullyConnectedNet` class.

Example:
    To run this module:

        $ python neural_net.py

Dependences:
    The following python packages are required and easily installed via pip::

        numpy
        scipy
        matplotlib
        mpl_toolkits
        autograd
        tensorflow

"""

from __future__ import print_function

import tensorflow as tf
import autograd.numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, gridspec
from mpl_toolkits.mplot3d import Axes3D
from autograd import grad
from scipy.optimize import fmin_l_bfgs_b


class FullyConnectedNet(object):
    """
    A feed-forward, fully-connected (dense) neural network with an arbitrary number
    of hidden layers, activations functions (nonlinearlities) and regularized cost.
    The net has an input dimension of N, a hidden layer dimension of I_l, and performs
    classification over O classes (non-linear functions in this case). We train the
    network with a mean squared error loss function and L2 regularization on the final
    weight matrix. The network uses a ReLU nonlinearity (by default) after all but the
    last fully connected layer.

    For a network with L layers, the architecture will be:

    Input - {Affine - ReLU  } x (L - 1) - Affine - Loss

    where the {...} block is repeated L - 1 times.

    The outputs of the last fully-connected layer are the approximations for each
    nonlinear function h_i evaluated at all the points in x.
    """

    # 'Private' Class Attributes ----------------------------------------------

    _loss_repr = {'mse': 'Mean Squared Error',
                  'reg_mse': 'Regularized Least Squares'}
    _activ_repr = {'relu': 'Rectified Linear Units (ReLU)',
                   'tanh': 'Hyperbolic Tangent (tanh)',
                   'sigmoid': 'Sigmoid Function'}

    _back_repr = {'scipy': 'Scipy L-BFGS-B',
                  'tf': 'Tensorflow Stochastic Gradient Descent'}
    _losses = _loss_repr.keys()
    _activations = _activ_repr.keys()
    _backends = _back_repr.keys()

    def __init__(self, layer_sizes, activation='relu', loss='reg_mse', C=1, backend='scipy'):
        """
        Initialize a new FullyConnectedNet model. Weights are initialized to
        small random values and biases are initialized to zero. Weights and
        biases are stored in the variable self.params, which is a dictionary
        with the following keys:

        X0: Input data; has shape (D, N)
        b0: Input layer biases; has shape (N,)
        W1: First layer weights; has shape (N, I)
        b1: First layer biases; has shape (I,)
        W2: Second layer weights; has shape (I, O)
        b2: Second layer biases; has shape (O,)

        Args:
            layer_sizes (list): A list of intergers giving the size of each layer.
                Format:
                    [input_size, hidden1_size, ..., hiddenN_size, output_size]
                Example:
                    layer_sizes = [N, I, O]
            activation (str): The activation function applied after each layer.
            loss (str): The loss function applied to the last layer.
            backend (str): The desired optimizer (SciPy or Tensorflow)

        """

        self.layer_sizes = layer_sizes
        self.activation = activation
        self.loss = loss
        self.backend = backend
        if self.backend == 'tf':
            self.params = self._tf_init_params()
        else:
            self.params = self._scipy_init_params()

    def __repr__(self):

        # Display various properties
        rep = "Feed-Forward Neural Network with {} Fully-Connected Layers \n"
        rep = rep.format(self.num_layers)
        rep += "Activation function: {} \n"
        rep = rep.format(FullyConnectedNet._activ_repr[self.activation])
        rep += "Loss/Cost function: {} \n"
        rep = rep.format(FullyConnectedNet._loss_repr[self.loss])
        rep += "Backend Optimization: {} \n"
        rep = rep.format(FullyConnectedNet._back_repr[self.backend])

        layer_i = " Layer {} "
        arrow = " --> "
        label = " " * len("Inputs ({}) --> ".format(self.layer_sizes[0]))
        diag = "Inputs ({}) --> ".format(self.layer_sizes[0])

        for i, lshapes in enumerate(self.layer_shapes):
            if i + 1 == self.num_layers:
                op = self.loss
            else:
                op = self.activation

            add = "{{affine - {} {}}}".format(op, lshapes)
            diag += add + arrow
            n = (len(add))
            dash = "-" * (np.round(((n - len(layer_i)) / 2)))
            label += dash + layer_i.format(i + 1) + dash + " " + " " * len(arrow)

        label += "\n"
        diag += "Outputs ({})".format(self.layer_sizes[-1])

        return rep + "Network Diagram: \n" + label + diag

    # Instance Attributes -----------------------------------------------------

    @property
    def layer_sizes(self):
        return self._layer_sizes

    @layer_sizes.setter
    def layer_sizes(self, sizes):
        sizes = np.array(sizes)
        self._layer_sizes = sizes
        self._num_layers = len(sizes) - 1
        self._layer_shapes = list(map(list, zip(sizes[:-1], sizes[1:])))

    @property
    def num_layers(self):
        return self._num_layers

    @num_layers.setter
    def num_layers(self, num_layers):
        self._num_layers = num_layers

    @property
    def layer_shapes(self):
        return self._layer_shapes

    @layer_shapes.setter
    def layer_shapes(self, layer_shapes):
        self._layer_shapes = layer_shapes

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, act):

        act = act.lower()
        if act in ['relu']:
            self._activation = 'relu'
        elif act in ['tanh', 'hyperbolic_tan']:
            self._activation = 'tanh'
        elif act in ['sigm', 'sigmoid', 'sigmoid_function', 'sigma']:
            self._activation = 'sigmoid'
        else:
            print('ERROR: Activation function not recognized: defaulting to ReLU.')
            self._backend = 'relu'

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, back):

        back = back.lower()
        if back in ['scipy', 'numpy', 'np']:
            self._backend = 'scipy'
            self._params = self._scipy_init_params()
        elif back in ['tensorflow', 'tf']:
            self._backend = 'tf'
            self._params = self._tf_init_params()
        else:
            print('ERROR: Backend optimizer {} not recognized.'.format(back))
            print('Defaulting to SciPy...')
            self._backend = 'scipy'
            self._params = self._scipy_init_params()

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        self._params = params

    # Instance Methods --------------------------------------------------------

    def init_params(self):
       # TODO: call appropriate initilization fn given a backend.
        pass

    def forward_pass(self, input, params=None, backend=None):
        """Compute the forward pass of a neural network given an input `input`.

        Args:
            input (ndarray): Input data with shape (D, N).
            params (1D array/dict): Weights and biases for the network.
                (1D array) if backend is scipy
                (dict) if backend is tf
                * Note: Soon, this will handled behind the scenes.
            backend (stf): scipy or tf. Default: scipy

        Returns:
            out (ndarray): Output data array with shape (I_L, O).
        """

        if backend not in FullyConnectedNet._backends:
            backend = self.backend

        if params is None:
            params = self.params

        if backend is 'scipy':
            out, w2 = self._scipy_forward_pass(input, params)
        elif backend is 'tf':
            out = self._tf_forward_pass(input)
        return out

    def Cost(self, params, X, Y, C=1):
        """ Compute the of cost of params given inputs X and outputs Y. """

        if self.backend is 'scipy':
            cost = self._scipy_cost(params, X, Y, C=C)
        elif self.backend is 'tf':
            cost = self._tf_eval_cost(params, X, Y, C=C)
        return cost

    def dCost(self, params, X, Y, C=1):
        """ Compute the gradient of the cost. """

        f = grad(self.Cost)
        return f(params, X, Y, C)

    def train(self, X, Y, C=1, backend=None, reinitialize=False,
              learning_rate=0.01, num_iter=5e4, verbose=False):
        """
        Train this neural network using SciPy's L_FBGS_B minimizer or Tensorflow's
        stochastic gradient descent optimizer.

        Args:
            X (ndarray): Input training data array with shape (D, N).
            Y (ndarray): Output training data array with shape (I_L, O).
            C (int/float): Scalar indicating the regularization strength.
            backend (str): The desired optimizer ('scipy' or 'tf')
            learning_rate (int/float): The learning rate of optimization.
            num_iter (int): Number of iterations during optimization.
            verbose (bool): If true, print progress during optimization.

        Returns:
            stats (dict): Error and cost histories generated during training.
        """

        C = 1 if C is None else C
        if backend not in FullyConnectedNet._backends:
            backend = self.backend

        error_history = []
        cost_history = []

        def _record_history(params):
            curr_error = self._scipy_cost(params, X, Y, C=0)
            curr_cost = self._scipy_cost(params, X, Y, C=1)
            iter = len(error_history)
            if verbose and (iter % 1000 == 0):
                print('iter: {:d} / {:d}; '.format(iter, int(num_iter)) +
                      'current error = {:.9f}; '.format(curr_error) +
                      'current cost = {:.9f}'.format(curr_cost))
            iter += 1
            error_history.append(curr_error)
            cost_history.append(curr_cost)

        if backend == 'scipy':

            def _cost(params, X, Y, C=C):
                return self._scipy_cost(params, X, Y, C=C)

            if verbose:
                print("Using {} for training".format(self.backend))

            if reinitialize:
                x0 = self._scipy_init_params()
            else:
                x0 = self.params
            result = fmin_l_bfgs_b(_cost,
                                   x0=x0,
                                   fprime=self.dCost,
                                   args=(X, Y),
                                   maxiter=num_iter,
                                   maxfun=num_iter,
                                   disp=100,
                                   factr=1e4,
                                   callback=_record_history)
            self.params = result[0]
            training_stats = {'error_history': error_history,
                              'cost_history': cost_history}

        elif backend == 'tf':

            final_out, training_stats = self._tf_train(X, Y,
                                                       C=C,
                                                       learning_rate=learning_rate,
                                                       num_iter=num_iter,
                                                       verbose=verbose)

        return training_stats

    # 'Private' instance methods ----------------------------------------------

    def _scipy_init_params(self):
        """ Generate a list of weight matrices, one for each layer. """

        b0_size = self.layer_sizes[0]
        lshapes_with_bias = [[l[0] + 1, l[1]] for l in self.layer_shapes]

        # List (m + 1) x n parameter arrays for each layer (+1 for bias)
        params = [np.random.randn(m, n) * np.sqrt(2.0 / m) for m, n in lshapes_with_bias]
        b0 = [np.random.random(b0_size) * np.sqrt(2.0 / b0_size)]
        return self._scipy_flatten_params(b0 + params)

    def _scipy_cost(self, params, X, Y, C=None):
        """Compute the of cost when using scipy backend."""

        C = 1 if C is None else C
        out, w2 = self._scipy_forward_pass(X, params)
        error = np.mean(np.power((out - Y), 2))
        cost = error + C * np.square(np.linalg.norm(w2))
        return cost

    def _scipy_flatten_params(self, shaped_params):
        """ Convert a list of weights for each layer to a 1D array. """
        return np.array([w for l in shaped_params for w in l.flat])

    def _scipy_shape_params(self, flat_params):
        """ Convert a 1D array of weights to a list of weights for each layer. """

        idx = self.layer_sizes[0]
        shaped_params = [flat_params[0:idx]]
        lshapes_with_bias = [[l[0] + 1, l[1]] for l in self.layer_shapes]
        weights_per_layer = np.prod(np.array(lshapes_with_bias), axis=1)

        for wl, lshapes in zip(weights_per_layer, lshapes_with_bias):
            shaped_params.append(flat_params[idx:(idx + wl)].reshape(lshapes))
            idx += wl

        return shaped_params

    def _scipy_train(self, X, Y, C=1, learning_rate=0.001, num_iter=5e4, verbose=False):
        """ Initiates training using the scipy backend."""

        def _scipy_cost(params, X, Y, C=C):
                return self._scipy_cost(params, X, Y, C=C)

        result = fmin_l_bfgs_b(_scipy_cost,
                               x0=self.params,
                               fprime=self.dCost,
                               args=(X, Y),
                               maxiter=num_iter,
                               disp=100,
                               callback=_record_history)
        self.params = result[0]
        error = result[1]
        training_stats = {'error_history': error_history,
                              'cost_history': cost_history}
        return training_stats

    def _scipy_forward_pass(self, input, params=None):
        """ Compute the forward pass of the net using the scipy backend."""

        if params is None:
            params = self.params

        sparams = self._scipy_shape_params(params)
        input = input + sparams[0]
        for wb in sparams[1:]:
            output = np.dot(input, wb[:-1, :])         # wb[:-1, :]: w_i (weights)
            input = np.maximum(0, output) + wb[-1, :]  # wb[-1, :]: b_i (biases)
        return output, sparams[-1][-1, :]

    def forward_pass_old(self, input, params=None):
        """ Compute the forward pass of a neural network given an input `input`. """

        if params is None:
            params = self.params

        sparams = self._scipy_shape_params(params)
        input = input + sparams[0]
        h1 = np.dot(input, sparams[1][:-1, :])
        h1 = np.maximum(0, h1) + sparams[1][-1, :]
        w2 = sparams[2][-1, :]
        out = np.dot(h1, sparams[2][:-1, :]) + w2
        return out, w2

    def _tf_init_params(self):
        """Initialize parameters for tf net in dict.

        Returns:
            params (dict): params[layer_i] = {'weights': w_l, 'biases': b_l}

        """

        layer = 'layer_{}'
        layer_shapes = self.layer_shapes
        b0 = tf.Variable(tf.zeros([self.layer_sizes[0]]), name='b0')
        params = {layer.format(0): {'biases': b0}}
        for i, lshapes in enumerate(layer_shapes):
            l_curr = layer.format(i + 1)
            with tf.name_scope(l_curr):
                weights = tf.Variable(tf.random_normal(layer_shapes[i]), name='weights')
                biases = tf.Variable(tf.zeros(layer_shapes[i][1]), name='biases')
                params[l_curr] = {'weights': weights, 'biases': biases}
        self.params = params
        return params

    def _tf_init_placeholders(self):
        """Generate placeholder variables to represent the input tensors.

        These placeholders are used as inputs by the rest of the model building
        code.

        Returns:
            input_placeholder: input data (X) placeholder
            labels_placeholder: labels data (Y) placeholder.
        """
        lsizes = self.layer_sizes
        input_placeholder = tf.placeholder(tf.float32, [None, lsizes[0]], name='X')
        labels_placeholder = tf.placeholder(tf.float32, [None, lsizes[-1]], name='Y')
        return input_placeholder, labels_placeholder

    def _tf_inference(self, input, params=None):
        """Build the network up to where is may be used for inference.

        Args:
            input: Input placeholder

        Returns:
            output: Dictionary of the outpt tensors at each layer.
        """

        if params is None:
            params = self.params

        layer = 'layer_{}'
        l_curr = layer.format(0)
        outputs = {l_curr: input + params[l_curr]['biases']}
        for i, lshapes in enumerate(self.layer_shapes):
            l_prev = layer.format(i)
            l_curr = layer.format(i + 1)
            weights = params[l_curr]['weights']
            biases = params[l_curr]['biases']
            if i + 1 == self.num_layers:
                outputs[l_curr] = tf.matmul(outputs[l_prev], weights) + biases
            else:
                outputs[l_curr] = tf.nn.relu(tf.matmul(outputs[l_prev], weights) + biases)
        return outputs

    def _tf_logits(self, inputs, params):
        """Return output tensor with computed logits."""
        return self._tf_inference(inputs, params)['layer_{}'.format(self.num_layers)]

    def _tf_loss(self, logits, labels, loss_fn=None):
        """Calculates the loss from the logits and the labels.

        Args:
            logits: Logits tensor, float - [INPUT_DIM, NUM_OUTPUTS].
            labels: Labels tensor, float - [INPUT_DIM, NUM_OUTPUTS].

        Returns:
            loss: Loss tensor of type float.
        """
        if loss_fn is None:
            loss = tf.losses.mean_squared_error(logits, labels)
        return loss

    def _tf_cost(self, logits, labels, loss_fn=None, C=1):
        """Calculates the loss from the logits and the labels.

        Args:
            output: Dictionary of the outpt tensors at each layer.
            labels: Labels tensor.

        Returns:
            cost: Cost tensor of type float.
        """

        params = self.params

        if loss_fn is None:
            loss = tf.losses.mean_squared_error(logits, labels)
        reg = 2 * tf.nn.l2_loss(params['layer_{}'.format(self.num_layers)]['weights'])
        return tf.add(loss, C * reg)

    def _tf_training(self, cost, learning_rate):
        """Sets up the training Ops.

        Creates a summarizer to track the loss over time in TensorBoard, and
        an optimizer and applies the gradients to all trainable variables.

        Args:
            loss: Loss tensor, from loss().
            learning_rate: The learning rate to use for gradient descent.

        Returns:
            train_op: The Op for training.
        """

        # Add a scalar summary for the snapshot loss.
        tf.summary.scalar('cost', cost)

        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(cost, global_step=global_step)
        return train_op

    def _tf_train(self, X, Y, C=1, learning_rate=0.001, num_iter=1e4, verbose=False):
        """Builds full tf graph and initiates training."""

        with tf.Graph().as_default():

            # Initalize tf graph input
            x, y = self._tf_init_placeholders()

            # Initialize model parameters
            params = self._tf_init_params()

            # Build tf graph that computes the forward pass.
            outputs = self._tf_inference(x, params)

            # Compute predictions from inference model.
            logits = self._tf_logits(x, params)

            # Add loss Op to the graph.
            loss = self._tf_loss(logits, y)

            # Add regularization to the loss to produce an overall cost.
            cost = self._tf_cost(logits, y)

            # Lastly, add training Op to the graph.
            # train_op = self._training(cost, learning_rate)
            train_op = self._tf_training(cost, learning_rate)

            # Initialize list to monitor training.
            error_history = []
            cost_history = []

            # Add the variable initializer Op.
            init = tf.global_variables_initializer()

            # Create a session for running Ops on the Graph.
            sess = tf.Session()

            # Run th Op to initialize the variables.
            sess.run(init)

            # Start the training loop.
            for iter in range(1, int(num_iter) + 1):

                # Run optimization op (backprop) and cost op (to get loss value)
                _, c, e = sess.run([train_op, cost, loss], feed_dict={x: X, y: Y})

                # Display logs per iter step
                display_step = 1000
                if iter % display_step == 0 and verbose:
                    print("iter:", '%04d' % (iter), "cost = {:.9f}, error = {}".format(c, e))
                error_history.append(e)
                cost_history.append(c)

            print("Optimization finished!")
            print("The final error is {}.".format(e))

            final_out = sess.run(logits, {x: X})
            error = sess.run(loss, {x: X, y: Y})
            self.params = sess.run(params)
            return final_out, {'error_history': error_history, 'cost_history': cost_history}

    def _tf_forward_pass(self, X):
        """Compute the forward pass of the net using the tf backend."""

        with tf.Graph().as_default():
            # tf graph input
            x, y = self._tf_init_placeholders()
            params = self.params
            logits = self._tf_logits(x, params)

            # Initialize the variables and launch the graph
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                return sess.run(logits, {x: X})

    def _tf_eval_cost(self, params, X, Y, C=1):
        """Evalulates the cost given params, inputs/outputs and reg. strength."""

        # tf graph input
        x, y = self._tf_init_placeholders()
        params = self.params
        logits = self._logits(x, params)
        cost = self._tf_cost(logits, Y, C=C)

        # Initialize the variables and launch the graph
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            return sess.run(cost, {x: X})

# Helper functions -------------------------------------------------------------

def get_nonlinear_funcs():
    """Generate list of nonlinear function to be approximated by the nn."""
    funcs = [lambda x: np.power(x, 2),
             lambda x: np.power(x, 3) - 10 * np.power(x, 2) + x - 1,
             lambda x: np.power(x, 3.0 / 2.0) - 20 * np.power(x, 0.5) + 2 * x + 2,
             lambda x: 3 * np.power(x, 5.0 / 2.0) - 20 * np.power(x, 0.3) - 10 * x + 5,
             lambda x: np.sin(np.pi * x),
             lambda x: np.cos(np.pi * x),
             lambda x: np.sin(2 * np.pi * x)]
             # lambda x: np.tan(np.pi*(x + 0.5))]
    return funcs


def generate_data(input_size, funcs=get_nonlinear_funcs()):

    # X = np.tile(np.arange(0, 10, 0.1, dtype=np.float32), (input_size, 1)).T
    X = np.tile(np.arange(0, 10, 0.1), (input_size, 1)).T
    Y = np.hstack([np.array(f(X[:, 0])).reshape(-1, 1)
                   for i, f in enumerate(funcs)])
    return X, Y


def plot_approximations(X, Y, FX):
    # Plot on two subplots due to differences in scale

    # h_i(x) for i = 1, 2, 3, 4
    fig = plt.figure()
    fig.suptitle('Nonlinear function approximations', fontsize='xx-large')

    # set height ratios for sublots
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

    # the fisrt subplot
    ax0 = plt.subplot(gs[0])
    ax0.plot(X, Y[:, [0, 1, 2, 3]], 'o', linewidth=0.5)
    ax0.plot(X, FX[:, [0, 1, 2, 3]])
    ax0.spines['bottom'].set_visible(True)
    ax0.spines['bottom'].set_linewidth(2)
    plt.ylabel('Y', fontsize='x-large')
    plt.setp(ax0.get_xticklabels(), visible=False)

    # h_i(x) for i = 5, 6, 7
    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.plot(X, Y[:, [4, 5, 6]], 'o', linewidth=1)
    ax1.plot(X, FX[:, [4, 5, 6]])
    plt.xlabel('X', fontsize='x-large')
    plt.ylabel('Y', fontsize='x-large')
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)

    plt.subplots_adjust(hspace=0.01)
    plt.show()


def final_error(input_size, hidden_size, C=1, fns=get_nonlinear_funcs(), verb=False):
    """Obtain the final error trained net with given input and hidden sizes."""

    output_size = len(fns)
    layer_sizes = [input_size, hidden_size, output_size]
    X, Y = generate_data(input_size, fns)
    net = FullyConnectedNet(layer_sizes)
    train_hist = net.train(X, Y, C=C, verbose=verb)
    error = train_hist['error_history'][-1]
    res_str = 'input units = {}; hidden units = {}; final error = {}'
    # print(res_str.format(input_size, hidden_size, error))
    return error


def error_surface(input_max, hidden_max, step=10, funcs=get_nonlinear_funcs()):
    """Generate the error surface Error(input_size, hidden_size)."""

    input_sizes = np.arange(10, input_max, step)
    hidden_sizes = np.arange(10, hidden_max, step)
    X, Y = np.meshgrid(input_sizes, hidden_sizes)
    errors = np.array([final_error(x, y, fns=funcs, verb=False)
                       for x, y in zip(np.ravel(X), np.ravel(Y))])
    E = errors.reshape(X.shape)
    return X, Y, E


def plot_surface(X, Y, Z):
    """ Utility function to plot a surface Z = f(X, Y). """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('Inputs')
    ax.set_ylabel('Hidden Units')
    ax.set_zlabel('Error')
    plt.show()
    plt.savefig('error_surface.png')


if __name__ == '__main__':

    # Create a 2 layer (not including input) net to represent our function F(X)
    # and display its (default) properties:

    # For repeatability
    np.random.seed(1)

    # Network architecture
    input_size = 100   # N = 100
    hidden_size = 100  # I = 100
    output_size = 7    # O = 7
    layer_sizes = [input_size, hidden_size, output_size]

    # Init and display
    net = FullyConnectedNet(layer_sizes)
    print(net)

    # Train the network via the Limited Memory BFGS (Broyden-Fletcher-Goldfarb-Shanno algorithm)

    # Generate training data
    X, Y = generate_data(net.layer_sizes[0])

    # Perform optimization (training) using SciPy's L_BFGS_B fmin routine
    training_stats = net.train(X, Y, backend='scipy', verbose=True)

    # Plot loglog plot of training history:

    plt.loglog(training_stats['error_history'], label='Training Error')
    plt.loglog(training_stats['cost_history'], label='Training Cost')
    plt.legend(loc='upper right', fontsize='large')
    plt.xlabel('Iteration', fontsize='x-large')
    plt.ylabel('Error/Cost', fontsize='x-large')
    plt.title('Training History', fontsize='xx-large')
    plt.show()

    # Compute and visualize approximations

    # Make a forward pass with our trained network
    FX = net.forward_pass(X)[0]

    plot_approximations(X, Y, FX)

    N, I, E = error_surface(16, 16, step=2)
    plot_surface(X, Y, Z)

    training_stats0 = net.train(X, Y, C=0, reinitialize=True)

    print('Final training error with C = {}: {}'.format(1, training_stats['error_history'][-1]))
    print('Final training error with C = {}: {}'.format(0, training_stats0['error_history'][-1]))

    # Now, create another 2 layer (not including input) net with a Tensorflow backend
    # and display its (default) properties:

    # Init and display
    net_tf = FullyConnectedNet(layer_sizes, backend='tf')  # set tf as the backend
    print(net_tf)

    # Make a forward pass with our trained network
    FX1 = net_tf.forward_pass(X)
    plot_approximations(X, Y, FX1)
