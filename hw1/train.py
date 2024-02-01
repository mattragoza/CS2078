import matplotlib
matplotlib.use('Agg')
import sys, os
import numpy as np
import matplotlib.pyplot as plt


def xavier_init(fan_in, fan_out):
    '''
    Xavier weight initialization.

    Returns:
        (fan_in, fan_out) weight matrix
    '''
    return np.random.randn(fan_in, fan_out) / np.sqrt(fan_in)


def concat_bias(x, axis):
    '''
    Concatenate matrix with a vector of ones.
    '''
    if axis == 0:
        bias = np.ones((1, x.shape[1]))
    elif axis == 1:
        bias = np.ones((x.shape[0], 1))
    return np.concatenate([x, bias], axis=axis)


def forward(x, w1, w2):
    '''
    Neural network forward pass.

    Args:
        x: (N, D) input features
        w1: (M, D + 1) hidden weights
        w2: (K, M + 1) output weights
    Returns:
        y_pred: (N, K) output predictions
        z: (N, M) hidden activations
    '''
    (N, D) = x.shape
    M = w1.shape[0]
    K = w2.shape[0]
    assert w1.shape == (M, D + 1)
    assert w2.shape == (K, M + 1)

    a = w1 @ concat_bias(x.T, axis=0)
    z = np.tanh(a)
    assert z.shape == (M, N)

    y_pred = w2 @ concat_bias(z, axis=0)
    assert y_pred.shape == (K, N)

    return y_pred.T, z.T


def backprop(x, y, M, iters, eta):
    '''
    Neural network training.

    Args:
        x: (N, D) input features
        y: (N, K) output labels
        M: Number of hidden units
        iters: Number of iterations
        eta: Learning rate
    Returns:
        w1: (M, D + 1) hidden weights
        w2: (K, M + 1) output weights
        error: (iters + 1, 1) training error
    '''
    (N, D) = x.shape
    K = y.shape[1]
    assert y.shape == (N, K)

    # weight initialization
    w1 = xavier_init(D + 1, M).T
    w2 = xavier_init(M + 1, 1).T
    assert w1.shape == (M, D + 1)
    assert w2.shape == (K, M + 1)

    # gradient descent
    error = []
    for i in range(iters + 1):

        # forward pass
        y_pred, z = forward(x, w1, w2)
        y_diff = (y_pred - y)
        E = (y_diff**2).sum() / (2*N)
        assert y_diff.shape == (N, K)
        assert z.shape == (N, M)

        print(f'[iteration {i}] error = {E:.4f}')
        error.append(E)

        if i == iters: # stopping criteria
            break

        # backward pass
        dE_dy  = y_diff / N
        dy_dw2 = concat_bias(z, axis=1)
        dy_dz  = w2
        assert dE_dy.shape  == (N, K)
        assert dy_dw2.shape == (N, M + 1)
        assert dy_dz.shape  == (K, M + 1)

        dE_dw2 = dE_dy.T @ dy_dw2
        assert dE_dw2.shape == (K, M + 1)

        dE_dz = dE_dy @ dy_dz[:,:-1]
        assert dE_dz.shape == (N, M)

        # hidden layer
        dz_da  = (1 - z**2)
        da_dw1 = concat_bias(x, axis=1)
        da_dx  = w1
        assert dz_da.shape  == (N, M)
        assert da_dw1.shape == (N, D + 1)
        assert da_dx.shape  == (M, D + 1)

        dE_dw1 = (dE_dz * dz_da).T @ da_dw1
        assert dE_dw1.shape == (M, D + 1)

        dE_dx = (dE_dz * dz_da) @ da_dx[:,:-1]
        assert dE_dx.shape == (N, D)

        # weight update
        w1 -= eta * dE_dw1
        w2 -= eta * dE_dw2

    return w1, w2, error


if __name__ == '__main__':

    # XOR data
    x = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([[0, 1, 1, 0]]).T
    print(x)
    print(y)

    # train neural network
    w1, w2, error = backprop(x, y, M=10, iters=20000, eta=5e-3)

    # final evaluation
    y_pred, z = forward(x, w1, w2)
    print(y_pred)

    # plot training error
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(error, label='train')
    ax.legend(frameon=False)
    ax.set_xlabel('iteration')
    ax.set_ylabel('error')
    max_error = np.max(error)
    ax.set_ylim(-0.05, 0.5)
    ax.grid(linestyle=':')
    ax.set_axisbelow(True)
    fig.savefig('training.png', bbox_inches='tight')
