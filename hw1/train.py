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


def concat_bias(x):
    bias = np.ones((x.shape[0], 1))
    return np.concatenate([x, bias], axis=1)


def forward(x, w1, w2):
    '''
    Neural network forward pass.

    Args:
        x: (N, D) input features
        w1: (M, D) hidden weights
        w2: (K, M) output weights
    Returns:
        y_pred: (N, 1) output predictions
        z: (N, M) hidden activations
    '''
    assert x.shape[1] == w1.shape[1]
    assert w1.shape[0] == w2.shape[1]
    a = w1 @ x.T
    z = np.tanh(a)
    y_pred = w2 @ z
    return y_pred.T, z.T


def backprop(x, y, M, iters, eta):
    '''
    Neural network training.

    Args:
        x: (N, D) input features
        y: (N, 1) output labels
        M: Number of hidden units
        iters: Number of iterations
        eta: Learning rate
    Returns:
        w1: (M, D) hidden weights
        w2: (K, M) output weights
        error: (iters, 1) training error
    '''
    (N, D) = x.shape
    assert y.shape == (N, 1)

    w1 = xavier_init(D, M).T
    w2 = xavier_init(M, 1).T

    error = []
    for i in range(iters + 1):

        y_pred, z = forward(x, w1, w2)
        y_diff = (y_pred - y)
        E = (y_diff**2).mean() / 2

        print(f'[iteration {i}] error = {E:.4f}')
        error.append(E)

        if i == iters: # stopping criteria
            break

        # backward pass
        dE_dy  = y_diff / N
        dy_dw2 = z
        dy_dz  = w2
        assert dE_dy.shape  == (N, 1)
        assert dy_dw2.shape == (N, M)
        assert dy_dz.shape  == (1, M)

        dE_dw2 = dE_dy.T @ dy_dw2
        assert dE_dw2.shape == (1, M)

        dE_dz = dE_dy * dy_dz
        assert dE_dz.shape == (N, M)

        dz_da  = (1 - z**2)
        da_dw1 = x
        da_dx  = w1
        assert dz_da.shape  == (N, M)
        assert da_dw1.shape == (N, D)

        dE_dw1 = dE_dz.T @ da_dw1
        assert dE_dw1.shape == (M, D)

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

    x = concat_bias(x)
    w1, w2, error = backprop(x, y, M=3, iters=50000, eta=1e-4)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(error, label='train')
    ax.legend(frameon=False)
    ax.set_xlabel('iteration')
    ax.set_ylabel('error')
    max_error = np.max(error)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(linestyle=':')
    ax.set_axisbelow(True)
    fig.savefig('training.png', bbox_inches='tight')
