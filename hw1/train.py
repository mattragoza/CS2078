import matplotlib
matplotlib.use('Agg')
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def xavier_init(fan_in, fan_out):
    '''
    Xavier weight initialization.

    Args:
        fan_in: num input units
        fan_out: num output units
    Returns:
        (fan_in, fan_out) weight matrix
    '''
    return np.random.randn(fan_in, fan_out) / np.sqrt(fan_in)


def append_bias(x, axis):
    '''
    Append matrix with a vector of ones.
    '''
    assert x.ndim == 2 and axis in {0, 1}
    if axis == 0:
        bias = np.ones((1, x.shape[1]))
    elif axis == 1:
        bias = np.ones((x.shape[0], 1))
    return np.append(x, bias, axis=axis)


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

    a = w1 @ append_bias(x.T, axis=0)
    z = np.tanh(a)
    assert z.shape == (M, N)

    y_pred = w2 @ append_bias(z, axis=0)
    assert y_pred.shape == (K, N)

    return y_pred.T, z.T


def backprop(x, y, M, iters, eta, B=1):
    '''
    Neural network training.

    Args:
        x: (N, D) input features
        y: (N, K) output labels
        M: Number of hidden units
        iters: Number of iterations
        eta: Learning rate
        B: batch size
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

        if B < N and i % N == 0: # shuffle
            shuffle_inds = np.random.permutation(N)
            x = x[shuffle_inds]
            y = y[shuffle_inds]

        batch_inds = np.arange(i, i + B) % N
        xi = x[batch_inds]
        yi = y[batch_inds]

        # forward pass
        y_pred, z = forward(xi, w1, w2)
        y_diff = (y_pred - yi)
        E = (y_diff**2).sum() / (2*N)
        assert y_diff.shape == (B, K), y_diff.shape
        assert z.shape == (B, M), z.shape

        print(f'[iteration {i}] error = {E:.4f}')
        error.append(E)

        if i == iters: # stopping criteria
            break

        # backward pass
        dE_dy  = y_diff / N
        dy_dw2 = append_bias(z, axis=1)
        dy_dz  = w2
        assert dE_dy.shape  == (B, K), dE_dy.shape
        assert dy_dw2.shape == (B, M + 1), dy_dw2.shape
        assert dy_dz.shape  == (K, M + 1), dy_dz.shape

        dE_dw2 = dE_dy.T @ dy_dw2
        assert dE_dw2.shape == (K, M + 1), dE_dw2.shape

        dE_dz = dE_dy @ dy_dz[:,:-1]
        assert dE_dz.shape == (B, M), dE_dz.shape

        # hidden layer
        dz_da  = (1 - z**2)
        da_dw1 = append_bias(xi, axis=1)
        da_dx  = w1
        assert dz_da.shape  == (B, M), dz_da.shape
        assert da_dw1.shape == (B, D + 1), da_dw1.shape
        assert da_dx.shape  == (M, D + 1), da_dx.shape

        dE_dw1 = (dE_dz * dz_da).T @ da_dw1
        assert dE_dw1.shape == (M, D + 1), dE_dw1.shape

        dE_dx = (dE_dz * dz_da) @ da_dx[:,:-1]
        assert dE_dx.shape == (B, D), dE_dx.shape

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

    x_mean = x.mean()
    y_mean = y.mean()

    # train neural network
    w1, w2, error = backprop(x - x_mean, y - y_mean, M=30, iters=20000, eta=5e-3, B=1)

    # final evaluation
    y_pred, z = forward(x - x_mean, w1, w2)
    y_pred = y_pred + y_mean
    print(y_pred)

    df = pd.DataFrame(dict(error=error))
    m = df.groupby(df.index//4).mean()
    s = df.groupby(df.index//4).std()

    # plot training error
    fig, ax = plt.subplots(figsize=(6,6))
    ax.fill_between(
        m.index, m.error + s.error, m.error - s.error, alpha=0.5
    )
    ax.plot(m.index, m.error, label='train')
    ax.legend(frameon=False)
    ax.set_xlabel('iteration')
    ax.set_ylabel('error')
    max_error = np.max(error)
    ax.set_ylim(-0.02, 0.2)
    ax.grid(linestyle=':')
    ax.set_axisbelow(True)
    fig.savefig('training.png', bbox_inches='tight')
