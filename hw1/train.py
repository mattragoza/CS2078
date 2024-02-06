import matplotlib
matplotlib.use('Agg')
import sys, os
import numpy as np
import pandas as pd
import scipy.stats
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

        if False and i % N == 0: # shuffle data
            shuffle_inds = np.random.permutation(N)
            x = x[shuffle_inds]
            y = y[shuffle_inds]

        batch_inds = np.arange(i, i + B) % N
        xi = x[batch_inds]
        yi = y[batch_inds]

        # forward pass
        y_pred, z = forward(xi, w1, w2)
        y_diff = (y_pred - yi)
        E = (y_diff**2).sum() / (2*B)
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


def load_xor_dataset():
    # simple test dataset of XOR function
    x = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([[0, 1, 1,0 ]]).T
    return x, x, y, y


def load_wine_dataset(data_root='wine+quality'):

    # load training data
    data_file = os.path.join(data_root, 'winequality-red.csv')
    data = pd.read_csv(data_file, sep=';')

    # select input and output columns
    output_cols = ['quality']
    input_cols = [c for c in data.columns if c not in output_cols]
    x = data[input_cols].values
    y = data[output_cols].values

    # randomly split into train and test set
    N = len(data)
    N_test = N // 2
    N_train = N - N_test
    shuffle_inds = np.random.permutation(N)
    x_train, x_test = np.split(x[shuffle_inds], [N_train])
    y_train, y_test = np.split(y[shuffle_inds], [N_train])

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    print('MAIN')

    #x_train, x_test, y_train, y_test = load_xor_dataset()
    x_train, x_test, y_train, y_test = load_wine_dataset()

    # standardize the data
    x_mean, x_std = x_train.mean(), x_train.std()
    y_mean, y_std = y_train.mean(), y_train.std()
    x_train = (x_train - x_mean) / x_std
    y_train = (y_train - y_mean) / y_std
    x_test = (x_test - x_mean) / x_std
    y_test = (y_test - y_mean) / y_std

    print(x_train.shape, x_mean, x_std)
    print(y_train.shape, y_mean, y_std)

    # train neural network
    B = 800
    w1, w2, error = backprop(x_train, y_train, M=30, iters=50000, eta=5e-3, B=B)

    # final test evaluation
    yh_train, z = forward(x_train, w1, w2)
    yh_test, z = forward(x_test, w1, w2)

    # plot training error and correlation
    fig, ax = plt.subplots(1, 3, figsize=(12,4))

    df = pd.DataFrame(dict(error=error))
    bin_size = 10
    m = df.groupby(df.index // bin_size).mean()
    s = df.groupby(df.index // bin_size).std()
    ax[0].fill_between(
        m.index * bin_size,
        m.error + s.error,
        m.error - s.error,
        alpha=0.5,
        label='std'
    )
    ax[0].plot(m.index * bin_size, m.error, label='mean')
    ax[0].legend(frameon=False)
    ax[0].set_xlabel('iteration')
    ax[0].set_ylabel('error')
    ax[0].grid(linestyle=':')
    ax[0].set_axisbelow(True)
    ax[0].set_xlabel('iteration')
    ax[0].set_ylabel('error')

    ax[1].scatter(y_train, yh_train, alpha=0.2)
    x_plot = np.linspace(-2, 2, 100)
    res = scipy.stats.linregress(y_train[:,0], yh_train[:,0])
    ax[1].plot(
        x_plot, res.slope * x_plot + res.intercept, label=f'R = {res.rvalue:.2f}'
    )
    ax[1].set_xlabel('y_train')
    ax[1].set_ylabel('yh_train')
    ax[1].legend(frameon=False)

    res = scipy.stats.linregress(y_test[:,0], yh_test[:,0])
    ax[2].scatter(y_test, yh_test, alpha=0.2)
    res = scipy.stats.linregress(y_test[:,0], yh_test[:,0])
    ax[2].plot(
        x_plot, res.slope * x_plot + res.intercept, label=f'R = {res.rvalue:.2f}'
    )
    ax[2].set_xlabel('y_test')
    ax[2].set_ylabel('yh_test')
    ax[2].legend(frameon=False)

    fig.tight_layout()
    fig.savefig('training.png', bbox_inches='tight')

