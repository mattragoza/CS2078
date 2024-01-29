import matplotlib
matplotlib.use('Agg')

import collections
import random
import time

import matplotlib.pyplot as plt
import numpy as np

# You should not use any other libraries.


def generate_random_numbers(num_rows=1000000, num_cols=1, mean=0.0, std=5.0):
    """Generates random numbers using `numpy` library.

    Generates a vector of shape 1000000 x 1 (one million by one) with random
    numbers from Gaussian (normal) distribution, with mean of 0 and standard
    deviation of 5.

    Note: You can use `num_rows`, `num_cols`, `mean` and `std` directly so no need
    to hard-code these values.

    Hint: This can be done in one line of code.

    Args:
        num_rows: Optional, number of rows in the matrix. Default to be 1000000.
        num_cols: Optional, number of columns in the matrix. Default to be 1.
        mean: Optional, mean of the Gaussian distribution. Default to be 0.
        std: Optional, standard deviation of the Gaussian dist. Default to be 5.

    Returns:
        ret: A np.ndarray object containing the desired random numbers.
    """
    ret = np.random.normal(mean, std, (num_rows, num_cols))
    return ret


def add_one_by_loop(matrix):
    """Adds one to all elements in a given np.ndarray *using* loop.

    Hints:
    - The following link may be helpful for determining how many times to loop:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html

    Args:
        matrix: A np.ndarray of shape [M, N] with arbitrary values. M represents
            number of rows in the matrix, and N represents number of columns.

    Returns:
        ret: A np.ndarray of the same shape as `matrix`, with each element being
            added by 1.
    """
    ret = matrix.copy()
    M, N = matrix.shape
    for i in range(M):
        for j in range(N):
            ret[i,j] += 1

    return ret


def add_one_without_loop(matrix):
    """Adds one to all elements in a given np.ndarray *without* using loop.

    Args:
        matrix: A np.ndarray of shape [M, N] with arbitrary values.

    Returns:
        ret: A np.ndarray of the same shape as `matrix`.
    """
    ret = matrix.copy()
    ret[...] += 1
    return ret


def measure_time_consumptions():
    """Measures the time for executing functions.

    Measures the time consumption for `add_one_by_loop` and `add_one_without_loop`
    after you completed the two functions above. You can create a random matrix
    with the function `generate_random_numbers` as input, assuming you have
    completed that function.
    Please remember to print the execution time in your code, and write down the
    numbers in answers.txt as well.

    Hint:
    - Python has built-in libararies `time` that you may find helpful:
    https://docs.python.org/3/library/time.html

    Args:
        No argument is required.

    Returns:
        None.
    """
    matrix = generate_random_numbers()
    t0 = time.time()
    add_one_by_loop(matrix)
    t1 = time.time()
    add_one_without_loop(matrix)
    t2 = time.time()
    dt1 = t1 - t0
    dt2 = t2 - t1
    print(f'add_one_by_loop:      {dt1:.4f}s')
    print(f'add_one_without_loop: {dt2:.4f}s (x{dt1/dt2:.2f} speedup)')
    return None


def plot_without_loop(saving_path="exponential.png"):
    """Plots in Python3 with `matplotlib` library.

    Plot the exponential function 2**x, for non-negative even values of x smaller
    than 30, *without* using loops.

    Args:
        saving_path: Optional, path for saving the plot.

    Returns:
        None.
    """
    x = np.arange(0, 30, 2)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(x, 2**x, label='$y = 2^x$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Exponential function')
    ax.legend(frameon=False)
    ax.grid(linestyle=':')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(saving_path, bbox_inches='tight')
    return None


def matrix_multiplication_by_loop(matrix_a, matrix_b):
    """Calculates the matrix multiplication *using* loop.

    Given two matrices `matrix_a` and `matrix_b`, calculates the product of the
    two matrices using loop. You are *NOT* allowed to use the built-in matrix
    multiplication in this question, such as `np.matmul`, `np.dot` or `@`. You
    should implement the function using loops.

    Args:
        matrix_a: A np.ndarray of shape [M, N] with arbitrary values.
        matrix_b: A np.ndarray of shape [N, K] with arbitrary values.

    Returns:
        ret: A np.ndarray of shape [M, K] which is equivalent to the product of
            `matrix_a` by `matrix_b`.
    """
    assert matrix_a.shape[1] == matrix_b.shape[0]

    M, N = matrix_a.shape
    N, K = matrix_b.shape
    ret = np.zeros((M, K))
    for i in range(M):
        for j in range(N):
            for k in range(K):
                ret[i,k] += matrix_a[i,j] * matrix_b[j,k]

    # The following code is to verify that your implementation is correct.
    assert np.all(np.isclose(ret, matrix_a @ matrix_b)), 'matmul failed'
    print('matmul passed')
    return ret


def matrix_manipulation():
    """Generates a matrix of shape [10, 10] with elements ranging from 0 to 99.

    In this question you need to take practice of manipulating numpy matrix. More
    specifically, given a np.ndarray of shape [10, 1] with elements from 0 to 9,
    you need to manipuate the vector to obtain a matrix of shape [10, 10] and
    contains elements from 0 to 99. You *cannot* manually create this new matrix,
    instead, the matrix should come from the given vector via matrix manipulation
    (addition, broadcast, transpose, etc.).

    Hints: You may find the following numpy functions useful:
    - https://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast_to.html
    - https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.T.html

    Args:
        No argument is required.

    Returns:
        ret: A np.ndarray matrix of shape [10, 10] containing elements from 0 to 99.
    """
    vector = np.expand_dims(np.arange(10), 1)
    ret = vector * 10 + vector.T
    return ret


def normalize_rows(matrix):
    """Normalizes the given matrix in a row-wise manner *without* using loops.

    By row-wise normalizing a matrix, the sum of the entries in each row should be
    1. If the elements in a row were not identical before the normalization, they
    should not be identical after the normalization. The relative order should be
    preserved though.

    Note: Assume that all elements in the matrix are *non-negative*, and all rows
    contain *at least* one non-zero element.

    Hint: This can be done in one line of code.

    Args:
        matrix: A np.ndarray of shape [M, N] with non-negative values.

    Returns:
        ret: A np.ndarray of the same shape as `matrix`.
    """
    assert np.all(matrix >= 0) and np.all(matrix.sum(axis=1) > 0)
    ret = matrix.astype(np.float32, copy=True)
    ret /= matrix.sum(axis=1, keepdims=True)
    assert np.allclose(ret.sum(axis=1), 1), 'normalize failed'
    print('normalize passed')
    return ret


if __name__ == "__main__":
    # Your submission should at least run successfully with the below function
    #   calls as a minimal test cases.
    # Feel free to implement more test cases to test your functions more
    #   thoroughly; we have more for grading.

    matrix = generate_random_numbers()
    measure_time_consumptions()
    plot_without_loop()
    matrix_multiplication_by_loop(np.random.rand(4, 100), np.random.rand(100, 3))
    matrix_manipulation()
    normalize_rows(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
