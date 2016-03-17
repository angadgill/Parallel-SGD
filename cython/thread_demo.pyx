"""
Code credits: http://nealhughes.net/parallelcomp2/
"""

import numpy as np
from math import exp
from libc.math cimport exp as c_exp
from cython.parallel import parallel, prange
from cython import boundscheck, wraparound

from joblib import Parallel, delayed


def numpy_array(X):

    Y = np.zeros(X.shape)
    index = X > 0.5
    Y[index] = np.exp(X[index])

    return Y


def python_loop(X):

    Y = np.zeros(X.shape)
    index = X > 0.5
    for i in index:
        Y[index] = np.exp(X[index])
    return Y


def cython_loop(double[:] X):

    cdef int N = X.shape[0]
    cdef double[:] Y = np.zeros(N)
    cdef int i

    for i in range(N):
        if X[i] > 0.5:
            Y[i] = c_exp(X[i])
        else:
            Y[i] = 0

    return Y

@boundscheck(False)
def cython_parallel(double[:] X, int n_jobs):
    cdef int N = X.shape[0]
    cdef double[:] Y = np.zeros(N)
    cdef int i

    with nogil, parallel(num_threads=n_jobs):
        for i in prange(N):
            if X[i] > 0.5:
                Y[i] = c_exp(X[i])
            else:
                Y[i] = 0

    return Y

def python_parallel(X, n_jobs):
    N = X.shape[0]
    Y = np.zeros(N)

    X_split = np.split(X, n_jobs)
    Y_split = np.split(Y, n_jobs)

    args = zip(X_split, Y_split, [N for _ in range(n_jobs)])

    Parallel(n_jobs=n_jobs, backend='threading')(
    delayed(_c_array_f_joblib)(
    x, y, i/n_jobs) for x, y, i in args)

    return Y


def _c_array_f_joblib(X, Y, N):
    cython_loop(X) 
    # for i in range(N):
    #     if X[i] > 0.5:
    #         Y[i] = c_exp(X[i])
    #     else:
    #         Y[i] = 0
