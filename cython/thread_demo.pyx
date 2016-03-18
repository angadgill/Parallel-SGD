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
    with nogil:
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

    Parallel(n_jobs=n_jobs, backend='threading')(
    delayed(cython_loop)(
    x) for x in X_split)

    return Y
