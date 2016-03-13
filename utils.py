"""
All helper functions for this project are implemented here.

Author: Angad Gill

Code here is inspired by the following papers:
1. Zinkevich, M.(2010).Parallelized stochastic gradient descent.
Advances in Neural Information Processing Systems,
Retrieved from http://papers.nips.cc/paper/4006-parallelized-stochastic-gradient-descent

2. Niu, F. (2011). HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent.
Advances in Neural Information Processing Systems,
Retrieved from http://arxiv.org/abs/1106.5730
"""

from sys import stdout

import numpy as np
from matplotlib import pyplot as plt

from sklearn.linear_model import SGDRegressor
from joblib import Parallel, delayed
import threading


def split_data(X_train, y_train, n_jobs, split_per_job, overlap=False):
    """
    Split the data across workers. Outputs a nested list of X_train and y_train
     [[X_train for worker 1, y_train for worker 1], [X_train for worker 2, y_train for worker 2],...]

    Parameters
    ----------
    X_train: Input training data. May be split across workers, see split_per_job
    y_train: Target training dat
    n_jobs: Number of workers
    split_per_job: Fraction of input data that each worker should have
    overlap: Bool. Should there be overlap in the data split across workers, i.e. should the function use bootstraping

    Returns
    -------
    data: Outputs a nested list of X_train and y_train
     [[X_train for worker 1, y_train for worker 1], [X_train for worker 2, y_train for worker 2],...]

    """
    if overlap:  # Bootstrap the input data across workers
        data_size = len(X_train)
        # np.random.choice uses replace=False so that one worker gets unique samples
        splits = [np.random.choice(data_size, size=int(split_per_job*data_size), replace=False) for _ in range(n_jobs)]
        data = zip([X_train[split] for split in splits], [y_train[split] for split in splits])
    else:
        if split_per_job != 1/n_jobs:  # Data must be split evenly if there is no overlap
            raise Exception("split_per_job must be equal to 1/n_jobs")
        data = zip(np.split(X_train, n_jobs), np.split(y_train, n_jobs))
    return data


def sim_parallel_sgd(X_train, y_train, X_test, y_test,
                     n_iter, n_jobs, split_per_job, n_sync=1,
                     overlap=False, verbose=False):
    """
    Simulate parallel execution of SGDRegressor.

    Parameters
    ----------
    X_train: Input training data. May be split across workers, see split_per_job
    y_train: Target training data
    X_test: Input test data. Used by all workers
    y_test: Target test data
    n_iter: Number of iterations for each worker
    n_jobs: Number of simulated workers
    n_sync: Number of times weights should be syncrhonized, including the one at the end
    split_per_job: Fraction of input data that each worker should have
    overlap: Bool. Should there be overlap in the data split across workers, i.e. should the function use bootstraping

    Returns
    -------
    scores: nested list of scores of each machine in each iteration
        Each element contains scores for each machine. The last being the aggregate score
        e.g.: [[machine 1 score in iter 1, machine 2 score in iter 1, ..., aggregate score in iter 1]
               [machine 1 score in iter 2, machine 2 score in iter 2, ..., aggregate score in iter 2]
               ...
               [machine 1 score in iter n, machine 2 score in iter n, ..., aggregate score in iter n]]
    """

    """ Split data """
    data = split_data(X_train, y_train, n_jobs, split_per_job, overlap)

    """ Simulate parallel execution """
    scores = []  # List containing final output

    sgds = []  # List of SGDRegressor objects for each "worker"
    for n in range(n_jobs):
        # warm_start=True is important for iterative training
        sgds += [SGDRegressor(n_iter=1, warm_start=True)]
    sgds += [SGDRegressor()]  # For calculating aggregate score for each iteration

    for i in range(n_iter):  # Execute iterations one-by-one
        if verbose:
            stdout.write("Iteration: " + str(i))
            stdout.write('\r')

        iter_scores = []
        iter_coefs = []
        iter_intercepts = []

        for n, sgd in enumerate(sgds):  # Fit model for each "worker" one-by-by
            if n < n_jobs:
                sgd.partial_fit(data[n][0], data[n][1])  # partial_fit() allows iterative training
                iter_scores += [sgd.score(X_test, y_test)]
                iter_coefs += [sgd.coef_]
                iter_intercepts += [sgd.intercept_]
            else:
                # Calcuate aggregate score fo this iteration
                iter_coefs = np.mean(np.array(iter_coefs), axis=0)
                iter_intercepts = np.mean(np.array(iter_intercepts), axis=0)

                sgd.coef_ = iter_coefs
                sgd.intercept_ = iter_intercepts
                iter_scores += [sgd.score(X_test, y_test)]

        scores += [iter_scores]

        if i % int(n_iter/n_sync) == 0 and i != 0:  # Sync weights every (n_iter/n_sync) iterations
            if verbose:
                print "Synced at iteration:", i
            for sgd in sgds[:-1]:  # Iterate through all workers except the last (which is used for aggregates)
                sgd.coef_ = iter_coefs
                sgd.intercept_ = iter_intercepts

    return scores


def plot_scores(scores, agg_only=True):
    """
    Plot scores produced by the parallel SGD function

    Parameters
    ----------
    scores: nested list of scores produced by sim_parallel_sgd_scores
        Each element contains scores for each machine. The last being the aggregate score
        e.g.: [[machine 1 score in iter 1, machine 2 score in iter 1, ..., aggregate score in iter 1]
                ...[..]]
    agg_only: plot only the aggregated scores -- last value in each nested list

    Returns
    -------
    No return
    """
    scores = np.array(scores).T
    if not agg_only:
        for s in scores[:-1]:
            plt.figure(1)
            plt.plot(range(len(s)), s)
    plt.figure(1)
    plt.plot(range(len(scores[-1])), scores[-1], '--')


""" Parallel implementation """


def psgd_method(args):
    """
    SGD method run in parallel using map.

    Parameters
    ----------
    args: tuple (sgd, data), where
        sgd is SGDRegressor object and
        data is a tuple: (X_train, y_train)

    Returns
    -------
    sgd: object returned after executing .fit()

    """
    sgd, data = args
    X_train, y_train = data
    sgd.fit(X_train, y_train)
    return sgd


def psgd_method_1(sgd, X_train, y_train):
    """
    SGD method run in parallel using map.

    Parameters
    ----------
    args: tuple (sgd, data), where
        sgd is SGDRegressor object and
        data is a tuple: (X_train, y_train)

    Returns
    -------
    sgd: object returned after executing .fit()

    """
    sgd.fit(X_train, y_train)
    return sgd


def psgd_method2(sgd, n_iter, coef, intercept, X_train, y_train):
    """
    SGD method run in parallel using map.

    Parameters
    ----------
    args: tuple (sgd, data), where
        sgd is SGDRegressor object and
        data is a tuple: (X_train, y_train)

    Returns
    -------
    sgd: object returned after executing .fit()

    """
    # print threading.current_thread()
    # n_sync = 2
    # for i in [n_iter/n_sync for _ in range(n_sync)]:
    #     # sgd.coef_ = coef
    #     # sgd.intercept_ = intercept
    #     # sgd.partial_fit(X_train, y_train)
    #     sgd.n_iter = i
    #     sgd.fit(X_train, y_train, coef_init=coef, intercept_init=intercept)
    #     coef = sgd.coef_
    #     intercept = sgd.intercept_

    for _ in range(n_iter):
        sgd.coef_ = coef
        sgd.intercept_ = intercept
        sgd.partial_fit(X_train, y_train)
        coef = sgd.coef_
        intercept = sgd.intercept_
    return sgd


def parallel_sgd(pool, sgd, n_iter, n_jobs, n_sync, data):
    """
    High level parallelization of SGDRegressor.

    Parameters
    ----------
    pool: multiprocessor pool to use for this parallelization
    sgd: SGDRegressor instance whose coef and intercept need to be updated
    n_iter: number of iterations per worker
    n_jobs: number of parallel workers
    n_sync: number of synchronization steps. Syncs are spread evenly through out the iterations
    data: list of (X, y) data for the workers. This list should have n_jobs elements

    Returns
    -------
    sgd: SGDRegressor instance with updated coef and intercept
    """
    # eta = sgd.eta0*n_jobs
    eta = sgd.eta0
    n_iter_sync = n_iter/n_sync  # Iterations per model between syncs
    sgds = [SGDRegressor(warm_start=True, n_iter=n_iter_sync, eta0=eta)
            for _ in range(n_jobs)]

    for _ in range(n_sync):
        args = zip(sgds, data)
        sgds = pool.map(psgd_method, args)
        coef = np.array([x.coef_ for x in sgds]).mean(axis=0)
        intercept = np.array([x.intercept_ for x in sgds]).mean(axis=0)
        for s in sgds:
            s.coef_ = coef
            s.intercept_ = intercept


    sgd.coef_ = coef
    sgd.intercept_ = intercept

    return sgd


def psgd_1(sgd, n_iter_per_job, n_jobs, X_train, y_train):
    """
    Parallel SGD implementation using multiprocessing. All workers sync once after running SGD independently for
    n_iter_per_job iterations.

    Parameters
    ----------
    sgd: input SGDRegression() object
    n_iter_per_job: number of iterations per worker
    n_jobs: number of parallel processes to run
    X_train: train input data
    y_train: train target data

    Returns
    -------
    sgd: the input SGDRegressor() object with updated coef_ and intercept_
    """

    sgds = Parallel(n_jobs=n_jobs)(
        delayed(psgd_method_1)(s, X_train, y_train)
        for s in [SGDRegressor(n_iter=n_iter_per_job) for _ in range(n_jobs)])
    sgd.coef_ = np.array([x.coef_ for x in sgds]).mean(axis=0)
    sgd.intercept_ = np.array([x.intercept_ for x in sgds]).mean(axis=0)
    return sgd


def psgd_2(sgd, n_iter_per_job, n_jobs, n_syncs, X_train, y_train):
    """
    Parallel SGD implementation using multiprocessing. All workers sync n_syncs times while running SGD independently
    for n_iter_per_job iterations.

    Parameters
    ----------
    sgd: input SGDRegression() object
    n_iter_per_job: number of iterations per worker
    n_jobs: number of parallel processes to run
    n_syncs: number of syncs
    X_train: train input data
    y_train: train target data

    Returns
    -------
    sgd: the input SGDRegressor() object with updated coef_ and intercept_

    """
    # n_syncs = n_jobs
    n_iter_sync = n_iter_per_job/n_syncs  # Iterations per model between syncs

    sgds = [SGDRegressor(warm_start=True, n_iter=n_iter_sync)
            for _ in range(n_jobs)]

    for _ in range(n_syncs):
        sgds = Parallel(n_jobs=n_jobs)(
            delayed(psgd_method_1)(s, X_train, y_train) for s in sgds)
        coef = np.array([x.coef_ for x in sgds]).mean(axis=0)
        intercept = np.array([x.intercept_ for x in sgds]).mean(axis=0)
        for s in sgds:
            s.coef_ = coef
            s.intercept_ = intercept

    sgd.coef_ = coef
    sgd.intercept_ = intercept

    return sgd


def psgd_3(sgd, n_iter_per_job, n_jobs, n_syncs, X_train, y_train):
    """
    Parallel SGD implementation using multiprocessing. All workers sync n_syncs times while running SGD independently
    for n_iter_per_job iterations. Each worker will have an increased learning rate -- multiple of n_jobs.

    Parameters
    ----------
    sgd: input SGDRegression() object
    n_iter_per_job: number of iterations per worker
    n_jobs: number of parallel processes to run
    n_syncs: number of syncs
    X_train: train input data
    y_train: train target data

    Returns
    -------
    sgd: the input SGDRegressor() object with updated coef_ and intercept_

    """
    n_iter_sync = n_iter_per_job/n_syncs  # Iterations per model between syncs
    eta = sgd.eta0 * n_jobs

    sgds = [SGDRegressor(warm_start=True, n_iter=n_iter_sync, eta0=eta)
            for _ in range(n_jobs)]

    for _ in range(n_syncs):
        sgds = Parallel(n_jobs=n_jobs)(
            delayed(psgd_method_1)(s, X_train, y_train) for s in sgds)
        coef = np.array([x.coef_ for x in sgds]).mean(axis=0)
        intercept = np.array([x.intercept_ for x in sgds]).mean(axis=0)
        for s in sgds:
            s.coef_ = coef
            s.intercept_ = intercept

    sgd.coef_ = coef
    sgd.intercept_ = intercept

    return sgd


def psgd_4(sgd, n_iter_per_job, n_jobs, X_train, y_train, coef, intercept):
    """
    Parallel SGD implementation using multithreading. All workers read coef and intercept from share memory,
    process them, and then overwrite them.

    Parameters
    ----------
    sgd: input SGDRegression() object
    n_iter_per_job: number of iterations per worker
    n_jobs: number of parallel processes to run
    X_train: train input data
    y_train: train target data
    coef: randomly initialized coefs stored in shared memory
    intercept: randomly initialized intercept stored in shared memory

    Returns
    -------
    sgd: the input SGDRegressor() object with updated coef_ and intercept_
    """
    sgds = [SGDRegressor(warm_start=True, n_iter=n_iter_per_job)
            for _ in range(n_jobs)]

    sgds = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(psgd_method2) (s, n_iter_per_job, coef, intercept, X_train, y_train)
        for s in sgds)

    sgd.coef_ = np.array([x.coef_ for x in sgds]).mean(axis=0)
    sgd.intercept_ = np.array([x.intercept_ for x in sgds]).mean(axis=0)
    return sgd

