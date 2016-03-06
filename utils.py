"""
All helper functions for this project are implemented here.

Author: Angad Gill
"""
from sys import stdout

import numpy as np
from matplotlib import pyplot as plt

from sklearn.linear_model import SGDRegressor


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
    if overlap:  # Bootstrap the input data across workers
        data_size = len(X_train)
        # np.random.choice uses replace=False so that one worker gets unique samples
        splits = [np.random.choice(data_size, size=int(split_per_job*data_size), replace=False) for _ in range(n_jobs)]
        data = zip([X_train[split] for split in splits], [y_train[split] for split in splits])
    else:
        if split_per_job != 1/n_jobs:  # Data must be split evenly if there is no overlap
            raise Exception("split_per_job must be equal to 1/n_jobs")
        data = zip(np.split(X_train, n_jobs), np.split(y_train, n_jobs))

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
            # print "synced"
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


""" Trial parallel implementation"""


def psgd_method(args):
    X_train, y_train, n_iter = args
    sgd = SGDRegressor(n_iter=n_iter)
    sgd.fit(X_train, y_train)
    return sgd.coef_, sgd.intercept_


def add_results(x, y):
    return x[0]+y[0], x[1]+y[1]


def parallel_sgd(sgd, pool, X, y, n_iter, n_jobs):

    # Split data into n_jobs chunks
    print "Spliting data..."
    data = zip(np.split(X, n_jobs),
               np.split(y, n_jobs),
               [n_iter for _ in range(n_jobs)])

    # Execute in parallel
    print "Executing in parallel..."
    result = pool.map(psgd_method, data)

    # Combine results
    print "Combining results..."
#     result = reduce(add_results, result)
#     result = [x/n_jobs for x in result]
    result = np.mean(np.array(result), axis=0)

    # Add coefs and intercept to the sgd object
    coef, intercept = result
    sgd.coef_ = coef
    sgd.intercept_ = intercept

    return sgd
