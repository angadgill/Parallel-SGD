"""
Parallel implementation of Stochastic Gradient Descent from SciKit-Learn library.

This file sub-classes the BaseSGDRegressor class from sklearn and parallelizes
the SGD part of the code.

Author: Angad Gill
Credits: Zinkevich, Martin, et al. "Parallelized stochastic gradient descent." Advances in neural information processing systems. 2010.

"""

# from abc import abstractmethod
from sklearn.linear_model.stochastic_gradient import BaseSGDRegressor, DEFAULT_EPSILON
from sklearn.utils import check_random_state
from sklearn.linear_model.base import make_dataset
from sklearn.linear_model.sgd_fast import plain_sgd, average_sgd

import numpy as np


class ParallelSGDRegressor(BaseSGDRegressor):

    def __init__(self, loss="squared_loss", penalty="l2", alpha=0.0001,
                 l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True,
                 verbose=0, epsilon=DEFAULT_EPSILON, random_state=None,
                 learning_rate="invscaling", eta0=0.01, power_t=0.25,
                 warm_start=False, average=False):
        super(ParallelSGDRegressor, self).__init__(loss=loss, penalty=penalty,
                                               alpha=alpha, l1_ratio=l1_ratio,
                                               fit_intercept=fit_intercept,
                                               n_iter=n_iter, shuffle=shuffle,
                                               verbose=verbose,
                                               epsilon=epsilon,
                                               random_state=random_state,
                                               learning_rate=learning_rate,
                                               eta0=eta0, power_t=power_t,
                                               warm_start=warm_start,
                                               average=average)

    # TODO: Update this method to make it parallel
    def _fit_regressor(self, X, y, alpha, C, loss, learning_rate,
                       sample_weight, n_iter):
        dataset, intercept_decay = make_dataset(X, y, sample_weight)

        loss_function = self._get_loss_function(loss)
        penalty_type = self._get_penalty_type(self.penalty)
        learning_rate_type = self._get_learning_rate_type(learning_rate)

        if self.t_ is None:
            self.t_ = 1.0

        random_state = check_random_state(self.random_state)
        # numpy mtrand expects a C long which is a signed 32 bit integer under
        # Windows
        seed = random_state.randint(0, np.iinfo(np.int32).max)

        if self.average > 0:
            self.standard_coef_, self.standard_intercept_, \
                self.average_coef_, self.average_intercept_ =\
                average_sgd(self.standard_coef_,
                            self.standard_intercept_[0],
                            self.average_coef_,
                            self.average_intercept_[0],
                            loss_function,
                            penalty_type,
                            alpha, C,
                            self.l1_ratio,
                            dataset,
                            n_iter,
                            int(self.fit_intercept),
                            int(self.verbose),
                            int(self.shuffle),
                            seed,
                            1.0, 1.0,
                            learning_rate_type,
                            self.eta0, self.power_t, self.t_,
                            intercept_decay, self.average)

            self.average_intercept_ = np.atleast_1d(self.average_intercept_)
            self.standard_intercept_ = np.atleast_1d(self.standard_intercept_)
            self.t_ += n_iter * X.shape[0]

            if self.average <= self.t_ - 1.0:
                self.coef_ = self.average_coef_
                self.intercept_ = self.average_intercept_
            else:
                self.coef_ = self.standard_coef_
                self.intercept_ = self.standard_intercept_

        else:
            self.coef_, self.intercept_ = \
                plain_sgd(self.coef_,
                          self.intercept_[0],
                          loss_function,
                          penalty_type,
                          alpha, C,
                          self.l1_ratio,
                          dataset,
                          n_iter,
                          int(self.fit_intercept),
                          int(self.verbose),
                          int(self.shuffle),
                          seed,
                          1.0, 1.0,
                          learning_rate_type,
                          self.eta0, self.power_t, self.t_,
                          intercept_decay)

            self.t_ += n_iter * X.shape[0]
            self.intercept_ = np.atleast_1d(self.intercept_)
