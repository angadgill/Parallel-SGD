from multiprocessing import Process, Value, Array

def fit_binary(est, i, X, y, alpha, C, learning_rate, n_iter,
               pos_weight, neg_weight, sample_weight):
    """Fit a single binary classifier.

    The i'th class is considered the "positive" class.
    """
    # if average is not true, average_coef, and average_intercept will be
    # unused
    y_i, coef, intercept, average_coef, average_intercept = \
        _prepare_fit_binary(est, y, i)
    assert y_i.shape[0] == y.shape[0] == sample_weight.shape[0]
    dataset, intercept_decay = make_dataset(X, y_i, sample_weight)

    penalty_type = est._get_penalty_type(est.penalty)
    learning_rate_type = est._get_learning_rate_type(learning_rate)

    # XXX should have random_state_!
    random_state = check_random_state(est.random_state)
    # numpy mtrand expects a C long which is a signed 32 bit integer under
    # Windows
    seed = random_state.randint(0, np.iinfo(np.int32).max)

    if not est.average:
        return parallelizer(coef, intercept, est.loss_function,
                         penalty_type, alpha, C, est.l1_ratio,
                         dataset, n_iter, int(est.fit_intercept),
                         int(est.verbose), int(est.shuffle), seed,
                         pos_weight, neg_weight,
                         learning_rate_type, est.eta0,
                         est.power_t, est.t_, intercept_decay)

    else:
        standard_coef, standard_intercept, average_coef, \
            average_intercept = average_sgd(coef, intercept, average_coef,
                                            average_intercept,
                                            est.loss_function, penalty_type,
                                            alpha, C, est.l1_ratio, dataset,
                                            n_iter, int(est.fit_intercept),
                                            int(est.verbose), int(est.shuffle),
                                            seed, pos_weight, neg_weight,
                                            learning_rate_type, est.eta0,
                                            est.power_t, est.t_,
                                            intercept_decay,
                                            est.average)

        if len(est.classes_) == 2:
            est.average_intercept_[0] = average_intercept
        else:
            est.average_intercept_[i] = average_intercept

        return standard_coef, standard_intercept
		
def parallelizer (coef, intercept, est.loss_function,
                         penalty_type, alpha, C, est.l1_ratio,
                         dataset, n_iter, int(est.fit_intercept),
                         int(est.verbose), int(est.shuffle), seed,
                         pos_weight, neg_weight,
                         learning_rate_type, est.eta0,
                         est.power_t, est.t_, intercept_decay):
						 
 

	for i in range(0,10):
		p = Process(target=plain_sgd, args=(coef, intercept, est.loss_function,
                         penalty_type, alpha, C, est.l1_ratio,
                         dataset, n_iter, int(est.fit_intercept),
                         int(est.verbose), int(est.shuffle), seed,
                         pos_weight, neg_weight,
                         learning_rate_type, est.eta0,
                         est.power_t, est.t_, intercept_decay))
		p.start()
		p.join()