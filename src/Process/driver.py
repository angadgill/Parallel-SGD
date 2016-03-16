from parallel import ParallelSGDRegressor
from sklearn.linear_model import SGDRegressor
from sklearn import datasets
from sklearn.datasets import make_regression
from sklearn.cross_validation import ShuffleSplit
import numpy as np
import time

if __name__ == '__main__':


	n_samples =100
	n_features=100
	seed = 1
	effective_rank =100
	X, y = make_regression(n_samples=n_samples, n_features=n_features,
                       random_state=seed, noise=0.1, effective_rank=effective_rank)

	for train, test in ShuffleSplit(n=n_samples, n_iter=1, test_size=0.2):
		pass

	X_train = X[train]
	X_test = X[test]
	y_train = y[train]
	y_test = y[test]

	#iris = datasets.load_iris()
	#print(iris.data)
	#X = iris.data[:,0:3]
	#y = iris.data[:,3]

	start_time = time.time()
	sgd = SGDRegressor(alpha=0.01, average=False, epsilon=0.1, eta0=0.01,
             fit_intercept=True, l1_ratio=0.15, learning_rate='invscaling',
             loss='squared_loss', n_iter=1000, penalty='l2', power_t=0.25,
             random_state=None, shuffle=True, verbose=0, warm_start=False)
	sgd.fit(X_train.astype('float64'),y_train)
	elapsed_time = time.time() - start_time
	print("Time %s"%elapsed_time)
	print(sgd.coef_,sgd.intercept_)
	print("Accuracy %s"%sgd.score(X_test,y_test))

	start_time = time.time()
	psgd = ParallelSGDRegressor(alpha=0.01, average=False, epsilon=0.1, eta0=0.01,
             fit_intercept=True, l1_ratio=0.15, learning_rate='invscaling',
             loss='squared_loss', n_iter=1000, penalty='l2', power_t=0.25,
             random_state=None, shuffle=True, verbose=0, warm_start=False)
	psgd.fit(X_train.astype('float64'), y_train)
	elapsed_time = time.time() - start_time
	print("Time %s" %elapsed_time)
	print(psgd.coef_,psgd.intercept_)
	print("Accuracy %s"%psgd.score(X_test,y_test))
