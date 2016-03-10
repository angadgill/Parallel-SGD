from parallel import ParallelSGDRegressor
from sklearn.linear_model import SGDRegressor
from sklearn import datasets
from sklearn.datasets import make_regression
import numpy as np

if __name__ == '__main__':


	n_samples =100000
	n_features=100
	seed = 1
	effective_rank =100
	X, y = make_regression(n_samples=n_samples, n_features=n_features,
                       random_state=seed, noise=0.1, effective_rank=effective_rank)

	#iris = datasets.load_iris()
	#print(iris.data)
	#X = iris.data[:,0:3]
	#y = iris.data[:,3]

	sgd = SGDRegressor()
	sgd.fit(X.astype('float64'),y)
	print(sgd.coef_,sgd.intercept_)

	psgd = ParallelSGDRegressor()
	psgd.fit(X.astype('float64'), y)
	print(psgd.coef_,psgd.intercept_)
