from parallel import ParallelSGDRegressor as psgd
from sklearn.linear_model import SGDRegressor as sgd
from sklearn import datasets
import numpy as np

if __name__ == '__main__':

	iris = datasets.load_iris()
	#print(iris.data)
	X = iris.data[:,0:3]
	y = iris.data[:,3]

	sgd.fit(X.astype('float64'),y)
	print(sgd.coef_,sgd.intercept_)


	psgd.fit(X.astype('float64'), y)
	print(sgd.coef_,sgd.intercept_)
