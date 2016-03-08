from parallel import ParallelSGDRegressor
from sklearn import datasets
import numpy as np

if __name__ == '__main__':

	iris = datasets.load_iris()
	#print(iris.data)
	X = iris.data[:,0:3]
	y = iris.data[:,3]
	psgd = ParallelSGDRegressor()
	psgd.fit(X.astype('float64'), y)
