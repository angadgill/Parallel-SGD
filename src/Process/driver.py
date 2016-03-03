from parallel import ParallelSGDRegressor
import numpy as np

if __name__ == '__main__':

	X = np.matrix('1.0 2.0 3.0; 4.0 5 6;7.0 8 9')
	y = np.array([1,2,3], dtype='float64')

	psgd = ParallelSGDRegressor()
	psgd.fit(X.astype('float64'), y)