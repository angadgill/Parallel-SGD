from parallel_sgd_modified import ParallelSGDRegressor
import numpy as np

if __name__ == '__main__':

	X = np.matrix('1.0 2.0 3.0; 4.0 5 6;7.0 8 9')
	y = np.array([1,2,3], dtype='float64')
	alpha=0.01
	C=0.0
	est =0.01
	learning_rate ='constant'
	n_iter=10
	sample_weight =np.array([1,2,3], dtype='float64')
	loss='squared_loss'
	psgd = ParallelSGDRegressor()
	
	psgd._fit_regressor(X.astype('float64'), y, alpha, C,loss, learning_rate,sample_weight,n_iter)