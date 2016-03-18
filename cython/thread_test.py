"""
Code credits: http://nealhughes.net/parallelcomp2/
"""

import time
from thread_demo import *
import numpy as np

X = -1 + 2*np.random.rand(9e7)

# start_time = time.time()
# a = numpy_array(X)
# print "Time numpy_array:", time.time() - start_time

# start_time = time.time()
# a = python_loop(X)
# print "Time python_loop", time.time() - start_time

start_time = time.time()
cython_loop(X)
print "Time cython_loop:", time.time() - start_time

num_procs = 4
t_iter = 5

for i in range(num_procs):
    start_time = time.time()
    for _ in range(t_iter):
        b = python_parallel(X, i+1)
    timei = (time.time() - start_time)/t_iter
    if i == 0:
        time1 = timei
    print "python_parallel with %d workers. Time: %f, Speedup: %f" % (i+1, timei, time1/timei)


for i in range(num_procs):
    start_time = time.time()
    for _ in range(t_iter):
        cython_parallel(X, i+1)
    timei = (time.time() - start_time)/t_iter
    if i == 0:
        time1 = timei
    print "cython_parallel with %d workers. Time: %f, Speedup: %f" % (i+1, timei, time1/timei)


# if sum(a-b) != 0:
#     print "Result incorrect"
