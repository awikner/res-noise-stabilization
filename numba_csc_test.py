import numpy as np
import matplotlib.pyplot as plt
from ks_etdrk4 import *
from scipy.sparse import csr_matrix, csc_matrix
import csr
from numba.types import int32, float32, int64, double
from numba.experimental import jitclass
from numba import prange, objmode, njit
import time

spec = [('data', double[:]),
        ('indices', int32[:]),
        ('indptr', int32[:]),
        ('shape', int64[:])]

@jitclass(spec)
class csc_matrix_numba(object):
    def __init__(self, data, indices, indptr, shape):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = shape

    @property
    def size(self):
        return self.shape[0]*self.shape[1]

    def nnz(self):
        return self.data.size

    def matrix_dot_right(self, mat):
        with objmode(out = 'double[:,:]'):
            out = csc_matrix((self.data,self.indices,self.indptr),\
                 shape = (self.shape[0], self.shape[1])).dot(mat)
        return out

    def matrix_dot_left_T(self, mat):
        with objmode(out = 'double[:,:]'):
            out = csc_matrix((self.data,self.indices,self.indptr),\
                  shape = (self.shape[0], self.shape[1])).T.dot(mat).T
        return out

    def mult_vec(self, mat):
        out = np.zeros(self.shape[0])
        for i in range(mat.size):
            for k in range(self.indptr[i], self.indptr[i+1]):
                 out[self.indices[k]] += self.data[k] * mat[i]
        return out

@njit
def matrix_dot_right_csc(a, b):
    with objmode(out = 'double[:,:]'):
        out = csc_matrix((a.data,a.indices,a.indptr),\
            shape = (a.shape[0], a.shape[1])).dot(b)
    return out

np.random.seed(10)
mat_size = 2000
vals_per_row = 10
idxs = np.random.permutation(mat_size*mat_size)[:mat_size*(mat_size-vals_per_row)]
vals = np.random.rand(mat_size*mat_size)
vals[idxs] = 0
vals = vals.reshape(mat_size,-1)
print(vals)
vals_csc = csc_matrix(vals)
csc_nb = csc_matrix_numba(vals_csc.data, vals_csc.indices, vals_csc.indptr, np.array(list(vals_csc.shape)))
vals_dense = np.random.rand(mat_size,mat_size)


iters = 50
tic = time.perf_counter()
for i in range(iters):
    out = vals @ vals_dense
toc = time.perf_counter()
print('Dense Runtime: %f sec' % ((toc - tic)/iters))

tic = time.perf_counter()
for i in range(iters):
    out = vals_csc.dot(vals_dense)
toc = time.perf_counter()
print('Scipy Runtime: %f sec' % ((toc - tic)/iters))

tic = time.perf_counter()
for i in range(iters):
    out = csc_matrix((csc_nb.data,csc_nb.indices,csc_nb.indptr),\
        shape = (csc_nb.shape[0], csc_nb.shape[1])).dot(vals_dense)
toc = time.perf_counter()
print('Scipy w/ Conversion Runtime: %f sec' % ((toc - tic)/iters))

out = csc_nb.matrix_dot_right(vals_dense)
tic = time.perf_counter()
for i in range(iters):
    out = csc_nb.matrix_dot_right(vals_dense)
toc = time.perf_counter()
print('Numba w/ Scipy Conversion Runtime: %f sec' % ((toc - tic)/iters))

out = matrix_dot_right_csc(csc_nb, vals_dense)
tic = time.perf_counter()
for i in range(iters):
    out = matrix_dot_right_csc(csc_nb, vals_dense)
toc = time.perf_counter()
print('Numba w/ Scipy Conversion Runtime: %f sec' % ((toc - tic)/iters))
