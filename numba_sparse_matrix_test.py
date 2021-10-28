import numpy as np
from scipy import sparse
import time
from csr import CSR
from numba import njit, objmode

@njit
def CSR_matmult(a: CSR, b: np.ndarray):
    out = np.zeros((a.nrows, b.shape[1]))
    for i in range(out.shape[1]):
        out[:,i] = a.mult_vec(b[:,i])
    return out

@njit
def sp_matmul(a,b):
    with objmode(out = 'double[:,:]'):
        out = sparse.csr_matrix(a).dot(b)
    return out

np.random.seed(10)
avg_degree = 10
mat_size   = 1000
base_mat = np.zeros((mat_size,mat_size)).flatten()
random_indices = np.random.permutation(base_mat.size)[:avg_degree*mat_size]
base_mat[random_indices] = np.random.rand(avg_degree*mat_size)
base_mat = base_mat.reshape(mat_size,-1)

sc_csr_mat = sparse.csr_matrix(base_mat)
numba_csr_mat = CSR(sc_csr_mat.shape[0], sc_csr_mat.shape[1], sc_csr_mat.nnz, sc_csr_mat.indptr, sc_csr_mat.indices, sc_csr_mat.data)

test_mat = np.random.rand(1000)

tic = time.perf_counter()
out = base_mat @ test_mat
toc = time.perf_counter()
print(out.flatten()[-5:])
runtime = toc - tic
print('Dense Runtime: %f sec.' % runtime)

tic = time.perf_counter()
out = sc_csr_mat.dot(test_mat)
toc = time.perf_counter()
print(out.flatten()[-5:])
runtime = toc - tic
print('Scipy Sparse runtime: %f sec.' % runtime)

tic = time.perf_counter()
out = sparse.csr_matrix(base_mat).dot(test_mat)
toc = time.perf_counter()
print(out.flatten()[-5:])
runtime = toc - tic
print('Scipy Sparse runtime: %f sec.' % runtime)

tic = time.perf_counter()
out = numba_csr_mat.mult_vec(test_mat)
toc = time.perf_counter()
print(out.flatten()[-5:])
runtime = toc - tic
print('Numba Sparse runtime: %f sec.' % runtime)

tic = time.perf_counter()
out = numba_csr_mat.mult_vec(test_mat)
toc = time.perf_counter()
print(out.flatten()[-5:])
runtime = toc - tic
print('Numba Sparse runtime: %f sec.' % runtime)
