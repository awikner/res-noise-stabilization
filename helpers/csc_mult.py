import numpy as np
from scipy.sparse import csc_matrix, diags, csr_matrix
from numba import jit, objmode

#def create_csc_matrix(data, indices, indptr, shape):
#    return csc_matrix((data, indices, indptr), shape = (shape[0], shape[1]))

@jit(nopython = True, fastmath = True)
def matrix_dot_left_T(data, indices, indptr, shape, mat):
    with objmode(out = 'double[:,:]'):
        csc = csc_matrix((data, indices, indptr), shape = (shape[0], shape[1]))
        out = csc.T.dot(mat).T
    return out

@jit(nopython = True, fastmath = True)
def matrix_diag_mult(dmat, b):
    with objmode(out = 'double[:,:]'):
        out = diags(dmat).dot(b)
    return out

@jit(nopython = True, fastmath = True)
def matrix_sparse_mult(sp, mat):
    with objmode(out = 'double[:,:]'):
        out = csc_matrix(sp).dot(mat)
    return out

@jit(nopython = True, fastmath = True)
def matrix_diag_sparse_mult_add(diag_mat, data, indices, indptr, shape, add_mat):
    with objmode(out = 'double[:,:]'):
        out_sp = diags(diag_mat).dot(csc_matrix((data, indices, indptr), shape = (shape[0], shape[1]))) + csr_matrix(add_mat)
        out    = out_sp.toarray()
    return out

@jit(nopython = True, fastmath = True)
def matrix_diag_sparse_mult(dmat, b_data, b_indices, b_indptr, b_shape):
    with objmode(data = 'double[:]', indices = 'int32[:]', indptr = 'int32[:]', shape = 'int64[:]'):
        out = diags(a).dot(csc_matrix((b_data, b_indices, b_indptr), shape = (b_shape[0], b_shape[1]))).tocsc()
        data = out.data
        indices = out.indices
        indptr = out.indptr
        shape = np.array(list(out.shape))

    return data, indices, indptr, shape

@jit(nopython = True, fastmath = True)
def dense_to_sparse(W):
    with objmode(data = 'double[:]', indices = 'int32[:]', indptr = 'int32[:]', shape = 'int64[:]'):
        out = csc_matrix(W)
        data = out.data
        indices = out.indices
        indptr = out.indptr
        shape = np.array(list(out.shape))

    return data, indices, indptr, shape


@jit(nopython = True, fastmath = True)
def mult_vec(data, indices, indptr, shape, mat):
    out = np.zeros(shape[0])
    for i in range(mat.size):
        for k in range(indptr[i], indptr[i+1]):
            out[indices[k]] += data[k] * mat[i]
    return out

@jit(nopython = True, fastmath = True)
def construct_jac_mat_csc(Win, data_in, indices_in, indptr_in, shape_in, rsvr_size, n):
    with objmode(data = 'double[:]', indices = 'int32[:]', indptr = 'int32[:]', shape = 'int64[:]'):
        W_conv = csc_matrix((data_in, indices_in, indptr_in), shape = (shape_in[0], shape_in[1])).toarray()
        mat    = csc_matrix(np.concatenate((Win[:,0].reshape(-1,1), W_conv, np.zeros((rsvr_size,n))), axis = 1))
        data   = mat.data
        indices = mat.indices
        indptr = mat.indptr
        shape = np.array(list(mat.shape))
    return data, indices, indptr, shape
