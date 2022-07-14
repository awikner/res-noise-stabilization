import numpy as np
<<<<<<< HEAD
from scipy.sparse import csc_matrix, diags, csr_matrix
=======
from scipy.sparse import csc_matrix, diags, csr_matrix, coo_matrix
>>>>>>> dev
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
        out = diags(dmat, format = 'csc').dot(b)
    return out

@jit(nopython = True, fastmath = True)
def matrix_sparse_mult(sp, mat):
    with objmode(out = 'double[:,:]'):
        out = csc_matrix(sp).dot(mat)
    return out

@jit(nopython = True, fastmath = True)
def matrix_diag_sparse_mult_add(diag_mat, data, indices, indptr, shape, add_mat):
    with objmode(out = 'double[:,:]'):
        out_sp = diags(diag_mat, format = 'csc').dot(csc_matrix((data, indices, indptr), shape = (shape[0], shape[1]))) + csc_matrix(add_mat)
        out    = out_sp.toarray()
    return out

@jit(nopython = True, fastmath = True)
def matrix_diag_sparse_mult_add_sparse(diag_mat, data, indices, indptr, shape, add_mat_data, add_mat_indices,\
    add_mat_indptr, add_mat_shape):
    with objmode(data = 'double[:]', indices = 'int32[:]', indptr = 'int32[:]', shape = 'int32[:]'):
        out = diags(diag_mat, format = 'csc').dot(csc_matrix((data, indices, indptr), shape = (shape[0], shape[1]))) + csc_matrix((add_mat_data, add_mat_indices, add_mat_indptr), shape = (shape[0], shape[1]))
        data = out.data
        indices = out.indices
        indptr = out.indptr
        shape = np.array(list(out.shape), dtype = np.int32)

    return data, indices, indptr, shape

@jit(nopython = True, fastmath = True)
def matrix_diag_sparse_diag_mult(dmat, b_data, b_indices, b_indptr, b_shape, dmat_inv):
    with objmode(data = 'double[:]', indices = 'int32[:]', indptr = 'int32[:]', shape = 'int32[:]'):
        out = diags(dmat, format = 'csc').dot(csc_matrix((b_data, b_indices, b_indptr), shape = (b_shape[0], b_shape[1]))).dot(diags(dmag_inv, format = 'csc'))
        data = out.data
        indices = out.indices
        indptr = out.indptr
        shape = np.array(list(out.shape), dtype = np.int32)

    return data, indices, indptr, shape

@jit(nopython = True, fastmath = True)
def matrix_diag_sparse_mult(dmat, b_data, b_indices, b_indptr, b_shape):
    with objmode(data = 'double[:]', indices = 'int32[:]', indptr = 'int32[:]', shape = 'int32[:]'):
        out = diags(dmat, format = 'csc').dot(csc_matrix((b_data, b_indices, b_indptr), shape = (b_shape[0], b_shape[1])))
        data = out.data
        indices = out.indices
        indptr = out.indptr
        shape = np.array(list(out.shape), dtype = np.int32)

    return data, indices, indptr, shape

@jit(nopython = True, fastmath = True)
def dense_to_sparse(W):
    with objmode(data = 'double[:]', indices = 'int32[:]', indptr = 'int32[:]', shape = 'int32[:]'):
        out = csc_matrix(W)
        data = out.data
        indices = out.indices
        indptr = out.indptr
        shape = np.array(list(out.shape), dtype = np.int32)

    return data, indices, indptr, shape


@jit(nopython = True, fastmath = True)
def mult_vec(data, indices, indptr, shape, mat):
    assert(shape[1] == mat.size)
    out = np.zeros(shape[0])
    for i in range(mat.size):
        for k in range(indptr[i], indptr[i+1]):
            out[indices[k]] += data[k] * mat[i]
    return out

@jit(nopython = True, fastmath = True)
def construct_jac_mat_csc(Win, data_in, indices_in, indptr_in, shape_in, rsvr_size, n, squarenodes = False):
    with objmode(data = 'double[:]', indices = 'int32[:]', indptr = 'int32[:]', shape = 'int32[:]'):
        W_conv = csc_matrix((data_in, indices_in, indptr_in), shape = (shape_in[0], shape_in[1])).toarray()
        if squarenodes:
            mat    = csc_matrix(np.concatenate((Win[:,0].reshape(-1,1), W_conv, np.zeros((rsvr_size,n+rsvr_size))), axis = 1))
        else:
            mat    = csc_matrix(np.concatenate((Win[:,0].reshape(-1,1), W_conv, np.zeros((rsvr_size,n))), axis = 1))
        data   = mat.data
        indices = mat.indices
        indptr = mat.indptr
        shape = np.array(list(mat.shape), dtype = np.int32)
    return data, indices, indptr, shape

@jit(nopython = True, fastmath = True)
def construct_jac_mat_csc_csc(Win_data, Win_indices, Win_indptr, Win_shape,\
        data_in, indices_in, indptr_in, shape_in, rsvr_size,  n, squarenodes = False):
    with objmode(data = 'double[:]', indices = 'int32[:]', indptr = 'int32[:]', shape = 'int32[:]'):
        W_conv = csc_matrix((data_in, indices_in, indptr_in), shape = (shape_in[0],  shape_in[1])).toarray()
        Win_conv = csc_matrix((Win_data, Win_indices, Win_indptr), shape = (Win_shape[0],  Win_shape[1])).toarray()
        if squarenodes:
            mat    = csc_matrix(np.concatenate((Win_conv[:,0].reshape(-1,1), W_conv, np.zeros((rsvr_size,n+rsvr_size))), axis = 1))
        else:
            mat    = csc_matrix(np.concatenate((Win_conv[:,0].reshape(-1,1), W_conv, np.zeros((rsvr_size,n))), axis = 1))
        data   = mat.data
        indices = mat.indices
        indptr = mat.indptr
        shape = np.array(list(mat.shape), dtype = np.int32)
    return data, indices, indptr, shape

@jit(nopython = True, fastmath = True)
def construct_leakage_mat(rsvr_size, n, leakage, squarenodes):
    if squarenodes:
        leakage_mat = np.concatenate((np.zeros((rsvr_size,1)), (1-leakage) *
                            np.identity(rsvr_size), np.zeros((rsvr_size, n+rsvr_size))), axis=1)
    else:
        leakage_mat = np.concatenate((np.zeros((rsvr_size,1)), (1-leakage) *
                             np.identity(rsvr_size), np.zeros((rsvr_size, n))),   axis=1)
    with objmode(data = 'double[:]', indices = 'int32[:]', indptr = 'int32[:]', shape = 'int32[:]'):
        mat = csc_matrix(leakage_mat)
        data   = mat.data
        indices = mat.indices
        indptr = mat.indptr
        shape = np.array(list(mat.shape), dtype = np.int32)

    return data, indices, indptr, shape

@jit(nopython = True, fastmath = True)
def construct_leakage_mat_mlonly(rsvr_size, n, leakage, squarenodes):
    leakage_mat = (1-leakage) * np.identity(rsvr_size)
    with objmode(data = 'double[:]', indices = 'int32[:]', indptr = 'int32[:]', shape = 'int32[:]'):
        mat = csc_matrix(leakage_mat)
        data   = mat.data
        indices = mat.indices
        indptr = mat.indptr
        shape = np.array(list(mat.shape), dtype = np.int32)

    return data, indices, indptr, shape

@jit(nopython = True, fastmath = True)
def get_Win_nobias(data_in, indices_in, indptr_in, shape_in):
    with objmode(data = 'double[:]', indices = 'int32[:]', indptr = 'int32[:]', shape = 'int32[:]'):
        Win = csc_matrix((data_in, indices_in, indptr_in), shape = (shape_in[0],  shape_in[1])).toarray()
        Win_nobias = csc_matrix(np.ascontiguousarray(Win[:, 1:]))
        data = Win_nobias.data
        indices = Win_nobias.indices
        indptr = Win_nobias.indptr
        shape = np.array(list(Win_nobias.shape), dtype = np.int32)
    return data, indices, indptr, shape


@jit(nopython = True, fastmath = True)
def coo_to_csc(data_in, row_in, col_in, shape_in):
    with objmode(data = 'double[:]', indices = 'int32[:]', indptr = 'int32[:]'):
        mat     = coo_matrix((data_in, (row_in, col_in)), shape = (shape_in[0], shape_in[1])).tocsc()
        data    = mat.data
        indices = mat.indices
        indptr  = mat.indptr
    return data, indices, indptr

@jit(nopython = True, fastmath = True)
def csc_to_coo(data_in, indices_in, indptr_in, shape_in):
    with objmode(data = 'double[:]', rows = 'int32[:]', cols = 'int32[:]'):
        mat    = csc_matrix((data_in, indices_in, indptr_in), shape = (shape_in[0], shape_in[1])).tocoo()
        data   = mat.data
        rows   = mat.row
        cols   = mat.col
    return data, rows, cols

@jit(nopython = True, fastmath = True)
def matrix_diag_sparse_mult_sparse_add(diag_mat, data, indices, indptr, shape, \
        add_mat_data, add_mat_indices, add_mat_indptr, add_mat_shape):
    with objmode(data = 'double[:]', indices = 'int32[:]', indptr = 'int32[:]', shape = 'int32[:]'):
        mat = diags(diag_mat, format = "csc").dot(csc_matrix((data, indices, indptr), shape = (shape[0], shape[1])))\
            + csc_matrix((add_mat_data, add_mat_indices, add_mat_indptr), shape = (add_mat_shape[0], add_mat_shape[1]))
        data   = mat.data
        indices = mat.indices
        indptr = mat.indptr
        shape = np.array(list(mat.shape), dtype = np.int32)
    return data, indices, indptr, shape

@jit(nopython = True, fastmath = True)
def matrix_sparse_sparseT_mult(mat1_data, mat1_indices, mat1_indptr, mat1_shape):
    with objmode(out = 'double[:,:]'):
        mat_sp = csc_matrix((mat1_data, mat1_indices, mat1_indptr), shape = (mat1_shape[0], mat1_shape[1]))
        mat_sp2 = mat_sp @ mat_sp.T
        out = mat_sp2.toarray()
    return out

@jit(nopython = True, fastmath = True)
def matrix_sparse_sparse_mult(mat1_data, mat1_indices, mat1_indptr, mat1_shape,\
                              mat2_data, mat2_indices, mat2_indptr, mat2_shape):
    with objmode(data = 'double[:]', indices = 'int32[:]', indptr = 'int32[:]', shape = 'int32[:]'):
        mat = csc_matrix((mat1_data, mat1_indices, mat1_indptr), shape = (mat1_shape[0], mat1_shape[1])) @ \
              csc_matrix((mat2_data, mat2_indices, mat2_indptr), shape = (mat2_shape[0], mat2_shape[1]))
        data     = mat.data
        indices  = mat.indices
        indptr   = mat.indptr
        shape    = np.array(list(mat.shape), dtype = np.int32)
    return data, indices, indptr, shape

@jit(nopython = True, fastmath = True)
def matrix_sparse_sparse_conv_mult(mat1_data, mat1_indices, mat1_indptr, mat1_shape,\
                              mat2_data, mat2_indices, mat2_indptr, mat2_shape):
    with objmode(data = 'double[:]', indices = 'int32[:]', indptr = 'int32[:]', shape = 'int32[:]'):
        mat = csc_matrix(csc_matrix((mat1_data, mat1_indices, mat1_indptr), shape = (mat1_shape[0], mat1_shape[1])) @ \
              csc_matrix((mat2_data, mat2_indices, mat2_indptr), shape = (mat2_shape[0], mat2_shape[1])).toarray())
        data     = mat.data
        indices  = mat.indices
        indptr   = mat.indptr
        shape    = np.array(list(mat.shape), dtype = np.int32)
    return data, indices, indptr, shape

@jit(nopython = True, fastmath = True)
def matrix_sparse_sparseT_conv_mult(mat1_data, mat1_indices, mat1_indptr, mat1_shape):
    with objmode(out = 'double[:,:]'):
        mat = csc_matrix((mat1_data, mat1_indices, mat1_indptr), shape = (mat1_shape[0], mat1_shape[1])).toarray()
        out = mat @ mat.T
    return out

@jit(fastmath = True, nopython = True)
def get_D_n(D, X, Win_nobias_data, Win_nobias_indices, Win_nobias_indptr, Win_nobias_shape, \
        D_n_shape, rsvr_size, res_feature_size, n, squarenodes):
    D_n_base_data, D_n_base_indices, D_n_base_indptr, D_n_base_shape = matrix_diag_sparse_mult(D, \
        Win_nobias_data, Win_nobias_indices, Win_nobias_indptr, Win_nobias_shape)
    D_n_data_coo, D_n_rows, D_n_cols = csc_to_coo(\
        D_n_base_data, D_n_base_indices, D_n_base_indptr, D_n_base_shape)
    D_n_rows += 1
    if squarenodes:
        D_n_2_data, D_n_2_indices, D_n_2_indptr, D_n_2_shape = matrix_diag_sparse_mult(2*X, D_n_base_data,\
            D_n_base_indices, D_n_base_indptr, D_n_base_shape)
        #D_n[1+rsvr_size:1+res_feature_size,:,i] = matrix_diag_mult(2*X[:, i], D_n_base)
        D_n_2_data_coo, D_n_2_rows, D_n_2_cols = csc_to_coo(\
            D_n_2_data, D_n_2_indices, D_n_2_indptr, D_n_2_shape)
        D_n_2_rows += 1+rsvr_size
        D_n_data_coo, D_n_rows, D_n_cols = np.concatenate((D_n_data_coo, D_n_2_data_coo, np.ones(n))),\
                np.concatenate((D_n_rows, D_n_2_rows, np.arange(res_feature_size+1,D_n_shape[0]))),\
                np.concatenate((D_n_cols, D_n_2_cols, np.arange(D_n_shape[1])))
    else:
        #D_n[1:rsvr_size+1, :, i] = D_n_base
        D_n_data_coo, D_n_rows, D_n_cols = np.append(D_n_data_coo, np.ones(n)),\
                np.append(D_n_rows, np.arange(res_feature_size+1,D_n_shape[0])),\
                np.append(D_n_cols, np.arange(D_n_shape[1]))
    D_n_data, D_n_indices, D_n_indptr = coo_to_csc(D_n_data_coo, D_n_rows, D_n_cols, D_n_shape)
        #D_n[res_feature_size+1:, :, i] = np.identity(n)
    return D_n_data, D_n_indices, D_n_indptr

@jit(fastmath = True, nopython = True)
def get_D_n_mlonly(D, Win_nobias_data, Win_nobias_indices, Win_nobias_indptr, Win_nobias_shape, \
        D_n_shape, rsvr_size, res_feature_size, n, squarenodes):
    D_n_data, D_n_indices, D_n_indptr, tmp = matrix_diag_sparse_mult(D, Win_nobias_data, Win_nobias_indices,\
        Win_nobias_indptr, Win_nobias_shape)
    return D_n_data,D_n_indices,D_n_indptr

@jit(fastmath = True, nopython = True)
def get_E_n(D, X, E_n_shape, rsvr_size, W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape, \
            leakage_mat_data, leakage_mat_indices, leakage_mat_indptr, leakage_mat_shape, squarenodes):
    E_n_base_data, E_n_base_indices, E_n_base_indptr, E_n_base_shape = matrix_diag_sparse_mult_sparse_add(
        D, W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape, leakage_mat_data, \
        leakage_mat_indices, leakage_mat_indptr, leakage_mat_shape)
    E_n_base_coo, E_n_base_rows, E_n_base_cols = csc_to_coo(E_n_base_data,\
        E_n_base_indices, E_n_base_indptr, E_n_base_shape)
    E_n_base_rows += 1
    if squarenodes:
        E_n_2_coo, E_n_2_rows, E_n_2_cols = csc_to_coo(*matrix_diag_sparse_mult(2*X,\
            E_n_base_data, E_n_base_indices, E_n_base_indptr, E_n_base_shape))
        E_n_2_rows += 1+rsvr_size
        #E_n[1+rsvr_size:1+res_feature_size,:,i-1] = matrix_diag_mult(2*X[:,  i], E_n_base)
        E_n_data_coo, E_n_rows, E_n_cols = np.append(E_n_base_coo, E_n_2_coo),\
            np.append(E_n_base_rows, E_n_2_rows), np.append(E_n_base_cols, E_n_2_cols)
    else:
        E_n_data_coo, E_n_rows, E_n_cols = E_n_base_coo, E_n_base_rows, E_n_base_cols
    E_n_data, E_n_indices, E_n_indptr = coo_to_csc(E_n_data_coo, E_n_rows, E_n_cols, E_n_shape)
    return E_n_data, E_n_indices, E_n_indptr

@jit(fastmath = True, nopython = True)
def get_E_n_mlonly(D, E_n_shape, rsvr_size, W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape, \
            leakage_mat_data, leakage_mat_indices, leakage_mat_indptr, leakage_mat_shape, squarenodes):
    E_n_data, E_n_indices, E_n_indptr, tmp = matrix_diag_sparse_mult_add_sparse(D, W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape, leakage_mat_data, leakage_mat_indices, leakage_mat_indptr, leakage_mat_shape)
    return E_n_data, E_n_indices, E_n_indptr
