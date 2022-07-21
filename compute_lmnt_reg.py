from csc_mult import *
from numba import njit, objmode
from numba.types import int32, int64, double
from numba.typed import List

@njit
def get_squared(X, rsvr_size, squarenodes, dim = 0):
    X_aug = np.copy(X)
    if not squarenodes:
        return X_aug
    else:
        X_out = np.vstack((X_aug[0].reshape(1,-1), X_aug[1:rsvr_size+1], X_aug[1:rsvr_size+1]**2.0, X_aug[rsvr_size+1:]))
        return X_out

@njit
def compute_lmnt_reg(X,D,u_arr_train,k,reg_train_time,\
    Win_data, Win_indices, Win_indptr, Win_shape,\
    W_data, W_indices, W_indptr, W_shape, leakage, squarenodes):
    # This function computes the LMNT regularization for a reservoir computer
    # which has an input pass through and constant bias in the output layer.
    # Inputs:
    #   X - rsvr_size x training_length array of reservoir node states during training
    #
    #   D - rsvr_size x training_length of reservoir node state derivatives during training
    #       For the activation function tanh(...), this is sech^2(...)
    #
    #   u_arr_train - num_inputs x training_length array of reservoir inputs
    #
    #   k - number of steps of noise addition to approximateQ
    #
    #   reg_train_time - training time for the LMNT regularization
    #
    #   Win_data, Win_indices, Win_indptr, Win_shape - CSC sparse matrix format arrays for
    #                                                  the input matrix. See csc_mult for
    #                                                  array types.
    #
    #   W_data, W_indices, W_indptr, W_shape - CSC sparse matrix format arrays for
    #                                          the adjacency matrix. See csc_mult for
    #                                          array types.
    #
    #   leakage - reservoir leakage
    #
    #   squarenodes - Boolean indicating whether or not squared node states should be added
    #                  to the feature vector
    #
    # Outputs:
    #   states_trstates - information matrix for reservoir feature vectors
    #
    #   X_train - reservoir feature vectors
    #
    #   gradient_reg - LMNT regularization matrix, scaled to have a similar magnitude to
    #                  the information matrix.
    #Linearized k-step noise
    reg_train_frac = 1.0/(reg_train_time-(k-1))
    sparse_cutoff = 0.89
    break_flag = False
    rsvr_size, d = X.shape
    n = u_arr_train.shape[0]
    if squarenodes:
        res_feature_size = 2*rsvr_size
    else:
        rs_feature_size = rsvr_size
    X_train = np.concatenate(
       (np.ones((1, d)), X, u_arr_train), axis=0)
    X_train = get_squared(X_train, rsvr_size, squarenodes)
    gradient_reg_base = np.zeros((res_feature_size+n+1, res_feature_size+n+1))
    Win_nobias_data, Win_nobias_indices, Win_nobias_indptr, Win_nobias_shape = \
       get_Win_nobias(Win_data, Win_indices, Win_indptr, Win_shape)
    D_n_datas = List()
    D_n_indices = List()
    D_n_indptrs = List()
    D_n_shape = np.array([res_feature_size+n+1, n])
    E_n_datas = List()
    E_n_indices = List()
    E_n_indptrs = List()
    E_n_shape = np.array([res_feature_size+n+1, res_feature_size+n+1])
    reg_comp_datas   = List()
    reg_comp_indices = List()
    reg_comp_indptrs = List()
    reg_comp_shape   = np.array([res_feature_size+n+1, n])
    W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape = construct_jac_mat_csc_csc(
       Win_data, Win_indices, Win_indptr, Win_shape, W_data, W_indices, W_indptr, W_shape, rsvr_size, n, squarenodes)
    leakage_data, leakage_indices, leakage_indptr, leakage_shape = construct_leakage_mat(rsvr_size, n, leakage, squarenodes)
    reg_sum_avg_runtime = 0.
    E_n_avg_runtime = 0.
    reg_mult_avg_runtime = 0.
    D_n_avg_runtime = 0.

    for i in range(k):
       D_n_data, D_n_idx, D_n_indptr = get_D_n(D[:,i], X[:,i], Win_nobias_data, Win_nobias_indices,\
           Win_nobias_indptr, Win_nobias_shape, D_n_shape, rsvr_size, res_feature_size, n, squarenodes)
       D_n_datas.append(np.ascontiguousarray(D_n_data))
       D_n_indices.append(np.ascontiguousarray(D_n_idx))
       D_n_indptrs.append(np.ascontiguousarray(D_n_indptr))
    if k > 1:
       for i in range(1, k):
           E_n_data, E_n_idx, E_n_indptr = get_E_n(D[:,i], X[:,i], E_n_shape, rsvr_size, W_mat_data,\
               W_mat_indices, W_mat_indptr, W_mat_shape, leakage_data, leakage_indices, \
               leakage_indptr, leakage_shape, squarenodes)
           E_n_datas.append(np.ascontiguousarray(E_n_data))
           E_n_indices.append(np.ascontiguousarray(E_n_idx))
           E_n_indptrs.append(np.ascontiguousarray(E_n_indptr))

    for i in range(k-1):
       reg_comp_data, reg_comp_idx, reg_comp_indptr =\
           np.copy(D_n_datas[i]), np.copy(D_n_indices[i]), np.copy(D_n_indptrs[i])
       if k > 1:
           for j in range(i, k-1):
               reg_comp_data, reg_comp_idx, reg_comp_indptr, tmp = matrix_sparse_sparse_mult(\
                   E_n_datas[j], E_n_indices[j], E_n_indptrs[j], E_n_shape,\
                   reg_comp_data, reg_comp_idx, reg_comp_indptr, reg_comp_shape)
       reg_comp_datas.append(np.ascontiguousarray(reg_comp_data))
       reg_comp_indices.append(np.ascontiguousarray(reg_comp_idx))
       reg_comp_indptrs.append(np.ascontiguousarray(reg_comp_indptr))
    reg_comp_datas.append(np.ascontiguousarray(D_n_datas[-1]))
    reg_comp_indices.append(np.ascontiguousarray(D_n_indices[-1]))
    reg_comp_indptrs.append(np.ascontiguousarray(D_n_indptrs[-1]))
    sparsity = np.array([reg_comp_datas[j].size/(reg_comp_shape[0]*reg_comp_shape[1]) for j in range(k)])

    for i in range(k, X.shape[1]):
       for j in range(k):
           if sparsity[j] < sparse_cutoff:
               gradient_reg_base += matrix_sparse_sparseT_conv_mult(\
                   reg_comp_datas[j], reg_comp_indices[j], reg_comp_indptrs[j], reg_comp_shape)
           else:
               gradient_reg_base += matrix_sparse_sparseT_mult(reg_comp_datas[j], reg_comp_indices[j],\
                   reg_comp_indptrs[j], reg_comp_shape)
       if i == reg_train_time:
           gradient_reg = gradient_reg_base*reg_train_frac
           break_flag = True
           break
       if k > 1:
           E_n_data, E_n_idx, E_n_indptr = get_E_n(D[:,i], X[:,i], \
               E_n_shape, rsvr_size, W_mat_data, W_mat_indices, W_mat_indptr, W_mat_shape,\
               leakage_data, leakage_indices, leakage_indptr, leakage_shape,\
               squarenodes)
           E_n_datas[k-2] = np.ascontiguousarray(E_n_data)
           E_n_indices[k-2] = np.ascontiguousarray(E_n_idx)
           E_n_indptrs[k-2] = np.ascontiguousarray(E_n_indptr)
       for j in range(k-1):
           if k > 1:
               if sparsity[j+1] < sparse_cutoff:
                   reg_comp_data, reg_comp_idx, reg_comp_indptr, tmp = matrix_sparse_sparse_conv_mult(\
                       E_n_datas[k-2], E_n_indices[k-2], E_n_indptrs[k-2], E_n_shape,\
                       reg_comp_datas[j+1], reg_comp_indices[j+1], reg_comp_indptrs[j+1], reg_comp_shape)
               else:
                   reg_comp_data, reg_comp_idx, reg_comp_indptr, tmp = matrix_sparse_sparse_mult(\
                       E_n_datas[k-2], E_n_indices[k-2], E_n_indptrs[k-2], E_n_shape,\
                       reg_comp_datas[j+1], reg_comp_indices[j+1], reg_comp_indptrs[j+1], reg_comp_shape)
               reg_comp_datas[j], reg_comp_indices[j], reg_comp_indptrs[j] = \
                   np.ascontiguousarray(reg_comp_data), np.ascontiguousarray(reg_comp_idx),\
                   np.ascontiguousarray(reg_comp_indptr)
       reg_comp_data, reg_comp_idx, reg_comp_indptr = get_D_n(D[:,i], X[:,i], \
           Win_nobias_data, Win_nobias_indices, Win_nobias_indptr, Win_nobias_shape, D_n_shape, \
           rsvr_size, res_feature_size, n, squarenodes)
       reg_comp_datas[k-1], reg_comp_indices[k-1], reg_comp_indptrs[k-1] = \
           np.ascontiguousarray(reg_comp_data), np.ascontiguousarray(reg_comp_idx),\
           np.ascontiguousarray(reg_comp_indptr)
    if not break_flag:
       for j in range(k):
           if sparsity[j] < sparse_cutoff:
               gradient_reg_base += matrix_sparse_sparseT_conv_mult(\
                   reg_comp_datas[j], reg_comp_indices[j], reg_comp_indptrs[j], reg_comp_shape)
           else:
               gradient_reg_base += matrix_sparse_sparseT_mult(reg_comp_datas[j], reg_comp_indices[j],\
                   reg_comp_indptrs[j], reg_comp_shape)
       if i+1 == reg_train_time:
           gradient_reg = gradient_reg_base*reg_train_frac
    states_trstates = (X_train @ X_train.T)/d
    return states_trstates, X_train, gradient_reg
