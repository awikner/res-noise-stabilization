import numpy as np
import ray
from ks_etdrk4 import *
from poincare_max import *
import os
import sys
import time
from numba import jit, objmode
from wasserstein_distance_empirical import *

@jit(nopython = True, fastmath = True)
def sum_numba_axis0(mat):
    res = np.zeros(mat.shape[1])
    for i in range(mat.shape[1]):
        res[i] = np.sum(mat[:,i])
    return res

@jit(nopython = True, fastmath = True)
def compute_onestep_mss(base_true, d):
    N = base_true.shape[0]
    ks_onestep = np.zeros((N, base_true.shape[1]-1))
    for j in range(ks_onestep.shape[1]):
        ks_onestep[:,j] = kursiv_predict(base_true[:,j], T = 1, d = d, N = N)[0][:,-1]
    sum_square = sum_numba_axis0((ks_onestep - base_true[:,1:])**2.0)/np.sqrt(2.0)
    mean_sum_square = np.mean(sum_square)
    return mean_sum_square

@ray.remote
def ks_poincare_map(base_true, true_pmap_flat, u0, T, d, N, idxs, transient = 1000):
    tic = time.perf_counter()
    ks_true, tmp = kursiv_predict(u0, T = T, d = d, N = N)
    ks_true = np.ascontiguousarray(ks_true[:,transient:])
    variance = np.var(ks_true.flatten())
    poincare_map = poincare_max(ks_true, idxs)
    for i, idx in enumerate(idxs):
        if i == 0:
            poincare_map_flat = poincare_map[i]
        else:
            poincare_map_flat = np.append(poincare_map_flat, poincare_map[i])
    toc = time.perf_counter()
    print('Poincare map found in %f sec.' % (toc - tic))
    max_map_len = max([len(i) for i in poincare_map])
    wass_dist   = wasserstein_distance_empirical(ks_true.flatten(), base_true.flatten())
    toc2 = time.perf_counter()
    print('Wasserstein distance found in %f sec.' % (toc2 - toc))
    true_pmap_flat_copy = np.copy(true_pmap_flat)
    true_pmap_flat_copy.setflags(write=True)
    pmap_wass_dist = wasserstein_distance_empirical(poincare_map_flat, true_pmap_flat_copy)
    toc3 = time.perf_counter()
    print('Pmap wasserstein distance found in %f sec.' % (toc3 - toc2))
    mean_sum_square = compute_onestep_mss(base_true, d)
    toc4 = time.perf_counter()
    print('Onestep error found in %f sec.' % (toc4 - toc3))
    print('Total runtime: %s sec.' % (toc4 - tic))
    return poincare_map, max_map_len, wass_dist, pmap_wass_dist, mean_sum_square, variance

def main(argv):
    end = int(argv[0])
    print(end)
    ray.init(address=os.environ["ip_head"])
    d_vals = np.linspace(20, 24, 401)[end-80:end]
    base_d = 22.0
    N = 64
    T = 100000
    transient = 1000
    idxs = np.arange(N)
    seed = 10
    np.random.seed(seed)
    u0 = 0.6*(np.random.rand(N)*2-1)
    base_true, tmp = kursiv_predict(u0, T = T, d = base_d, N = N)
    base_true = np.ascontiguousarray(base_true[:,transient:])
    true_pmap = poincare_max(base_true, idxs)
    true_pmap_flat = np.array([])
    for i, idx in enumerate(idxs):
        true_pmap_flat = np.append(true_pmap_flat, true_pmap[i])
    out_base  = ray.get([ks_poincare_map.remote(base_true, true_pmap_flat, u0, T, d, N, idxs, transient) for d in d_vals])
    max_map_len = max([elem[1] for elem in out_base])
    wass_dist   = np.array([elem[2] for elem in out_base])
    pmap_wass_dist = np.array([elem[3] for elem in out_base])
    mean_sum_squared = np.array([elem[4] for elem in out_base])
    variances = np.array([elem[5] for elem in out_base])
    print('Maximum map length: %d' % max_map_len)
    foldername = '/lustre/awikner1/res-noise-stabilization/'
    top_folder = 'KS_poincare_maps/'
    if not os.path.isdir(os.path.join(foldername, top_folder)):
        os.mkdir(os.path.join(foldername, top_folder))
    np.savetxt(foldername + top_folder + 'wass_dist_dmin%0.4f_dmax%0.4f.csv' % (np.min(d_vals), np.max(d_vals)),\
            wass_dist, delimiter = ',')
    np.savetxt(foldername + top_folder + 'pmap_max_wass_dist_dmin%0.4f_dmax%0.4f.csv' % (np.min(d_vals), np.max(d_vals)),\
            pmap_wass_dist, delimiter = ',')
    np.savetxt(foldername + top_folder + 'mean_sum_squared_dmin%0.4f_dmax%0.4f.csv' % (np.min(d_vals), np.max(d_vals)),\
            mean_sum_squared, delimiter = ',')
    np.savetxt(foldername + top_folder + 'variances_dmin%0.4f_dmax%0.4f.csv' % (np.min(d_vals), np.max(d_vals)),\
            variances, delimiter = ',')
    for j, d in enumerate(d_vals):
        filename = 'poincare_map_d%0.4f.csv' % d
        poincare_map = np.empty((idxs.size, max_map_len))
        poincare_map[:] = np.nan
        for i, idx in enumerate(idxs):
            map_len = len(out_base[j][0][i])
            poincare_map[i,:map_len] = np.array(out_base[j][0][i])
        np.savetxt(foldername + top_folder + filename, poincare_map, delimiter = ',')
    print('Finished computation!')




if __name__ == "__main__":
    main(sys.argv[1:])
