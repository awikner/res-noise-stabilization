import numpy as np
from numba import jit, objmode

@jit(nopython = True, fastmath = True)
def wasserstein_distance_empirical(measured_samples, true_samples):
    if np.any(np.isnan(measured_samples)):
        return np.NAN
    if np.any(np.isinf(measured_samples)):
        return np.inf
    measured_samples.sort()
    true_samples.sort()
    n, m, n_inv, m_inv = (measured_samples.size, true_samples.size, 1/measured_samples.size, 1/true_samples.size)
    n_itr = 0; m_itr = 0; measured_cdf = 0; true_cdf = 0; wass_dist = 0
    if measured_samples[n_itr] < true_samples[m_itr]:
        prev_sample = measured_samples[n_itr]
        measured_cdf += n_inv
        n_itr +=1
    elif true_samples[m_itr] < measured_samples[n_itr]:
        prev_sample = true_samples[m_itr]
        true_cdf += m_inv
        m_itr += 1
    else:
        prev_sample = true_samples[m_itr]
        measured_cdf += n_inv; true_cdf += m_inv
        n_itr +=1; m_itr += 1
    while n_itr < n and m_itr < m:
        if measured_samples[n_itr] < true_samples[m_itr]:
            wass_dist += np.abs((measured_cdf - true_cdf)*(measured_samples[n_itr]-prev_sample))
            prev_sample = measured_samples[n_itr]
            measured_cdf += n_inv
            n_itr += 1
        elif true_samples[m_itr] < measured_samples[n_itr]:
            wass_dist += np.abs((measured_cdf - true_cdf)*(true_samples[m_itr]-prev_sample))
            prev_sample = true_samples[m_itr]
            true_cdf += m_inv
            m_itr += 1
        else:
            wass_dist += np.abs((measured_cdf - true_cdf)*(true_samples[m_itr]-prev_sample))
            prev_sample = true_samples[m_itr]
            measured_cdf += n_inv; true_cdf += m_inv
            n_itr +=1; m_itr += 1
    if n_itr == n:
        for itr in range(m_itr, m):
            wass_dist += np.abs((1.0- true_cdf)*(true_samples[itr] - prev_sample))
            prev_sample = true_samples[itr]
            true_cdf += m_inv
    else:
        for itr in range(n_itr, n):
            wass_dist += np.abs((measured_cdf - 1.0)*(measured_samples[itr] - prev_sample))
            prev_sample = measured_samples[itr]
            measured_cdf += n_inv

    return wass_dist
