#!/homes/awikner1/anaconda3/envs/reservoir-rls/bin/python -u
#Assume will be finished in no more than 18 hours
#SBATCH -t 2:00:00
#Launch on 12 cores distributed over as many nodes as needed
#SBATCH --ntasks=1
#Assume need 6 GB/core (6144 MB/core)
#SBATCH --mail-user=awikner1@umd.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
import sys, getopt, os
import numpy as np
from scipy.stats import mode
from tqdm import tqdm
import matplotlib.pyplot as plt

def count_elems(array):
    count_dict = {}
    for elem in array:
        if elem in count_dict:
            count_dict[elem] += 1
        else:
            count_dict[elem] = 1
    return count_dict

def main(argv):
    trainlen  = 50
    rsvr_size = 100
    res_per_test = 50
    num_tests  = 20
    num_trains = 50
    noise_realizations = 1

    traintype = 'normal'
    noisetype = 'gaussian'
    system    = 'lorenz'
    bias_type  = 'old'
    tau_flag  = True

    try:
        opts, args = getopt.getopt(argv, "T:N:r:", \
                ['noisetype=','traintype=', 'system=', 'res=',\
                'tests=','trains=', 'tau=', 'win_type=', 'bias_type='])
    except getopt.GetoptError:
        print('Error: Some options not recognized')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-T':
            train_time = int(arg)
            print('Training iterations: %d' % train_time)
        elif opt == '-N':
            res_size = int(arg)
            print('Reservoir nodes: %d' % res_size)
        elif opt == '-r':
            noise_realizations = int(arg)
            print('Noise Realizations: %d' % noise_realizations)
        elif opt == '--tau':
            tau  = float(arg)
            tau_flag = False
            print('Reservoir timestep: %f' % tau)
        elif opt == '--win_type':
            win_type = str(arg)
            print('Win Type: %s' % win_type)
        elif opt == '--bias_type':
            bias_type = str(arg)
            print('Bias Type: %s' % bias_type)
        elif opt == '--res':
            res_per_test = int(arg)
            print('Number of reservoirs: %d' % res_per_test)
        elif opt == '--tests':
            num_tests = int(arg)
            print('Number of tests: %d' % num_tests)
        elif opt == '--trains':
            num_trains = int(arg)
            print('Number of training data sequences: %d' % num_trains)
        elif opt == '--noisetype':
            noisetype = str(arg)
            print('Noise type: %s' % noisetype)
        elif opt == '--traintype':
            traintype = str(arg)
            print('Training type: %s' % traintype)
        elif opt == '--system':
            system = str(arg)
            print('System: %s' % system)
    if tau_flag:
        if system == 'lorenz':
            tau = 0.1
        elif system == 'KS':
            tau = 0.25

    noisevals = np.logspace(-3.666666666666, 0, num = 12, base = 10)[0:8]
    noise_values_array = noisevals
    print(noisevals)
    rhos = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    sigmas = np.array([0.5,1.0,1.5,2.0])
    #sigmas = np.array([1.0])
    leakages = np.array([0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0])
    #leakages = np.array([0.5,0.625,0.75,0.875,1.0])
    rhos_mat, sigmas_mat, leakages_mat = np.meshgrid(rhos, sigmas, leakages)
    rhos_mat     = rhos_mat.flatten()
    sigmas_mat   = sigmas_mat.flatten()
    leakages_mat = leakages_mat.flatten()
    stable_frac_mat = np.zeros(rhos_mat.size)

    foldername = '/lustre/awikner1/res-noise-stabilization/' + '%s_noisetest_noisetype_%s_traintype_%s/' % (system, noisetype, traintype)

    best_stable_frac = 0.
    best_valid_time  = 0.
    total_count = rhos_mat.size*noisevals.size*num_trains
    with tqdm(total = total_count) as pbar:
        for i, (rho, sigma, leakage) in enumerate(zip(rhos_mat, sigmas_mat, leakages_mat)):
            folder = '%s_more_noisetest_rho%0.1f_sigma%1.1e_leakage%0.1f_win_%s_bias_%s_tau%0.2f_%dnodes_%dtrain_%dreals_noisetype_%s_traintype_%s/' % (system, rho,    sigma, leakage, win_type, bias_type, tau, res_size, train_time, noise_realizations, noisetype, traintype)
            folder_new = '%s_more_noisetest_rho%0.1f_sigma%1.1e_leakage%0.3f_win_%s_bias_%s_tau%0.2f_%dnodes_%dtrain_%dreals_noisetype_%s_traintype_%s/' % (system, rho,sigma, leakage, win_type, bias_type, tau,res_size, train_time, noise_realizations, noisetype, traintype)
            filename_sf = 'stable_frac_%dnodes_%dtrainiters_%dnoisereals_raytest.csv' % (res_size, train_time, noise_realizations)
            filename_sf_reg = 'stable_frac_alpha_%dnodes_%dtrainiters_%dnoisereals_raytest.csv' % (res_size, train_time, noise_realizations)
            if leakage in [0.125,0.25,0.375]:
                noisevals_loop = noisevals[4:8]
            else:
                noisevals_loop = noisevals
            median_valid_time = np.zeros(noisevals_loop.size)
            for j, noise in enumerate(noisevals_loop):
                valid_time = np.zeros((num_trains, res_per_test, num_tests))
                for k, train in enumerate(np.arange(num_trains)):
                    valid_time[k] = np.loadtxt(foldername+folder+'valid_time_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d.csv' \
                        %(res_size, train_time, noise_realizations, noise, train+1), delimiter = ',')
                    pbar.update(1)
                median_valid_time[j] = np.median(valid_time.flatten())
            stable_frac = np.loadtxt(foldername + folder + filename_sf, delimiter = ',')
            stable_frac_alpha = np.loadtxt(foldername + folder + filename_sf_reg, delimiter = ',')

            stable_fracs  = np.mean(stable_frac, axis = 1)
            max_noise_itr = np.argmax(stable_fracs)
            stable_frac_mat[i] = stable_fracs[max_noise_itr]

            max_stable_frac = stable_fracs[max_noise_itr]
            if max_stable_frac > best_stable_frac:
                best_stable_frac = max_stable_frac
                best_rho     = rho
                best_sigma   = sigma
                best_leakage = leakage
                best_noise   = noisevals_loop[max_noise_itr]
                best_reg, temp = mode(stable_frac_alpha[max_noise_itr])
                #print(count_elems(stable_frac_alpha[max_noise_itr]))
                best_folder  = folder_new

            max_noise_itr_vt = np.argmax(median_valid_time)
            max_median_vt    = np.max(median_valid_time)

            if max_median_vt > best_valid_time:
                best_valid_time = max_median_vt
                best_rho_vt     = rho
                best_sigma_vt   = sigma
                best_leakage_vt = leakage
                best_noise_vt   = noisevals_loop[max_noise_itr_vt]
                best_reg_vt, temp = mode(stable_frac_alpha[max_noise_itr_vt])
                #print(count_elems(stable_frac_alpha[max_noise_itr]))
                best_folder_vt  = folder_new

    print('Best Valid Time: %f' % best_valid_time)
    print('Best Spectral Radius: %f' % best_rho_vt)
    print('Best Input Weight: %f' % best_sigma_vt)
    print('Best Leakage: %f' % best_leakage_vt)
    print('Best Noise Magnitude: %e' % best_noise_vt)
    print('Best Regularization: %e' % best_reg_vt)
    print(best_folder_vt[:-1])

    best_stable_frac_rho = np.zeros(rhos.size)
    best_sigma_rho       = np.zeros(rhos.size)
    best_leakage_rho     = np.zeros(rhos.size)
    best_reg_rho         = np.zeros(rhos.size)
    best_noise_rho       = np.zeros(rhos.size)
    for j, rho in enumerate(rhos):
        sigmas_rho_cs = sigmas_mat[np.argwhere(rhos_mat == rho)]
        leakages_rho_cs = leakages_mat[np.argwhere(rhos_mat == rho)]
        for i, (sigma, leakage) in tqdm(enumerate(zip(sigmas_rho_cs, leakages_rho_cs)), total = sigmas_rho_cs.size):
            folder = '%s_more_noisetest_rho%0.1f_sigma%1.1e_leakage%0.1f_win_%s_bias_%s_tau%0.2f_%dnodes_%dtrain_%dreals_noisetype_%s_traintype_%s/' % (system, rho, sigma, leakage, win_type, bias_type, tau,res_size, train_time, noise_realizations, noisetype, traintype)
            folder_new = '%s_more_noisetest_rho%0.1f_sigma%1.1e_leakage%0.3f_win_%s_bias_%s_tau%0.2f_%dnodes_%dtrain_%dreals_noisetype_%s_traintype_%s/' % (system, rho, sigma, leakage, win_type, bias_type, tau, res_size, train_time, noise_realizations, noisetype, traintype)
            filename_sf = 'stable_frac_%dnodes_%dtrainiters_%dnoisereals_raytest.csv' % (res_size, train_time, noise_realizations)
            filename_sf_reg = 'stable_frac_alpha_%dnodes_%dtrainiters_%dnoisereals_raytest.csv' % (res_size, train_time, noise_realizations)
            stable_frac = np.loadtxt(foldername + folder + filename_sf, delimiter = ',')
            stable_frac_alpha = np.loadtxt(foldername + folder + filename_sf_reg, delimiter = ',')

            if leakage in [0.125,0.25,0.375]:
                noisevals_loop = noisevals[4:8]
            else:
                noisevals_loop = noisevals

            stable_fracs  = np.mean(stable_frac, axis = 1)
            max_noise_itr = np.argmax(stable_fracs)

            max_stable_frac = stable_fracs[max_noise_itr]
            if max_stable_frac > best_stable_frac_rho[j]:
                best_stable_frac_rho[j] = max_stable_frac
                best_sigma_rho[j]       = sigma
                best_leakage_rho[j]     = leakage
                best_noise_rho[j]       = noisevals_loop[max_noise_itr]
                best_reg_rho[j], temp   = mode(stable_frac_alpha[max_noise_itr])

    """
    mean_sum_squared = np.zeros((rhos.size, noisevals.size, num_trains, res_per_test, num_tests))
    variances        = np.zeros((rhos.size, noisevals.size, num_trains, res_per_test, num_tests))
    valid_time       = np.zeros((rhos.size, noisevals.size, num_trains, res_per_test, num_tests))
    stable_count     = np.zeros((rhos.size, noisevals.size))

    for j, rho in enumerate(rhos):
        folder = '%s_more_noisetest_rho%0.1f_sigma%1.1e_leakage%0.1f_win_%s_bias_%s_tau%0.2f_%dnodes_%dtrain_%dreals_noisetype_%s_traintype_%s/' % (system, rho, best_sigma_rho[j], best_leakage_rho[j], win_type, bias_type, tau,res_size, train_time, noise_realizations, noisetype, traintype)
        for i, noise in enumerate(noise_values_array):
            for k, train in enumerate(np.arange(num_trains)):
                mean_sum_squared[j,i,k] = np.loadtxt(foldername+folder+'mean_sum_squared_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d.csv' \
                    %(res_size, train_time, noise_realizations, noise, train+1), delimiter = ',')
                variances[j,i,k]         = np.loadtxt(foldername+folder+'variances_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d.csv' \
                    %(res_size, train_time, noise_realizations, noise, train+1), delimiter = ',')
                valid_time[j,i,k]       = np.loadtxt(foldername+folder+'valid_time_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d.csv' \
                    %(res_size, train_time, noise_realizations, noise, train+1), delimiter = ',')

            stable_count[j,i] = np.sum(np.logical_and(np.logical_and(variances[j,i].flatten() < 1.1, variances[j,i].flatten() > 0.95), \
                mean_sum_squared[j,i].flatten() < 1e-2))/variances[j,i].flatten().size

        print(np.max(stable_count[j]))
        i = np.argmax(stable_count[j])
        nan_count = np.sum(np.logical_or(np.isnan(variances[j,i].flatten()), np.isnan(mean_sum_squared[j,i].flatten())))/variances[j,i].flatten().size
        inf_count = np.sum(np.logical_or(np.isinf(variances[j,i].flatten()), np.isinf(mean_sum_squared[j,i].flatten())))/variances[j,i].flatten().size
        print('Nan percent: %f' % nan_count)
        print('Inf percent: %f' % inf_count)
        fig = plt.figure()
        plt.semilogy(variances[j,i].flatten(), mean_sum_squared[j,i].flatten(), '.', markersize = 1)
        plt.xlabel('Variance')
        plt.ylabel('Map Error')
        plt.xlim(0,1.5)
        plt.ylim(1e-5, 1e-1)
        plt.title('Spectral Radius: %0.1f' % rho)
        plt.show()
    """




    best_stable_frac_sigma = np.zeros(sigmas.size)
    best_rho_sigma       = np.zeros(sigmas.size)
    best_leakage_sigma     = np.zeros(sigmas.size)
    best_reg_sigma         = np.zeros(sigmas.size)
    best_noise_sigma       = np.zeros(sigmas.size)
    for j, sigma in enumerate(sigmas):
        rhos_sigma_cs = rhos_mat[np.argwhere(sigmas_mat == sigma)]
        leakages_sigma_cs = leakages_mat[np.argwhere(sigmas_mat == sigma)]
        for i, (rho, leakage) in tqdm(enumerate(zip(rhos_sigma_cs, leakages_sigma_cs)), total = rhos_sigma_cs.size):
            folder = '%s_more_noisetest_rho%0.1f_sigma%1.1e_leakage%0.1f_win_%s_bias_%s_tau%0.2f_%dnodes_%dtrain_%dreals_noisetype_%s_traintype_%s/' % (system, rho, sigma, leakage, win_type, bias_type, tau,   res_size, train_time, noise_realizations, noisetype, traintype)
            folder_new = '%s_more_noisetest_rho%0.1f_sigma%1.1e_leakage%0.3f_win_%s_bias_%s_tau%0.2f_%dnodes_%dtrain_%dreals_noisetype_%s_traintype_%s/' % (system, rho, sigma, leakage, win_type, bias_type,    tau, res_size, train_time, noise_realizations, noisetype, traintype)
            filename_sf = 'stable_frac_%dnodes_%dtrainiters_%dnoisereals_raytest.csv' % (res_size, train_time, noise_realizations)
            filename_sf_reg = 'stable_frac_alpha_%dnodes_%dtrainiters_%dnoisereals_raytest.csv' % (res_size, train_time, noise_realizations)
            stable_frac = np.loadtxt(foldername + folder + filename_sf, delimiter = ',')
            stable_frac_alpha = np.loadtxt(foldername + folder + filename_sf_reg, delimiter = ',')

            if leakage in [0.125,0.25,0.375]:
                noisevals_loop = noisevals[4:8]
            else:
                noisevals_loop = noisevals

            stable_fracs  = np.mean(stable_frac, axis = 1)
            max_noise_itr = np.argmax(stable_fracs)

            max_stable_frac = stable_fracs[max_noise_itr]
            if max_stable_frac > best_stable_frac_sigma[j]:
                best_stable_frac_sigma[j] = max_stable_frac
                best_rho_sigma[j]       = rho
                best_leakage_sigma[j]     = leakage
                best_noise_sigma[j]       = noisevals_loop[max_noise_itr]
                best_reg_sigma[j], temp   = mode(stable_frac_alpha[max_noise_itr])

    best_stable_frac_leakage = np.zeros(leakages.size)
    best_rho_leakage       = np.zeros(leakages.size)
    best_sigma_leakage     = np.zeros(leakages.size)
    best_reg_leakage         = np.zeros(leakages.size)
    best_noise_leakage       = np.zeros(leakages.size)
    for j, leakage in enumerate(leakages):
        rhos_leakage_cs = rhos_mat[np.argwhere(leakages_mat == leakage)]
        sigmas_leakage_cs = sigmas_mat[np.argwhere(leakages_mat == leakage)]
        for i, (rho, sigma) in tqdm(enumerate(zip(rhos_leakage_cs, sigmas_leakage_cs)), total = rhos_leakage_cs.size):
            folder = '%s_more_noisetest_rho%0.1f_sigma%1.1e_leakage%0.1f_win_%s_bias_%s_tau%0.2f_%dnodes_%dtrain_%dreals_noisetype_%s_traintype_%s/' % (system, rho, sigma, leakage, win_type, bias_type,        tau,   res_size, train_time, noise_realizations, noisetype, traintype)
            folder_new = '%s_more_noisetest_rho%0.1f_sigma%1.1e_leakage%0.3f_win_%s_bias_%s_tau%0.2f_%dnodes_%dtrain_%dreals_noisetype_%s_traintype_%s/' % (system, rho, sigma, leakage, win_type, bias_type,    tau, res_size, train_time, noise_realizations, noisetype, traintype)
            filename_sf = 'stable_frac_%dnodes_%dtrainiters_%dnoisereals_raytest.csv' % (res_size, train_time, noise_realizations)
            filename_sf_reg = 'stable_frac_alpha_%dnodes_%dtrainiters_%dnoisereals_raytest.csv' % (res_size, train_time, noise_realizations)
            stable_frac = np.loadtxt(foldername + folder + filename_sf, delimiter = ',')
            stable_frac_alpha = np.loadtxt(foldername + folder + filename_sf_reg, delimiter = ',')

            if leakage in [0.125,0.25,0.375]:
                noisevals_loop = noisevals[4:8]
            else:
                noisevals_loop = noisevals

            stable_fracs  = np.mean(stable_frac, axis = 1)
            max_noise_itr = np.argmax(stable_fracs)

            max_stable_frac = stable_fracs[max_noise_itr]
            if max_stable_frac > best_stable_frac_leakage[j]:
                best_stable_frac_leakage[j] = max_stable_frac
                best_rho_leakage[j]       = rho
                best_sigma_leakage[j]     = sigma
                best_noise_leakage[j]       = noisevals_loop[max_noise_itr]
                best_reg_leakage[j], temp   = mode(stable_frac_alpha[max_noise_itr])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (14,8))
    plt.ion()
    #plt.show()
    ax1.plot(rhos, best_stable_frac_rho, 'x-')
    for x,y,sigma,leakage,noise,reg in \
        zip(rhos, best_stable_frac_rho, best_sigma_rho, best_leakage_rho, best_noise_rho, best_reg_rho):
        label = 'Sigma: %0.1f\nLeakage: %0.3f\nNoise: %0.2e\nReg: %0.2e' % (sigma,leakage,noise,reg)
        ax1.annotate(label, (x,y), textcoords = "offset points", xytext = (0,10), ha = 'center')

    ax1.set_xlabel('Spectral Radius')
    ax1.set_ylabel('Optimized Stable Fraction')
    ax2.plot(sigmas, best_stable_frac_sigma, 'x-')
    for x,y,rho,leakage,noise,reg in \
        zip(sigmas, best_stable_frac_sigma, best_rho_sigma, best_leakage_sigma, best_noise_sigma, best_reg_sigma):
        label = 'Rho: %0.1f\nLeakage: %0.3f\nNoise: %0.2e\nReg: %0.2e' % (rho,leakage,noise,reg)
        ax2.annotate(label, (x,y), textcoords = "offset points", xytext = (0,10), ha = 'center')
    ax2.set_xlabel('Input Weight')
    ax2.set_ylabel('Optimized Stable Fraction')

    ax3.plot(leakages, best_stable_frac_leakage, 'x-')
    for x,y,rho,sigma,noise,reg in \
        zip(leakages, best_stable_frac_leakage, best_rho_leakage, best_sigma_leakage, best_noise_leakage, best_reg_leakage):
        label = 'Rho: %0.1f\nSigma: %0.1f\nNoise: %0.2e\nReg: %0.2e' % (rho,sigma,noise,reg)
        ax3.annotate(label, (x,y), textcoords = "offset points", xytext = (0,10), ha = 'center')
    ax3.set_xlabel('Leakage')
    ax3.set_ylabel('Optimized Stable Fraction')
    fig.suptitle('Noise: %s, Training: %s, Noise Reals.: %d, Num. Nodes: %d, Train Length: %d' % (noisetype, traintype, noise_realizations, res_size, train_time))
    plt.draw()
    plt.pause(0.001)
    plt.savefig(foldername+'%s_more_noisetest_win_%s_bias_%s_tau%0.2f_%dnodes_%dtrain_%dreals_noisetype_%s_traintype_%s_hyperoptim_plot.png' % (system, win_type, bias_type,  tau,res_size, train_time, noise_realizations, noisetype, traintype))


    print('Best Stable Fraction: %f' % best_stable_frac)
    print('Best Spectral Radius: %f' % best_rho)
    print('Best Input Weight: %f' % best_sigma)
    print('Best Leakage: %f' % best_leakage)
    print('Best Noise Magnitude: %e' % best_noise)
    print('Best Regularization: %e' % best_reg)
    print(best_folder[:-1])
    optim_folder = best_folder[:-1] + '_hyperoptim'

    if os.path.isdir(os.path.join(os.path.join(foldername, optim_folder), '/')):
        os.system('rm -rf %s%s' % (foldername, optim_folder))
    os.system('cp -R %s%s %s%s' % (foldername, folder[:-1], foldername, optim_folder))

    print('Finished!')
    #input('Press [Enter] to end.')




if __name__ == "__main__":
    main(sys.argv[1:])
