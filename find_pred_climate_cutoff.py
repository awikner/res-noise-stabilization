#!/homes/awikner1/anaconda3/envs/reservoir-rls/bin/python -u
#Assume will be finished in no more than 18 hours
#SBATCH -t 15:00
#SBATCH -p debug
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
import matplotlib
#matplotlib.use('tkagg')

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
    Win_type  = 'old'
    tau_flag  = True

    try:
        opts, args = getopt.getopt(argv, "T:N:r:", \
                ['noisetype=','traintype=', 'system=', 'res=',\
                'tests=','trains=', 'tau=', 'bias_type='])
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
        elif opt == '--bias_type':
            Win_type = str(arg)
            print('Bias Type: %s' % Win_type)
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
    #rhos = np.array([0.1,0.5])
    #sigmas = np.array([0.5,1.0])
    #leakages = np.array([0.875,1.0])
    rhos = np.array([0.1, 0.5])
    sigmas = np.array([0.5, 1.0])
    leakages = np.array([0.875, 1.0])

    mean_sum_squared = np.zeros((rhos.size, noisevals.size, num_trains, res_per_test, num_tests))
    variances        = np.zeros((rhos.size, noisevals.size, num_trains, res_per_test, num_tests))
    valid_time       = np.zeros((rhos.size, noisevals.size, num_trains, res_per_test, num_tests))
    stable_count     = np.zeros((rhos.size, noisevals.size))
    totalcount       = variances.size

    foldername = '/lustre/awikner1/res-noise-stabilization/' + '%s_noisetest_noisetype_%s_traintype_%s/' % (system, noisetype, traintype)
    with tqdm(total = totalcount) as pbar:
        for i, (rho, sigma, leakage) in enumerate(zip(rhos, sigmas, leakages)):
            folder = '%s_more_noisetest_wpred_rho%0.1f_sigma%1.1e_leakage%0.1f_bias_%s_tau%0.2f_%dnodes_%dtrain_%dreals_noisetype_%s_traintype_%s/' % (system, rho, sigma, leakage, Win_type, tau, res_size, train_time, noise_realizations, noisetype, traintype)
            if i == 0:
                test_pred = np.loadtxt(foldername+folder+'pred_%dnodes_%dtrainiters_%dnoisereals_noise%e_train%d_res%d_test%d.csv' \
                    %(res_size, train_time, noise_realizations, noisevals[0], 1, 1, 1), delimiter = ',')
                preds = np.zeros((rhos.size, noisevals.size, num_trains, res_per_test, num_tests, test_pred.shape[0], test_pred.shape[1]))
            for j, noise in enumerate(noise_values_array):
                for k, train in enumerate(np.arange(num_trains)):
                    mean_sum_squared[i,j,k] = np.loadtxt(foldername+folder+'mean_sum_squared_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d.csv' \
                        %(res_size, train_time, noise_realizations, noise, train+1), delimiter = ',')
                    variances[i,j,k]         = np.loadtxt(foldername+folder+'variances_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d.csv' \
                        %(res_size, train_time, noise_realizations, noise, train+1), delimiter = ',')
                    valid_time[i,j,k]       = np.loadtxt(foldername+folder+'valid_time_%dnodes_%dtrainiters_%dnoisereals_noise%e_test%d.csv' \
                        %(res_size, train_time, noise_realizations, noise, train+1), delimiter = ',')
                    for l, res in enumerate(np.arange(res_per_test)):
                        for m, test in enumerate(np.arange(num_tests)):
                            preds[i,j,k,l,m] = np.loadtxt(foldername+folder+'pred_%dnodes_%dtrainiters_%dnoisereals_noise%e_train%d_res%d_test%d.csv' \
                                %(res_size, train_time, noise_realizations, noise, train+1, res+1, test+1), delimiter = ',')
                            pbar.update(1)
                stable_count[i,j] = np.sum(np.logical_and(np.logical_and(variances[i,j].flatten() < 1.1, variances[i,j].flatten() > 0.95), \
                    mean_sum_squared[i,j].flatten() < 1e-2))/variances[i,j].flatten().size

    rhos_mat, noise_mat, train_mat, res_mat, test_mat = np.meshgrid(np.arange(rhos.size), np.arange(noisevals.size), np.arange(num_trains), np.arange(res_per_test), np.arange(num_tests))
    num_elems = rhos_mat.flatten().size
    np.random.seed(10)
    perms     = np.random.permutation(num_elems)
    rhos_mat  = rhos_mat.flatten()[perms]
    noise_mat = noise_mat.flatten()[perms]
    train_mat = train_mat.flatten()[perms]
    res_mat   = res_mat.flatten()[perms]
    test_mat  = test_mat.flatten()[perms]
    variances_test = np.array([])
    mss_test       = np.array([])
    corr_climate   = np.array([])
    end_flag = False
    itr = 0
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex = True, figsize = (21,12))
    plt.ion()
    plt.show()
    for i,j,k,l,m in zip(rhos_mat, noise_mat, train_mat, res_mat, test_mat):
        if end_flag:
            break
        plt.title('MSS: %e    Variance: %f' % (mean_sum_squared[i,j,k,l,m], variances[i,j,k,l,m]))
        ax1.plot(preds[i,j,k,l,m,0,-700:], linewidth = 2)
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('x(t)')
        ax2.plot(preds[i,j,k,l,m,1,-700:], linewidth = 2)
        ax2.set_ylabel('y(t)')
        ax3.plot(preds[i,j,k,l,m,2,-700:], linewidth = 2)
        ax3.set_ylabel('z(t)')
        plt.draw()
        plt.pause(0.001)
        print('MSS: %e    Variance: %f' % (mean_sum_squared[i,j,k,l,m], variances[i,j,k,l,m]))
        result = input('Does this prediction have the correct climate? (yes, no, end): ')
        cont_flag = True
        while cont_flag:
            if result == 'yes' or result == 'y':
                corr_climate = np.append(corr_climate, 1)
                variances_test = np.append(variances_test,variances[i,j,k,l,m])
                mss_test = np.append(mss_test,mean_sum_squared[i,j,k,l,m])
                itr += 1
                cont_flag = False
            elif result == 'no' or result == 'n':
                corr_climate = np.append(corr_climate,0)
                variances_test = np.append(variances_test,variances[i,j,k,l,m])
                mss_test = np.append(mss_test,mean_sum_squared[i,j,k,l,m])
                itr += 1
                cont_flag = False
            elif result == 'end' or result == 'e':
                end_flag = True
                cont_flag = False
            else:
                result = input('Does this prediction have the correct climate? (yes, no, end): ')
        ax1.clear()
        ax2.clear()
        ax3.clear()

    saveflag = input('Save these results? (yes, no): ')
    if saveflag == 'yes' or saveflag == 'y':
        data = np.stack((rhos[rhos_mat[:itr]], sigmas[rhos_mat[:itr]], leakages[rhos_mat[:itr]], \
            noisevals[noise_mat[:itr]], train_mat[:itr], res_mat[:itr], test_mat[:itr],\
            variances_test, mss_test, corr_climate), axis = -1)
        print(data.shape)
        savename = foldername + 'climate_classes_%dnodes_%dtrainiters_%dnoisereals.csv' \
             %(res_size, train_time, noise_realizations)
        print(savename)
        np.savetxt(savename, data, delimiter = ',')
    climate_pos = np.argwhere(corr_climate == 1)
    climate_neg = np.argwhere(corr_climate == 0)

    fig = plt.figure()
    plt.semilogy(variances_test[climate_neg], mss_test[climate_neg], 'b.', markersize = 1, label = 'Incorrect Climate')
    plt.semilogy(variances_test[climate_pos], mss_test[climate_pos], 'r.', markersize = 1, label = 'Correct Climate')
    plt.xlabel('Variance')
    plt.ylabel('Map Error')
    plt.xlim(0,1.5)
    plt.ylim(1e-5, 1e-1)
    plt.show()
    input('End program?')


if __name__ == "__main__":
    main(sys.argv[1:])
