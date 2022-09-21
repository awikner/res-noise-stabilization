import pandas as pd
from tqdm import tqdm
import numpy as np
import time
from itertools import product
import sys
import os
import getopt
from res_reg_lmnt_awikner.helpers import get_windows_path, get_filename
from res_reg_lmnt_awikner.lorenzrungekutta_numba import lorenzrungekutta
from res_reg_lmnt_awikner.ks_etdrk4 import kursiv_predict
from scipy.sparse.linalg import eigs
from scipy.sparse import csc_matrix

__docformat__ = "google"


class RunOpts:
    """This class contains the various parameters and options that will be used during the time series data generation,
    reservoir generation, training, testing, and output."""
    def __init__(self, argv=None,
                 runflag=True,
                 train_time=20000,
                 test_time=None,
                 sync_time=2000,
                 discard_time=500,
                 res_size=500,
                 res_per_test=1,
                 noise_realizations=1,
                 num_tests=1,
                 num_trains=1,
                 traintype='normal',
                 noisetype='gaussian',
                 system='KS',
                 savepred=False,
                 save_time_rms=False,
                 squarenodes=True,
                 rho=0.6,
                 sigma=0.1,
                 theta=0.1,
                 leakage=1.0,
                 bias_type='new_random',
                 win_type='full_0centered',
                 debug_mode=False,
                 pmap=False,
                 machine='deepthought2',
                 ifray=False,
                 tau=None,
                 num_cpus=1,
                 metric='mean_rms',
                 return_all=True,
                 save_eigenvals=False,
                 max_valid_time=2000,
                 noise_values_array=np.logspace(-4, 3, num=3, base=10),
                 reg_values=np.append(0., np.logspace(-11, -9, 5)),
                 res_start=0,
                 train_start=0,
                 test_start=0,
                 reg_train_times=None,
                 root_folder=None,
                 save_folder=None,
                 prior='zero',
                 save_truth=False,
                 dyn_noise=0):
        """Initializes the RunOpts class.

        Raises:
            ValueError: If an input instance variable has an unaccepted value.
            TypeError: If an input instance variable has an incorrect type."""
        self.argv = argv
        """Command line input for generating a RunOpts object. If left as None, then the object will be generated using
            the other inputs or defaults. If multiple instances of the same variable are given, the class will default
            to the command line input. See RunOpts.get_run_opts for more information. Default: None"""
        self.system = system
        """String denoting which dynamical system we are obtaining time series data from. Options: 'lorenz' for the
        Lorenz 63 equations, 'KS' for the Kuramoto-Sivashinsky equation with 64 grid points and a periodicity length of
        22, and KS_d2175 for the Kuramoto-Sivashinksy equation with 64 grid points and a periodicity length of 21.75.
        Default: 'KS'"""
        self.tau = tau
        """Time between each data point in the time series data. If system = 'lorenz', this value must be evenly divided
        by 0.01 (the integration time step). If left as None, then the class will set tau to the default value for the
        particular dynamical system. Default: None"""
        self.dyn_noise = dyn_noise
        """Magnitude of dynamical noise added to true system state at each integration step. Default: 0"""
        self.train_time = train_time
        """Number of training data points. Default: 20000"""
        self.test_time = test_time
        """Number of testing data points. If left as None, then the class will default to 4000 if system = 'lorenz', or
        16000 if system = 'KS' or 'KS_d2175'. Default: None"""
        self.sync_time = sync_time
        """Number of data points used to synchronize the reservoir to each test data set. Default: 2000"""
        self.discard_time = discard_time
        """Number of data points used to synchronize the reservoir to each training data set. Default: 500"""
        self.res_size = res_size
        """Number of nodes in the reservoir. Default: 500"""
        self.rho = rho
        """Reservoir spectral radius. Default: 0.6"""
        self.sigma = sigma
        """Reservoir input scaling. Default: 0.1"""
        self.theta = theta
        """Reservoir input bias scaling. Default: 0.1"""
        self.leakage = leakage
        """Reservoir leaking rate. Default: 1.0"""
        self.squarenodes = squarenodes
        """Boolean denoting whether or not the squared node states are including in the reservoir feature vector.
        Default: True"""
        self.bias_type = bias_type
        """Type of reservoir input bias to be used. See the Reservoir class for available options.
        Default: new_random"""
        self.win_type = win_type
        """Type of input coupling matrix to be used. See the Reservoir class for available options.
        Default: full_0centered"""
        self.traintype = traintype
        """Type of training to be used to determine the reservoir output coupling matrix. There are a number of options
        available, but those used in the paper are:

        'normal' - Standard reservoir training, potentially with input noise added.

        'gradientk%d' % (Number of noise steps) - Reservoir training with no noise and LMNT regularization for a number of
        noise steps > 1, or Jacobian regularization for a number of noise steps = 1.

        'regzerok%d' % (Number of noise steps) - Reservoir training with no noise and LMNT/Jacobian regularization computed
        using a zero-input and zero reservoir state.

        Default: 'normal'
        """
        self.noisetype = noisetype
        """Type of noise to be added to the reservoir input during training. Options are 'none' and 'gaussian'.
        Default: 'none'"""
        self.noise_values_array = noise_values_array
        """Numpy array containing the variance of the added input noise (if noisetype = 'gaussian') or the LMNT/Jacobian
        regularization parameter value (if traintype = 'gradientk%d' or 'regzerok%d'). Each value contained in the array
        will be tested separately using each of the reservoirs, training, and testing data sets.
        Default: np.logspace(-4, 3, num=3, base=10)"""
        self.reg_train_times = reg_train_times
        """Numpy array containing the number of training data points to be used to train the LMNT or Jacobian
        regularization. If left as None, then the class will default to an array containing only
        the total number of training data points for standard LMNT/Jacobian or the number of noise steps
        if using zero-input LMNT. Default: None"""
        self.noise_realizations = noise_realizations
        """Number of input noise realizations used to train the reservoir (if training with noise). If not training with
        noise, set to 1. Default: 1"""
        self.reg_values = reg_values
        """Numpy array containing the Tikhonov regularization parameter values. Each value contained in the array
        will be tested separately using each of the reservoirs, training, and testing data sets.
        Default: np.append(0., np.logspace(-11, -9, 5))"""
        self.prior = prior
        """Prior to be used when computing the output coupling matrix using Tikhonov regularization. Options are:

        'zero' - Standard Tikhonov regularization with a zero prior.

        'input_pass' - Tikhonov regularization with a persistence prior (i.e., set the input pass-through weights to 1).

        Default: 'zero'"""
        self.max_valid_time = max_valid_time
        """Maximum valid time for each valid time test during the testing period. This should be set so that
        test_time / max_valid_time is a whole number greater than 0. Default: 2000"""
        self.res_per_test = res_per_test
        """Number of random reservoir realizations to test. Default: 1"""
        self.num_trains = num_trains
        """Number of independently generated training data sets to test with. Default: 1"""
        self.num_tests = num_tests
        """Number of independently generated testing data sets to test with. Default: 1"""
        self.res_start = res_start
        """Starting iterate for generating the random seeds that are used to generate the reservoir. Default: 0"""
        self.train_start = train_start
        """Starting iterate for generating the random seeds that are used to generate the training data sets.
        Default: 0"""
        self.test_start = test_start
        """Starting iterate for generating the random seeds that are used to generate the testing data sets.
        Default: 0"""
        self.root_folder = root_folder
        """Location where output data will be stored in the Data folder. If None, then defaults to the current working
        directory. Default: None"""
        self.save_folder = save_folder
        """Location where processed output data will be stored in the Data folder. If none, then defaults to the 
        current working directory. Default: None"""
        self.return_all = return_all
        """Boolean for determining of all results should be returned, or only the results with the obtained using the
        "best" Tikhonov regularization parameter value based on the selected metric. Default: True"""
        self.metric = metric
        """Metric for determining the "best" results. Not used if return_all = True. Options include 'mean_rms', 'max_rms',
        and 'stable_frac'. Caution: Some options may be deprecated. Default: 'mss-var'"""
        self.savepred = savepred
        """Boolean for determining if reservoir prediction time series should be saved. Default: False"""
        self.save_time_rms = save_time_rms
        """Boolean for determining if reservoir prediction RMS error should be saved. Default: False"""
        self.pmap = pmap
        """Boolean for determining if reservoir prediction Poincare maximum map should be saved. Default: False"""
        self.save_eigenvals = save_eigenvals
        """Boolean for determining if the eigenvalues of the LMNT/Jacobian regularization matrices should be saved.
        Default: False"""
        self.save_truth = save_truth
        """Boolean for determining if the true testing data should be saved. Default: False"""
        self.ifray = ifray
        """Boolean for determining if ray should be used to compute results for multiple reservoirs and training
        data sets. Default: False"""
        self.num_cpus = num_cpus
        """If using ray for paralellization, this sets the number of cpus to be used. Default: 1"""
        self.machine = machine
        """Machine which results are computed on. Leave as personal unless you are connecting to a ray cluster
        elsewhere. Default: 'personal'"""
        self.runflag = runflag
        """True indicates that we are about to compute results, and the appropriate directories should be created.
        Otherwise, we do not create additional directories. Default: True"""
        self.debug_mode = debug_mode
        """Boolean for determining if errors during reservoir training which could arise from non-convergence of the
        eigenvalue solver should be suppressed. If left as False, will suppress errors im much of the core code,
        so this should be set to True if making changes. Default: False"""
        if not isinstance(argv, type(None)):
            self.get_run_opts()
        if isinstance(self.tau, type(None)):
            if self.system == 'lorenz':
                self.tau = 0.1
            elif self.system in ['KS', 'KS_d2175']:
                self.tau = 0.25
        if isinstance(self.test_time, type(None)):
            if self.system == 'lorenz':
                self.test_time = 4000
            elif 'KS' in self.system:
                self.test_time = 16000
        if not isinstance(self.reg_train_times, np.ndarray):
            if isinstance(self.reg_train_times, type(None)):
                self.reg_train_times = np.array([self.train_time])
            elif isinstance(self.reg_train_times, int):
                self.reg_train_times = np.array([self.reg_train_times])
            else:
                raise TypeError()
        if isinstance(self.root_folder, type(None)):
            self.root_folder = os.getcwd()
        if isinstance(self.save_folder, type(None)):
            self.save_folder = os.getcwd()
        if isinstance(self.reg_train_times, np.ndarray) or isinstance(self.reg_train_times, list):
            if (self.reg_train_times[0] != self.train_time or len(self.reg_train_times) != 1) and (
                    self.traintype in ['normal', 'normalres1', 'normalres2', 'rmean', 'rmeanres1',
                                       'rmeanres2', 'rplusq', 'rplusqres1',
                                       'rplusqres2'] or 'confined' in self.traintype):
                print(('Traintypes "normal", "rmean", and "rplusq" are not '
                       'compatible with fractional regularization training.'))
                raise ValueError
        if self.prior not in ['zero', 'input_pass']:
            print('Prior type not recognized.')
            raise ValueError
        self.save_file_name = ''
        self.run_folder_name = ''
        self.save_folder_name = ''
        self.get_file_name()

    def get_file_name(self):
        """Creates the folder and final data file name for the tests about to be run.
        Args:
            self: RunOpts object.
        Returns:
            RunOpts object with initialized folder and file name variables. Also creates the aforementioned folder
            if not already created.
        """
        if self.prior == 'zero':
            prior_str = ''
        else:
            prior_str = '_prior_%s' % self.prior
        if self.savepred:
            predflag = '_wpred'
        else:
            predflag = ''
        if self.save_time_rms:
            timeflag = '_savetime'
        else:
            timeflag = ''
        if self.squarenodes:
            squarenodes_flag = '_squarenodes'
        else:
            squarenodes_flag = ''
        if self.save_eigenvals:
            eigenval_flag = '_wmoregradeigs'
        else:
            eigenval_flag = ''
        if self.pmap:
            pmap_flag = '_wpmap0'
        else:
            pmap_flag = ''
        if self.dyn_noise == 0:
            dyn_noise_str = ''
        else:
            dyn_noise_str = '_dnoise%e' % self.dyn_noise
        data_folder_base = os.path.join(self.root_folder, 'Data')
        save_folder_base = os.path.join(self.save_folder, 'Data')
        if not os.path.isdir(data_folder_base):
            os.mkdir(data_folder_base)
        if not os.path.isdir(save_folder_base):
            os.mkdir(save_folder_base)
        if not self.return_all:
            data_folder = '%s_noisetest_noisetype_%s_traintype_%s' % (
                self.system, self.noisetype, self.traintype)
            run_name = ('%s%s%s%s%s%s%s_rho%0.1f_sigma%1.1e_leakage%0.3f_win_%s_bias_%s_tau%0.2f'
                        '_%dnodes_%dtrain_%dreals_noisetype_%s_traintype_%s%s_metric_%s') \
                       % (self.system, predflag, timeflag, eigenval_flag, pmap_flag, squarenodes_flag, dyn_noise_str,
                          self.rho, self.sigma, self.leakage, self.win_type, self.bias_type, self.tau, self.res_size,
                          self.train_time, self.noise_realizations, self.noisetype, self.traintype, prior_str,
                          self.metric)
        else:
            data_folder = '%s_noisetest_noisetype_%s_traintype_%s' % (
                self.system, self.noisetype, self.traintype)
            run_name = ('%s%s%s%s%s%s%s_rho%0.1f_sigma%1.1e_leakage%0.3f_win_%s_bias_%s_tau%0.2f'
                        '_%dnodes_%dtrain_%dreals_noisetype_%s_traintype_%s%s') % (
                           self.system, predflag, timeflag, eigenval_flag, pmap_flag, squarenodes_flag, dyn_noise_str,
                           self.rho, self.sigma,
                           self.leakage, self.win_type, self.bias_type, self.tau, self.res_size, self.train_time,
                           self.noise_realizations, self.noisetype, self.traintype, prior_str)
        root_data_folder = os.path.join(data_folder_base, data_folder)
        save_data_folder = os.path.join(save_folder_base, data_folder)
        if self.runflag:
            if not os.path.isdir(root_data_folder):
                os.mkdir(root_data_folder)
            if not os.path.isdir(save_data_folder):
                os.mkdir(save_data_folder)
            if not os.path.isdir(os.path.join(root_data_folder, run_name + '_folder')):
                os.mkdir(os.path.join(root_data_folder, run_name + '_folder'))
            if not os.path.isdir(os.path.join(save_data_folder, run_name + '_folder')) and (self.savepred or
                    self.pmap or self.save_truth):
                os.mkdir(os.path.join(save_data_folder, run_name + '_folder'))
        self.save_file_name = os.path.join(save_data_folder, run_name + '.bz2')
        self.run_folder_name = os.path.join(root_data_folder, run_name + '_folder')
        self.save_folder_name = os.path.join(save_data_folder, run_name + '_folder')

    def get_run_opts(self):
        """Processes the command line input into instance variables.
        Args:
            self: RunOpts object.
        Returns:
            RunOpts object with instance variables set from command line input.
        Raises:
            GetoptError: Raises an error of command line arguments no recognized."""
        try:
            opts, args = getopt.getopt(self.argv, "T:N:r:",
                                       ['testtime=', 'noisetype=', 'traintype=', 'system=', 'res=',
                                        'tests=', 'trains=', 'savepred=', 'tau=', 'rho=',
                                        'sigma=', 'theta=','leakage=', 'bias_type=', 'debug=', 'win_type=',
                                        'machine=', 'num_cpus=', 'pmap=', 'parallel=', 'metric=', 'returnall=',
                                        'savetime=', 'saveeigenvals=', 'noisevals=', 'regvals=', 'maxvt=',
                                        'resstart=', 'trainstart=', 'teststart=',
                                        'squarenodes=', 'regtraintimes=', 'discardlen=',
                                        'prior=', 'synctime=', 'datarootdir=', 'datasavedir=', 'savetruth='])
        except getopt.GetoptError:
            print('Error: Some options not recognized')
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-T':
                self.train_time = int(arg)
                print('Training iterations: %d' % self.train_time)
            elif opt == '-N':
                self.res_size = int(arg)
                print('Reservoir nodes: %d' % self.res_size)
            elif opt == '-r':
                self.noise_realizations = int(arg)
                print('Noise Realizations: %d' % self.noise_realizations)
            elif opt == '--savetruth':
                if arg == 'True':
                    self.save_truth = True
                elif arg == 'False':
                    self.save_truth = False
                else:
                    raise ValueError
            elif opt == '--datarootdir':
                self.root_folder = str(arg)
                print('Root directory for data: %s' % self.root_folder)
            elif opt == '--datasavedir':
                self.save_folder = str(arg)
                print('Root directory for processed data: %s' % self.save_folder)
            elif opt == '--synctime':
                self.sync_time = int(arg)
                print('Sync time: %d' % self.sync_time)
            elif opt == '--testtime':
                self.test_time = int(arg)
                if self.test_time == 0:
                    print('Testing duration: default')
                else:
                    print('Testing duration: %d' % self.test_time)
            elif opt == '--resstart':
                self.res_start = int(arg)
                print('Reservoir ensemble start: %d' % self.res_start)
            elif opt == '--trainstart':
                self.train_start = int(arg)
                print('Train ensemble start: %d' % self.train_start)
            elif opt == '--teststart':
                self.test_start = int(arg)
                print('Test ensemble start: %d' % self.test_start)
            elif opt == '--saveeigenvals':
                if arg == 'True':
                    self.save_eigenvals = True
                elif arg == 'False':
                    self.save_eigenvals = False
                else:
                    raise ValueError
                print('Save grad reg eigenvalues: %s' % arg)
            elif opt == '--prior':
                self.prior = str(arg)
                print('Prior type: %s' % self.prior)
            elif opt == '--discardlen':
                self.discard_time = int(arg)
                print('Discard iterations: %d' % self.discard_time)
            elif opt == '--squarenodes':
                if arg == 'True':
                    self.squarenodes = True
                elif arg == 'False':
                    self.squarenodes = False
                else:
                    raise ValueError
                print('Square reservoir nodes: %s' % arg)
            elif opt == '--maxvt':
                self.max_valid_time = int(arg)
                print('Maximum valid time: %d' % self.max_valid_time)
            elif opt == '--noisevals':
                self.noise_values_array = np.array([float(noise) for noise in arg.split(',')])
                noise_str = '[ '
                for noise in self.noise_values_array:
                    noise_str += '%0.3e, ' % noise
                noise_str = noise_str[:-2] + ' ]'
                print('Noise values: %s' % noise_str)
            elif opt == '--regvals':
                self.reg_values = np.array([float(reg) for reg in arg.split(',')])
                reg_str = '[ '
                for reg in self.reg_values:
                    reg_str += '%0.3e, ' % reg
                reg_str = reg_str[:-2] + ' ]'
                print('Regularization values: %s' % reg_str)
            elif opt == '--regtraintimes':
                if arg != 'None':
                    self.reg_train_times = np.array([int(reg_train) for reg_train in arg.split(',')])
                    reg_train_str = '[ '
                    for reg_train in self.reg_train_times:
                        reg_train_str += '%0.3e, ' % reg_train
                    reg_train_str = reg_train_str[:-2] + ' ]'
                    print('Regularization training times: %s' % reg_train_str)
            elif opt == '--savetime':
                if str(arg) == 'True':
                    self.save_time_rms = True
                elif str(arg) == 'False':
                    self.save_time_rms = False
                else:
                    raise ValueError
            elif opt == '--metric':
                self.metric = str(arg)
                if self.metric not in ['pmap_max_wass_dist', 'mean_rms', 'max_rms', 'mss_var']:
                    raise ValueError
                print('Stability metric: %s' % self.metric)
            elif opt == '--returnall':
                if arg == 'True':
                    self.return_all = True
                elif arg == 'False':
                    self.return_all = False
                else:
                    raise ValueError
            elif opt == '--parallel':
                parallel_in = str(arg)
                if parallel_in == 'True':
                    self.ifray = True
                elif parallel_in == 'False':
                    self.ifray = False
                else:
                    raise ValueError
            elif opt == '--pmap':
                pmap_in = str(arg)
                if pmap_in == 'True':
                    self.pmap = True
                elif pmap_in == 'False':
                    self.pmap = False
                else:
                    raise ValueError
            elif opt == '--machine':
                self.machine = str(arg)
                if self.machine not in ['deepthought2', 'zaratan', 'personal']:
                    raise ValueError
                print('Machine: %s' % self.machine)
            elif opt == '--num_cpus':
                self.num_cpus = int(arg)
                print('Number of CPUS: %d' % self.num_cpus)
            elif opt == '--rho':
                self.rho = float(arg)
                print('Rho: %f' % self.rho)
            elif opt == '--sigma':
                self.sigma = float(arg)
                print('Sigma: %f' % self.sigma)
            elif opt == '--theta':
                self.theta = float(arg)
                print('Theta: %f' % self.theta)
            elif opt == '--leakage':
                self.leakage = float(arg)
                print('Leakage: %f' % self.leakage)
            elif opt == '--tau':
                self.tau = float(arg)
                print('Reservoir timestep: %f' % self.tau)
            elif opt == '--win_type':
                self.win_type = str(arg)
                print('Win Type: %s' % self.win_type)
            elif opt == '--bias_type':
                self.bias_type = str(arg)
                print('Bias Type: %s' % self.bias_type)
            elif opt == '--res':
                self.res_per_test = int(arg)
                print('Number of reservoirs: %d' % self.res_per_test)
            elif opt == '--tests':
                self.num_tests = int(arg)
                print('Number of tests: %d' % self.num_tests)
            elif opt == '--trains':
                self.num_trains = int(arg)
                print('Number of training data sequences: %d' % self.num_trains)
            elif opt == '--savepred':
                if arg == 'True':
                    self.savepred = True
                elif arg == 'False':
                    self.savepred = False
                print('Saving predictions: %s' % arg)
            elif opt == '--noisetype':
                self.noisetype = str(arg)
                print('Noise type: %s' % self.noisetype)
            elif opt == '--traintype':
                self.traintype = str(arg)
                print('Training type: %s' % self.traintype)
            elif opt == '--system':
                self.system = str(arg)
                print('System: %s' % self.system)
            elif opt == '--debug':
                if arg == 'True':
                    self.debug_mode = True
                elif arg == 'False':
                    self.debug_mode = False
                print('Debug Mode: %s' % arg)


class Reservoir:
    """Class for initializing and storing the reservoir matrices and internal states."""
    def __init__(self, run_opts, res_gen, res_itr, input_size, avg_degree=3):
        """Initializes the Reservoir object.
        Args:
            run_opts: RunOpts object containing parameters used to generate the reservoir.
            res_gen: A numpy.random.Generator object used to generate the random matrices in the Reservoir.
            res_itr: Reservoir iteration tag.
            input_size: Number of elements in reservoir input.
            avg_degree: Average in-degree of each reservoir node (i.e., the average number of edges that connect into each vertex in the graph). Default: 3
        Returns:
            Constructed Reservoir object.
        Raises:
            ValueError: If win_type or bias_type is not recognized."""
        # Define class for storing reservoir layers generated from input parameters and an input random number generator
        self.rsvr_size = run_opts.res_size
        self.res_itr = res_itr

        density = avg_degree / self.rsvr_size

        if run_opts.win_type == 'full_0centered':
            unnormalized_W = 2 * res_gen.random((self.rsvr_size, self.rsvr_size)) - 1
        else:
            unnormalized_W = res_gen.random((self.rsvr_size, self.rsvr_size))
        for i in range(unnormalized_W[:, 0].size):
            for j in range(unnormalized_W[0].size):
                if res_gen.random(1) > density:
                    unnormalized_W[i][j] = 0

        max_eig = eigs(unnormalized_W, k=1,
                       return_eigenvectors=False, maxiter=10 ** 5, v0=res_gen.random(self.rsvr_size))

        W_sp = csc_matrix(np.ascontiguousarray(
            run_opts.rho / np.abs(max_eig[0]) * unnormalized_W))
        self.W_data, self.W_indices, self.W_indptr, self.W_shape = \
            (W_sp.data, W_sp.indices, W_sp.indptr, np.array(list(W_sp.shape)))
        # print('Avg. degree of W:')
        # print(self.W_data.size/rsvr_size)

        # print('Adjacency matrix section:')
        # print(self.W_data[:4])

        if run_opts.win_type == 'dense':
            if run_opts.bias_type != 'new_random':
                raise ValueError
            Win = (res_gen.random(self.rsvr_size * (input_size + 1)).reshape(self.rsvr_size,
                                                                             input_size + 1) * 2 - 1) * run_opts.sigma
        else:
            if 'full' in run_opts.win_type:
                input_vars = np.arange(input_size)
            elif run_opts.win_type == 'x':
                input_vars = np.array([0])
            else:
                print('Win_type not recognized.')
                raise ValueError()
            if run_opts.bias_type == 'old':
                const_frac = 0.15
                const_conn = int(self.rsvr_size * const_frac)
                Win = np.zeros((self.rsvr_size, input_size + 1))
                Win[:const_conn, 0] = (res_gen.random(
                    Win[:const_conn, 0].size) * 2 - 1) * run_opts.theta
                q = int((self.rsvr_size - const_conn) // input_vars.size)
                for i, var in enumerate(input_vars):
                    Win[const_conn + q * i:const_conn + q *
                                           (i + 1), var + 1] = (res_gen.random(q) * 2 - 1) * run_opts.sigma
            elif run_opts.bias_type == 'new_random':
                Win = np.zeros((self.rsvr_size, input_size + 1))
                Win[:, 0] = (res_gen.random(self.rsvr_size) * 2 - 1) * run_opts.theta
                q = int(self.rsvr_size // input_vars.size)
                for i, var in enumerate(input_vars):
                    Win[q * i:q * (i + 1), var + 1] = (res_gen.random(q) * 2 - 1) * run_opts.sigma
                leftover_nodes = self.rsvr_size - q * input_vars.size
                for i in range(leftover_nodes):
                    Win[self.rsvr_size - leftover_nodes + i, input_vars[res_gen.integers(input_vars.size)]] = \
                        (res_gen.random() * 2 - 1) * run_opts.sigma
            elif run_opts.bias_type == 'new_new_random':
                Win = np.zeros((self.rsvr_size, input_size + 1))
                Win[:, 0] = (res_gen.random(self.rsvr_size) * 2 - 1) * run_opts.theta
                q = int(self.rsvr_size // input_vars.size)
                for i, var in enumerate(input_vars):
                    Win[q * i:q * (i + 1), var + 1] = (res_gen.random(q) * 2 - 1) * run_opts.sigma
                leftover_nodes = self.rsvr_size - q * input_vars.size
                var = input_vars[res_gen.choice(
                    input_vars.size, size=leftover_nodes, replace=False)]
                Win[self.rsvr_size - leftover_nodes:, var +
                                                      1] = (res_gen.random(leftover_nodes) * 2 - 1) * run_opts.sigma
            elif run_opts.bias_type == 'new_const':
                Win = np.zeros((self.rsvr_size, input_size + 1))
                Win[:, 0] = run_opts.theta
                q = int(self.rsvr_size // input_vars.size)
                for i, var in enumerate(input_vars):
                    Win[q * i:q * (i + 1), var + 1] = (res_gen.random(q) * 2 - 1) * run_opts.sigma
                leftover_nodes = self.rsvr_size - q * input_vars.size
                var = input_vars[res_gen.integers(
                    input_vars.size, size=leftover_nodes)]
                Win[self.rsvr_size - leftover_nodes:, var +
                                                      1] = (res_gen.random(leftover_nodes) * 2 - 1) * run_opts.sigma
            else:
                print('bias_type not recognized.')
                raise ValueError()

        Win_sp = csc_matrix(Win)
        self.Win_data, self.Win_indices, self.Win_indptr, self.Win_shape = \
            np.ascontiguousarray(Win_sp.data), \
            np.ascontiguousarray(Win_sp.indices), \
            np.ascontiguousarray(Win_sp.indptr), \
            np.array(list(Win_sp.shape))

        self.X = (res_gen.random((self.rsvr_size, run_opts.train_time + run_opts.discard_time + 2)) * 2 - 1)
        self.Wout = np.array([])
        self.leakage = run_opts.leakage


class ResOutput:
    """Class for holding the output from a reservoir computer test. Is typically used to save the output from one of
    the (potentially parallel) runs."""
    def __init__(self, run_opts, noise_array):
        """Creates the ResOutput object.
        Args:
            self: ResOutput object
            run_opts: RunOpts object for the test.
            noise_array: Array of noise/Jacobian/LMNT regularization parameter values used in this test.
        Returns:
            self: initialized ResOutput object."""
        self.stable_frac_out = np.zeros((noise_array.size, run_opts.reg_values.size, run_opts.reg_train_times.size),
                                        dtype=object)
        self.mean_rms_out = np.zeros((noise_array.size, run_opts.reg_values.size, run_opts.reg_train_times.size),
                                     dtype=object)
        self.max_rms_out = np.zeros((noise_array.size, run_opts.reg_values.size, run_opts.reg_train_times.size),
                                    dtype=object)
        self.variances_out = np.zeros((noise_array.size, run_opts.reg_values.size, run_opts.reg_train_times.size),
                                      dtype=object)
        self.valid_time_out = np.zeros((noise_array.size, run_opts.reg_values.size, run_opts.reg_train_times.size),
                                       dtype=object)
        self.rms_out = np.zeros((noise_array.size, run_opts.reg_values.size, run_opts.reg_train_times.size),
                                dtype=object)
        self.preds_out = np.zeros((noise_array.size, run_opts.reg_values.size, run_opts.reg_train_times.size),
                                  dtype=object)
        self.wass_dist_out = np.zeros((noise_array.size, run_opts.reg_values.size, run_opts.reg_train_times.size),
                                      dtype=object)
        self.pmap_max_out = np.zeros((noise_array.size, run_opts.reg_values.size, run_opts.reg_train_times.size),
                                     dtype=object)
        self.pmap_max_wass_dist_out = np.zeros(
            (noise_array.size, run_opts.reg_values.size, run_opts.reg_train_times.size), dtype=object)
        self.stable_frac_out = np.zeros((noise_array.size, run_opts.reg_values.size, run_opts.reg_train_times.size),
                                        dtype=object)
        self.train_mean_rms_out = np.zeros((noise_array.size, run_opts.reg_values.size, run_opts.reg_train_times.size),
                                           dtype=object)
        self.train_max_rms_out = np.zeros((noise_array.size, run_opts.reg_values.size, run_opts.reg_train_times.size),
                                          dtype=object)
        self.grad_eigenvals_out = np.zeros((noise_array.size, run_opts.reg_train_times.size), dtype=object)
        self.pred_out = np.zeros((noise_array.size, run_opts.reg_values.size, run_opts.reg_train_times.size),
                                 dtype=object)

    def save(self, run_opts, noise_array, res_itr, train_seed, test_idxs):
        """Saves the data in the ResOutput object to a series of .csv files.
        Args:
            self: ResOutput object
            run_opts: RunOpts object for test.
            noise_array: Array of noise/Jacobian/LMNT regularization parameter values used in this test.
            res_itr: Index for the reservoir iteration used.
            train_seed: Index for the training data iteration used.
            test_idxs: Indices for the testing data iterations used.
        Returns:
            Saves .csv files."""
        for (i, noise_val), (j, reg_train_time) in product(enumerate(noise_array), enumerate(run_opts.reg_train_times)):
            print((i, j))
            stable_frac = np.zeros(run_opts.reg_values.size)
            for k, array_elem in enumerate(self.stable_frac_out[i, :, j]):
                stable_frac[k] = array_elem
            np.savetxt(
                get_filename(run_opts.run_folder_name, 'stable_frac', res_itr, train_seed, noise_val, reg_train_time),
                stable_frac, delimiter=',')

            mean_rms = np.zeros((run_opts.reg_values.size, *self.mean_rms_out[i, 0, j].shape))
            for k, array_elem in enumerate(self.mean_rms_out[i, :, j]):
                mean_rms[k] = array_elem
            np.savetxt(
                get_filename(run_opts.run_folder_name, 'mean_rms', res_itr, train_seed, noise_val, reg_train_time),
                mean_rms, delimiter=',')

            max_rms = np.zeros((run_opts.reg_values.size, *self.max_rms_out[i, 0, j].shape))
            for k, array_elem in enumerate(self.max_rms_out[i, :, j]):
                max_rms[k] = array_elem
            np.savetxt(
                get_filename(run_opts.run_folder_name, 'max_rms', res_itr, train_seed, noise_val, reg_train_time),
                max_rms, delimiter=',')

            variances = np.zeros((run_opts.reg_values.size, *self.variances_out[i, 0, j].shape))
            for k, array_elem in enumerate(self.variances_out[i, :, j]):
                variances[k] = array_elem
            np.savetxt(
                get_filename(run_opts.run_folder_name, 'variance', res_itr, train_seed, noise_val, reg_train_time),
                variances, delimiter=',')

            valid_time = np.zeros((run_opts.reg_values.size, *self.valid_time_out[i, 0, j].shape))
            for k, array_elem in enumerate(self.valid_time_out[i, :, j]):
                valid_time[k] = array_elem
            for k in range(run_opts.num_tests):
                np.savetxt(
                    get_filename(run_opts.run_folder_name, 'valid_time', res_itr, train_seed, noise_val, reg_train_time,
                                 test_idx=test_idxs[k]), valid_time[:, k], delimiter=',')

            train_mean_rms = np.zeros((run_opts.reg_values.size, *self.train_mean_rms_out[i, 0, j].shape))
            for k, array_elem in enumerate(self.train_mean_rms_out[i, :, j]):
                train_mean_rms[k] = array_elem
            np.savetxt(get_filename(run_opts.run_folder_name, 'train_mean_rms', res_itr, train_seed, noise_val,
                                    reg_train_time),
                       train_mean_rms, delimiter=',')

            train_max_rms = np.zeros((run_opts.reg_values.size, *self.train_max_rms_out[i, 0, j].shape))
            for k, array_elem in enumerate(self.train_max_rms_out[i, :, j]):
                train_max_rms[k] = array_elem
            np.savetxt(
                get_filename(run_opts.run_folder_name, 'train_max_rms', res_itr, train_seed, noise_val, reg_train_time),
                train_max_rms, delimiter=',')

            if run_opts.pmap:
                pmap_max_wass_dist = np.zeros((run_opts.reg_values.size, *self.pmap_max_wass_dist_out[i, 0, j].shape))
                for k, array_elem in enumerate(self.pmap_max_wass_dist_out[i, :, j]):
                    pmap_max_wass_dist[k] = array_elem
                np.savetxt(get_filename(run_opts.run_folder_name, 'pmap_max_wass_dist', res_itr, train_seed, noise_val,
                                        reg_train_time),
                           pmap_max_wass_dist, delimiter=',')

                pmap_max = np.zeros((run_opts.reg_values.size, run_opts.num_tests), dtype=object)
                for k, l in product(np.arange(run_opts.reg_values.size), np.arange(run_opts.num_tests)):
                    pmap_max[k, l] = self.pmap_max_out[i, k, j][l]
                    pmap_len = [len(arr) for arr in pmap_max[k, l]]
                    max_pmap_len = max(pmap_len)
                    pmap_max_save = np.zeros((len(pmap_max[k, l]), max_pmap_len))
                    for m in range(len(pmap_max[k, l])):
                        pmap_max_save[m, :pmap_len[m]] = pmap_max[k, l][m]
                    np.savetxt(get_filename(run_opts.run_folder_name, 'pmap_max', res_itr, train_seed, noise_val,
                                            reg_train_time,
                                            test_idx=test_idxs[l], reg=run_opts.reg_values[k]),
                               pmap_max_save, delimiter=',')
            if run_opts.save_time_rms:
                rms = np.zeros((run_opts.reg_values.size, *self.rms_out[i, 0, j].shape))
                for k, array_elem in enumerate(self.rms_out[i, :, j]):
                    rms[k] = array_elem
                for k in range(run_opts.num_tests):
                    np.savetxt(
                        get_filename(run_opts.run_folder_name, 'rms', res_itr, train_seed, noise_val, reg_train_time,
                                     test_idx=test_idxs[k]),
                        rms[:, k], delimiter=',')
            if run_opts.save_eigenvals:
                eigenvals = self.grad_eigenvals_out[i, j]
                np.savetxt(get_filename(run_opts.run_folder_name, 'gradreg_eigenvals', res_itr, train_seed, noise_val,
                                        reg_train_time),
                           eigenvals, delimiter=',')
            if run_opts.savepred:
                pred = np.zeros((run_opts.reg_values.size, *self.pred_out[i, 0, j].shape))
                for k, array_elem in enumerate(self.pred_out[i, :, j]):
                    pred[k] = array_elem
                for l, (k, reg) in product(range(run_opts.num_tests), enumerate(run_opts.reg_values)):
                    np.savetxt(
                        get_filename(run_opts.run_folder_name, 'pred', res_itr, train_seed, noise_val, reg_train_time,
                                     test_idx=test_idxs[l], reg=reg),
                        pred[k, l], delimiter=',')


class NumericalModel:
    """Class for generating training or testing data using one of the test numerical models."""
    def __init__(self, tau=0.1, int_step=1, T=300, ttsplit=5000, u0=0, system='lorenz',
                 params=np.array([[], []], dtype=np.complex128), dnoise_gen = None, dnoise_scaling = 0):
        """Creates the NumericalModel object.
        Args:
            self: NumericalModel object.
            tau: Time step between measurements of the system dynamics.
            int_step: the number of numerical integration steps between each measurement.
            T: Total number of measurements.
            ttsplit: Number of measurements to be used in the training data set.
            u0: Initial condition for integration.
            system: Name of system to generate data from. Options: 'lorenz', 'KS','KS_d2175'
            params: Internal parameters for model integration. Currently only used by KS options.
        Returns:
            Complete NumericalModel object with precomputed internal parameters."""
        if system == 'lorenz':
            if isinstance(dnoise_gen, type(None)) or dnoise_scaling == 0:
                u_arr = np.ascontiguousarray(lorenzrungekutta(
                    u0, T, tau, int_step))
            else:
                dnoise = dnoise_gen.standard_normal((3, T*int_step+int_step))*np.sqrt(dnoise_scaling)
                u_arr = np.ascontiguousarray(lorenzrungekutta(
                    u0, T, tau, int_step, dnoise))
            self.input_size = 3

            u_arr[0] = (u_arr[0] - 0) / 7.929788629895004
            u_arr[1] = (u_arr[1] - 0) / 8.9932616136662
            u_arr[2] = (u_arr[2] - 23.596294463016896) / 8.575917849311919
            self.params = params

        elif system == 'KS':
            if isinstance(dnoise_gen, type(None)) or dnoise_scaling == 0:
                u_arr, self.params = kursiv_predict(u0, tau=tau, T=T, params=params, int_steps = int_step)
            else:
                dnoise = dnoise_gen.standard_normal((64, T*int_step+int_step))*np.sqrt(dnoise_scaling)
                u_arr, self.params = kursiv_predict(u0, tau=tau, T=T, params=params, int_steps=int_step,
                                                    noise= dnoise)
            self.input_size = u_arr.shape[0]
            u_arr = np.ascontiguousarray(u_arr) / (1.1876770355823614)
        elif system == 'KS_d2175':
            if isinstance(dnoise_gen, type(None)) or dnoise_scaling == 0:
                u_arr, self.params = kursiv_predict(u0, tau=tau, T=T, d=21.75, params=params, int_steps = int_step)
            else:
                dnoise = dnoise_gen.standard_normal((64, T * int_step + int_step)) * np.sqrt(dnoise_scaling)
                u_arr, self.params = kursiv_predict(u0, tau=tau, T=T, d=21.75, params=params,
                                                    int_steps=int_step, noise = dnoise)
            self.input_size = u_arr.shape[0]
            u_arr = np.ascontiguousarray(u_arr) / (1.2146066380280796)
        else:
            raise ValueError

        self.train_length = ttsplit
        self.u_arr_train = u_arr[:, :ttsplit + 1]

        # u[ttsplit], the (ttsplit + 1)st element, is the last in u_arr_train and the first in u_arr_test
        self.u_arr_test = u_arr[:, ttsplit:]


class ResPreds:
    """Class for loading and storing prediction time series data generated from reservoir computer tests."""
    def __init__(self, run_opts):
        """Loads the prediction data from .csv files.
        Args:
            run_opts: RunOpts object containing the run parameters."""
        self.data_filename, self.pred_folder = run_opts.save_file_name, run_opts.save_folder_name
        self.noise_vals = run_opts.noise_values_array
        self.reg_train_vals = run_opts.reg_train_times
        self.reg_vals = run_opts.reg_values
        print('Starding data read...')
        # print(self.pred_folder)
        self.preds = np.zeros((run_opts.res_per_test, run_opts.num_trains, run_opts.num_tests, self.noise_vals.size,
                               self.reg_train_vals.size, self.reg_vals.size), dtype=object)
        total_vals = self.preds.size
        with tqdm(total=total_vals) as pbar:
            for (i, res), (j, train), (k, test), (l, noise), (m, reg_train), (n, reg) in product(
                    enumerate(np.arange(run_opts.res_start, run_opts.res_start + run_opts.res_per_test)),
                    enumerate(np.arange(run_opts.train_start, run_opts.train_start + run_opts.num_trains)),
                    enumerate(np.arange(run_opts.test_start, run_opts.test_start + run_opts.num_tests)),
                    enumerate(self.noise_vals),
                    enumerate(self.reg_train_vals), enumerate(self.reg_vals)):
                filename = get_filename(self.pred_folder, 'pred', res, train, noise, reg_train, reg = reg, test_idx = test)
                if os.name == 'nt' and len(filename) >= 260:
                    filename = get_windows_path(filename)
                self.preds[i, j, k, l, m, n] = np.loadtxt(filename, delimiter=',')
                pbar.update(1)


class ResPmap:
    """Class for loading and storing Poincare maximum map data from reservoir predictions
    generated from reservoir computer tests."""
    def __init__(self, run_opts):
        """Loads the Poincare maxmimum map data from .csv files.
        Args:
            run_opts: RunOpts object containing the run parameters."""
        self.data_filename, self.pred_folder = run_opts.save_file_name, run_opts.save_folder_name
        self.noise_vals = run_opts.noise_values_array
        self.reg_train_vals = run_opts.reg_train_times
        self.reg_vals = run_opts.reg_values
        print('Starding data read...')
        # print(self.pred_folder)
        self.pmap_max = np.zeros((run_opts.res_per_test, run_opts.num_trains, run_opts.num_tests, self.noise_vals.size,
                                  self.reg_train_vals.size, self.reg_vals.size), dtype=object)
        total_vals = self.pmap_max.size
        with tqdm(total=total_vals) as pbar:
            for (i, res), (j, train), (k, test), (l, noise), (m, reg_train), (n, reg) in product(
                    enumerate(np.arange(run_opts.res_start, run_opts.res_start + run_opts.res_per_test)),
                    enumerate(np.arange(run_opts.train_start, run_opts.train_start + run_opts.num_trains)),
                    enumerate(np.arange(run_opts.test_start, run_opts.test_start + run_opts.num_tests)),
                    enumerate(self.noise_vals),
                    enumerate(self.reg_train_vals), enumerate(self.reg_vals)):
                filename = get_filename(self.pred_folder, 'pmap_max', res, train, noise, reg_train, reg = reg, test_idx = test)
                if os.name == 'nt' and len(filename) >= 260:
                    filename = get_windows_path(filename)
                pmap_in = np.loadtxt(filename, delimiter=',')
                pmap_max = [pmap_in[o, pmap_in[o] != 0.] for o in range(pmap_in.shape[0])]
                self.pmap_max[i, j, k, l, m, n] = pmap_max
                pbar.update(1)


class ResData:
    """Class for loading and storing prediction analysis data generated from reservoir computer tests."""
    def __init__(self, run_opts):
        """Loads the prediction analysis data from a compressed pandas data file.
            Args:
                run_opts: RunOpts object containing the run parameters."""
        self.data_filename, self.pred_folder = run_opts.save_file_name, run_opts.save_folder_name
        print('Starding data read...')
        tic = time.perf_counter()
        # print(self.data_filename)
        if os.name == 'nt' and len(self.data_filename) >= 260:
            self.data_filename = get_windows_path(self.data_filename)
        self.data = pd.read_csv(self.data_filename, index_col=0)
        toc = time.perf_counter()
        print('Data reading finished in %0.2f sec.' % (toc - tic))
        self.res = pd.unique(self.data['res'])
        self.train = pd.unique(self.data['train'])
        self.test = pd.unique(self.data['test'])
        self.noise = pd.unique(self.data['noise'])
        self.reg = pd.unique(self.data['reg'])
        self.reg_train = pd.unique(self.data['reg_train'])
        self.nan = pd.isna(self.data['variance'])

    def shape(self):
        """Returns the shape of the pandas data frame"""
        return self.data.shape

    def size(self):
        """Returns the size of the pandas data frame"""
        return self.data.size

    def data_slice(self, res=np.array([]), train=np.array([]), test=np.array([]),
                   reg_train=np.array([]), noise=np.array([]), reg=np.array([]), median_flag=False,
                   reduce_axes=[], metric='', gross_frac_metric='valid_time', gross_err_bnd=1e2,
                   reduce_fun=pd.DataFrame.median):
        """Slices and/or finds the best results using a metric computed by the reduce_fun.
        Args:
            self: ResPreds object
            res: Indices for reservoir results to be returned/optimized over. If left as an empty array, uses all indices.
            train: Indices for training data set results to be returned/optimized over. If left as an empty array, uses all indices.
            test: Indices for testing data set results to be returned/optimized over. If left as an empty array, uses all indices.
            reg_train: Number of training data points for regularization to be returned/optimized over. If left as an empty array, uses all values.
            noise: Noise/Jacobian/LMNT regularization parameter value results to be returned/optimized over. If left as an empty array, uses all values.
            reg: Tikhonov regularization paramter value results to be returned/optimized over. If left as an empty array, uses all values.
            median_flag: Boolean indicating whether the data should be optimized.
            reduce_axes: List containing the axes to be optimized over. Elements can be 'res', 'train', 'test', 'reg_train', 'noise', or 'reg'.
            metric: Metric to be used to compute which parameters give the best result. Options are:
                'gross_frac': Lowest fraction of gross error
                'mean_rms': Lowest mean map error.
                'max_rms: Lowest maximum map error.
                'valid_time': Highest valid prediction time.'
            gross_frac_metrix: If using 'gross_frac' as a metric, this is the secondary metric that will be used if multiple parameters give equally good prediction results.
            gross_err_bnd: The cutoff in the mean map error above which predictions are considered to have gross error.
            reduce_fun: Function for computing the overall performance for a given set of parameters over many tests.
        Returns:
            Pandas DataFrame containing the sliced and optimized prediction results.
        Raises:
            ValueError: If any of the inputs are not recognized/incompatible."""
        input_list = [res, train, test, reg_train, noise, reg]
        name_list = np.array(['res', 'train', 'test', 'reg_train', 'noise', 'reg'])
        data_names = [name for name in self.data.columns if name not in name_list]
        base_list = [self.res, self.train, self.test, self.reg_train, self.noise, self.reg]
        slice_vals = np.zeros(len(input_list), dtype=object)
        if median_flag:
            if not isinstance(reduce_axes, list):
                print('reduce_axes must be a list.')
                return ValueError
            elif len(reduce_axes) == 0:
                print('median_flag is True, but no axes to compute the median over are specified.')
                raise ValueError
            elif not all(axis in ['res', 'train', 'test', 'noise', 'reg','reg_train'] for axis in reduce_axes) or \
                    len(reduce_axes) != len(set(reduce_axes)):
                print('reduce_axis max only contain res, train, test, noise, or reg.')
                raise ValueError
            elif metric not in ['', 'mean_rms', 'max_rms', 'valid_time', 'pmap_max_wass_dist', 'gross_frac']:
                print('Metric not recognized.')
                raise ValueError
            elif len(metric) == 0 and any(axis in ['noise', 'reg'] for axis in reduce_axes):
                print('Cannot reduce axes with unspecified metric.')
                raise ValueError

        for i, item in enumerate(input_list):
            if isinstance(item, np.ndarray):
                if item.size == 0:
                    slice_vals[i] = base_list[i]
                else:
                    slice_vals[i] = item
            elif isinstance(item, list):
                if len(item) == 0:
                    slice_vals[i] = self.item
                else:
                    slice_vals[i] = np.array(item)
            else:
                slice_vals[i] = np.array([item])

        sliced_data = self.data
        for name, slice_val in zip(name_list, slice_vals):
            sliced_data = sliced_data[sliced_data[name].isin(slice_val)]

        if not median_flag:
            return sliced_data
        elif median_flag:
            median_slice_data = pd.DataFrame()
            remaining_vars = [var not in reduce_axes for var in name_list[:3]]
            remaining_vars.extend([True, True, True])
            if np.all(remaining_vars):
                median_slice_data = sliced_data
                nans = pd.isna(median_slice_data['mean_rms'])
                near_nans = median_slice_data['mean_rms'] > gross_err_bnd
                median_slice_data['gross_count'] = np.zeros(median_slice_data.shape[0])
                median_slice_data['gross_frac'] = np.zeros(median_slice_data.shape[0])
                median_slice_data.loc[nans | near_nans, 'gross_count'] = 1.0
                median_slice_data.loc[nans | near_nans, 'gross_frac'] = 1.0
            else:
                total_vars = 1
                for slice_val in slice_vals[remaining_vars]:
                    total_vars *= slice_val.size
                for vars_set in product(*slice_vals[remaining_vars]):
                    row_dict = {}
                    reduced_sliced_data = sliced_data
                    for var, name in zip(vars_set, name_list[remaining_vars]):
                        row_dict[name] = var
                        reduced_sliced_data = reduced_sliced_data[reduced_sliced_data[name] == var]
                    if reduced_sliced_data.size == 0:
                        for name in data_names:
                            if 'variance' in name:
                                row_dict[name] = 0.
                            elif 'valid_time' not in name:
                                row_dict[name] = 1e10
                        row_dict['valid_time'] = 0.
                        row_dict['gross_count'] = 0.
                        row_dict['gross_frac'] = 1.0
                        row_dict['data_present'] = False
                    else:
                        nans = pd.isna(reduced_sliced_data['mean_rms'])
                        near_nans = reduced_sliced_data['mean_rms'] > gross_err_bnd
                        nan_count = reduced_sliced_data[nans | near_nans].shape[0]
                        nan_frac = nan_count / reduced_sliced_data.shape[0]
                        valid_times = np.array([])
                        for name in data_names:
                            if 'variance' in name:
                                row_dict[name] = reduce_fun(reduced_sliced_data.loc[~nans, name])
                            elif 'valid_time' not in name:
                                reduced_sliced_data.loc[nans, name] = 1e10
                                row_dict[name] = reduce_fun(reduced_sliced_data[name])
                            else:
                                reduced_sliced_data.loc[pd.isna(reduced_sliced_data[name]), name] = 0.0
                                valid_times = np.append(valid_times, reduced_sliced_data[name].to_numpy())
                        row_dict['valid_time'] = reduce_fun(pd.DataFrame(valid_times)).to_numpy()[0]
                        row_dict['gross_count'] = nan_count
                        row_dict['gross_frac'] = nan_frac
                        row_dict['data_present'] = True
                    for key in row_dict.keys():
                        if not isinstance(row_dict[key], list) and not isinstance(row_dict[key], np.ndarray):
                            row_dict[key] = [row_dict[key]]
                    median_slice_data = pd.concat([median_slice_data, pd.DataFrame(row_dict)], ignore_index=True,
                                                  verify_integrity=True)
            if len(metric) == 0:
                return median_slice_data
            else:
                best_median_slice_data = pd.DataFrame()
                remaining_vars = [var not in reduce_axes for var in name_list]
                total_vars = 1
                for slice_val in slice_vals[remaining_vars]:
                    total_vars *= slice_val.size
                for vars_set in product(*slice_vals[remaining_vars]):
                    row_dict = {}
                    best_reduced_slice_data = median_slice_data
                    for var, name in zip(vars_set, name_list[remaining_vars]):
                        row_dict[name] = var
                        best_reduced_slice_data = best_reduced_slice_data[best_reduced_slice_data[name] == var]
                    if metric == 'valid_time':
                        best_reduced_slice_data = best_reduced_slice_data[
                            best_reduced_slice_data[metric] == best_reduced_slice_data[metric].max()]
                    elif metric == 'gross_frac':
                        best_reduced_slice_data = best_reduced_slice_data[
                            best_reduced_slice_data[metric] == best_reduced_slice_data[metric].min()]
                        if len(best_reduced_slice_data.shape) != 1:
                            if gross_frac_metric == 'valid_time':
                                best_reduced_slice_data = best_reduced_slice_data[
                                    best_reduced_slice_data[gross_frac_metric] == best_reduced_slice_data[
                                        gross_frac_metric].max()]
                            else:
                                best_reduced_slice_data = best_reduced_slice_data[
                                    best_reduced_slice_data[gross_frac_metric] == best_reduced_slice_data[
                                        gross_frac_metric].min()]
                    else:
                        best_reduced_slice_data = best_reduced_slice_data[
                            best_reduced_slice_data[metric] == best_reduced_slice_data[metric].min()]
                    if len(best_reduced_slice_data.shape) != 1:
                        best_reduced_slice_data = best_reduced_slice_data[best_reduced_slice_data['noise'] ==
                                                                          best_reduced_slice_data['noise'].min()]
                    if len(best_reduced_slice_data.shape) != 1:
                        best_reduced_slice_data = best_reduced_slice_data[best_reduced_slice_data['reg'] ==
                                                                          best_reduced_slice_data['reg'].min()]
                    for key in best_reduced_slice_data.keys():
                        if not isinstance(best_reduced_slice_data[key], list) and not isinstance(
                                best_reduced_slice_data[key], np.ndarray):
                            best_reduced_slice_data[key] = [best_reduced_slice_data[key]]
                    best_median_slice_data = pd.concat([best_median_slice_data, best_reduced_slice_data],
                                                       ignore_index=True, verify_integrity=True)
                return best_median_slice_data
