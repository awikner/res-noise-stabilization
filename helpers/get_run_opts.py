import warnings
from itertools import product
import sys
import getopt
import os

import numpy as np

class RunOpts:
    def __init__(self, argv=None,\
        runflag=True,\
        train_time = 20000,\
        test_time = None,\
        sync_time = 2000,\
        discard_time = 500,\
        res_size = 500,\
        res_per_test = 1,\
        noise_realizations = 1,\
        num_tests = 1,\
        num_trains = 1,\
        traintype = 'normal',\
        noisetype = 'gaussian',\
        system = 'KS',\
        savepred = False,\
        save_time_rms = False,\
        squarenodes = True,\
        rho = 0.6,\
        sigma = 0.1,\
        leakage = 1.0,\
        bias_type = 'new_random',\
        win_type = 'full_0centered',\
        debug_mode = False,\
        pmap = False,\
        machine = 'deepthought2',\
        ifray = False,\
        tau_flag = True,\
        num_cpus = 1,\
        metric = 'mean_rms',\
        return_all = True,\
        save_eigenvals = False,\
        max_valid_time = 2000,\
        noise_streams_per_test = 1,\
        noise_values_array = np.logspace(-4, 3, num = 3, base = 10),\
        reg_values = np.append(0., np.logspace(-11, -9, 5)),\
        res_start = 0,\
        train_start = 0,\
        test_start = 0,\
        import_res = False,\
        import_train = False,\
        import_test = False,\
        import_noise = False,\
        reg_train_times = None,\
        root_folder = None,\
        prior = 'zero'):
        self.train_time = train_time
        self.test_time = test_time
        self.sync_time = sync_time
        self.discard_time = discard_time
        self.res_size = res_size
        self.res_per_test = res_per_test
        self.noise_realizations = noise_realizations
        self.num_tests = num_tests
        self.num_trains = num_trains
        self.traintype = traintype
        self.noisetype = noisetype
        self.system = system
        self.savepred = savepred
        self.save_time_rms = save_time_rms
        self.squarenodes = squarenodes
        self.rho = rho
        self.sigma = sigma
        self.leakage = leakage
        self.bias_type = bias_type
        self.win_type = win_type
        self.debug_mode = debug_mode
        self.pmap = pmap
        self.machine = machine
        self.ifray = ifray
        self.tau_flag = tau_flag
        self.num_cpus = num_cpus
        self.metric = metric
        self.return_all = return_all
        self.save_eigenvals = save_eigenvals
        self.max_valid_time = max_valid_time
        self.noise_streams_per_test = noise_streams_per_test
        self.noise_values_array = noise_values_array
        self.reg_values = reg_values
        self.res_start = res_start
        self.train_start = train_start
        self.test_start = test_start
        self.import_res = import_res
        self.import_train = import_train
        self.import_test = import_test
        self.import_noise = import_noise
        self.prior = prior
        self.reg_train_times = reg_train_times
        self.root_folder = root_folder
        if not isinstance(argv, type(None)):
            self.get_run_opts(argv, runflag)
        if isinstance(self.test_time, type(None)):
            if self.system == 'lorenz':
                self.test_time = 4000
            elif 'KS' in self.system:
                self.test_time = 16000
        if not isinstance(self.reg_train_times, np.ndarray):
            if isinstance(self.reg_train_times, type(None)):
                self.reg_train_times = np.array([self.train_time])
            elif isinstance(self.reg_train_times, int):
                self.reg_train_times =np.array([self.reg_train_times])
            else:
                raise TypeError()
        if isinstance(self.root_folder, type(None)):
            self.root_folder = os.getcwd()
        if isinstance(self.reg_train_times, np.ndarray) or isinstance(self.reg_train_times, list):
            if (self.reg_train_times[0] != self.train_time or len(self.reg_train_times) != 1) and (self.traintype in ['normal','normalres1','normalres2','rmean','rmeanres1',\
                'rmeanres2','rplusq','rplusqres1','rplusqres2'] or 'confined' in self.traintype):
                print('Traintypes "normal", "rmean", and "rplusq" are not compatible with fractional regularization training.')
                raise ValueError
        if self.prior not in ['zero','input_pass']:
            print('Prior type not recognized.')
            raise ValueError
        self.get_file_name(runflag)

    def get_file_name(self,runflag):
        if self.import_res:
            iresflag = '_ires'
        else:
            iresflag = ''
        if self.import_train:
            itrainflag = '_itrain'
        else:
            itrainflag = ''
        if self.import_test:
            itestflag = '_itest'
        else:
            itestflag = ''
        if self.import_noise:
            inoiseflag = '_inoise'
        else:
            inoiseflag = ''
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
        data_folder_base = os.path.join(self.root_folder, 'Data')
        if not os.path.isdir(data_folder_base):
            os.mkdir(data_folder_base)

        if not self.return_all:
            self.data_folder = os.path.join(data_folder_base,'%s_noisetest_noisetype_%s_traintype_%s' % (self.system, self.noisetype, self.traintype))
            self.run_name = '%s%s%s%s%s%s%s%s%s%s_rho%0.1f_sigma%1.1e_leakage%0.3f_win_%s_bias_%s_tau%0.2f_%dnodes_%dtrain_%dreals_noisetype_%s_traintype_%s%s_metric_%s' \
             % (self.system,predflag, timeflag, eigenval_flag, pmap_flag, squarenodes_flag, iresflag, itrainflag, itestflag, inoiseflag, self.rho, self.sigma, self.leakage, self.win_type, self.bias_type, self.tau, self.res_size, \
             self.train_time, self.noise_realizations, self.noisetype, self.traintype, prior_str, self.metric)
        elif self.return_all:
            self.data_folder = os.path.join(data_folder_base, '%s_noisetest_noisetype_%s_traintype_%s' % (self.system, self.noisetype, self.traintype))
            self.run_name = '%s%s%s%s%s%s%s%s%s%s_rho%0.1f_sigma%1.1e_leakage%0.3f_win_%s_bias_%s_tau%0.2f_%dnodes_%dtrain_%dreals_noisetype_%s_traintype_%s%s' % (self.system,predflag, timeflag, eigenval_flag, pmap_flag, squarenodes_flag, iresflag, itrainflag, itestflag, inoiseflag, self.rho, self.sigma, self.leakage, self.win_type, self.bias_type, self.tau,self.res_size, self.train_time, self.noise_realizations, self.noisetype, self.traintype, prior_str)

        if runflag:
            if not os.path.isdir(self.data_folder):
                os.mkdir(self.data_folder)
            if not os.path.isdir(os.path.join(self.data_folder, self.run_name + '_folder')):
                os.mkdir(os.path.join(self.data_folder, self.run_name + '_folder'))
        self.run_file_name = os.path.join(self.data_folder, self.run_name + '.bz2')
        self.run_folder_name = os.path.join(self.data_folder, self.run_name + '_folder')
        
    def get_run_opts(self, argv, runflag = True):

        if runflag:
            try:
                opts, args = getopt.getopt(argv, "T:N:r:",
                    ['testtime=', 'noisetype=', 'traintype=', 'system=', 'res=',
                    'tests=', 'trains=', 'savepred=', 'tau=', 'rho=',
                    'sigma=', 'leakage=', 'bias_type=', 'debug=', 'win_type=',
                    'machine=', 'num_cpus=', 'pmap=', 'parallel=', 'metric=','returnall=',
                    'savetime=', 'saveeigenvals=','noisevals=', 'regvals=', 'maxvt=', 'noisestreams=',
                    'resstart=','trainstart=','teststart=',
                    'squarenodes=', 'importres=','importtrain=',
                    'importtest=','importnoise=','regtraintimes=','discardlen=',
                    'prior=','synctime=','datarootdir='])
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
                elif opt == '--datarootdir':
                    self.root_folder = str(arg)
                    print('Root directory for data: %s' % self.root_folder)
                elif opt == '--synctime':
                    self.sync_time = int(arg)
                    print('Sync time: %d' %  self.sync_time)
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
                elif opt == '--importres':
                    if arg == 'True':
                        self.import_res = True
                    elif arg == 'False':
                        self.import_res = False
                    else:
                        raise ValueError
                    print('Importing reservoir from file: %s' % arg)
                elif opt == '--importtrain':
                    if arg == 'True':
                        self.import_train = True
                    elif arg == 'False':
                        self.import_train = False
                    else:
                        raise ValueError
                    print('Importing training data from file: %s' % arg)
                elif opt == '--importtest':
                    if arg == 'True':
                        self.import_test = True
                    elif arg == 'False':
                        self.import_test = False
                    else:
                        raise ValueError
                    print('Importing test data from file: %s' % arg)
                elif opt == '--importnoise':
                    if arg == 'True':
                        self.import_noise = True
                    elif arg == 'False':
                        self.import_noise = False
                    else:
                        raise ValueError
                    print('Importing noise from file: %s' % arg)
                elif opt == '--squarenodes':
                    if arg == 'True':
                        self.squarenodes = True
                    elif arg == 'False':
                        self.squarenodes = False
                    else:
                        raise ValueError
                    print('Square reservoir nodes: %s' % arg)
                elif opt == '--noisestreams':
                    self.noise_streams_per_test = int(arg)
                    print('Noise Streams per test: %d' % self.noise_streams_per_test)
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
                    if self.machine not in ['deepthought2', 'personal']:
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
                elif opt == '--leakage':
                    self.leakage = float(arg)
                    print('Leakage: %f' % self.leakage)
                elif opt == '--tau':
                    self.tau  = float(arg)
                    self.tau_flag = False
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
            if self.tau_flag:
                if self.system == 'lorenz':
                    self.tau = 0.1
                elif self.system in ['KS', 'KS_d2175']:
                    self.tau = 0.25
        else:
            self.train_time, self.res_size, self.noise_realizations, self.save_time_rms,\
                self.save_eigenvals, self.pmap, self.metric,\
                self.return_all, self.machine, self.rho, self.sigma, self.leakage,\
                self.tau, self.win_type, self.bias_type, self.res_per_test, \
                self.num_tests, self.num_trains, self.savepred, self.noisetype, \
                self.traintype, self.system, self.squarenodes, self.prior, \
                self.res_start, self.train_start, self.test_start, self.import_res,\
                self.import_train, self.import_test, self.import_noise, self.reg_train_times,\
                self.discard_time = argv
