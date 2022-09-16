import numpy as np
import os
import subprocess
import re
import time

def start_zaratan_run(system = 'KS', traintype = 'normal', noisetype = 'gaussian',
        noise_realizations = 1, res_size = 500, trainlen = 3000, testlen = 0, rho = 0.5, sigma = 1.0, theta = 1.0,
        leakage = 1.0, tau = 0.25, win_type = 'full', bias_type = 'old',
        noise_values_array = np.logspace(-3, 0, num = 19, base = 10)[5:11],
        alpha_values = np.append(0., np.logspace(-7, -3, 9)), num_res = 3,
        num_trains = 4, num_tests = 4, metric = 'mss_var', machine = 'zaratan',
        returnall = False, savepred = False, squarenodes = False, savetime = False, max_valid_time = 500,
        debug = False, num_nodes = 4, cpus_per_node = None, runtime = '2:00:00',
        account = 'physics-hi',debug_part = False, just_process = False, parallel = True,
        res_start = 0, train_start = 0, test_start = 0, reg_train_times = None, discard_time = 500, prior = 'zero',
        save_eigenvals = False, pmap = False, savetruth = False, sync_time = 2000, dyn_noise = 0,
        root_folder = "/scratch/zt1/project/edott-prj/user/awikner1/res-noise-stabilization",
        save_folder = "/scratch/zt1/project/edott-prj/user/awikner1/res-noise-stabilization"):

    if isinstance(reg_train_times, np.ndarray) or isinstance(reg_train_times, list):
        reg_train_times_str = '%d' % reg_train_times[0]
        for reg_train_time in reg_train_times[1:]:
            reg_train_times_str += ',%d' % reg_train_time
    else:
        reg_train_times_str = 'None'

    noise_values_str = '%e' % noise_values_array[0]
    for noise in noise_values_array[1:]:
        noise_values_str += ',%e' % noise

    reg_values_str = '%e' % alpha_values[0]
    for reg in alpha_values[1:]:
        reg_values_str += ',%e' % reg

    if isinstance(cpus_per_node, int):
        cpus_str = '--cpus-per-node=%d' % cpus_per_node
    else:
        cpus_str = ''

    if savetruth:
        savetruth_str = 'True'
    else:
        savetruth_str = 'False'

    if debug_part:
        debug_part_str = '-p debug'
    else:
        debug_part_str = ''

    if savepred:
        savepred_str = 'True'
    else:
        savepred_str = 'False'

    if returnall:
        returnall_str = 'True'
    else:
        returnall_str = 'False'
    if save_eigenvals:
        save_eigenvals_str = 'True'
    else:
        save_eigenvals_str = 'False'

    if debug:
        debug_str = 'True'
    else:
        debug_str ='False'

    if savetime:
        savetime_str = 'True'
    else:
        savetime_str = 'False'

    if parallel:
        parallel_str = 'True'
        parallel_str_bash = 'true'
    else:
        parallel_str = 'False'
        parallel_str_bash = 'false'

    if squarenodes:
        squarenodes_str = 'True'
    else:
        squarenodes_str = 'False'

    if pmap:
        pmap_str = 'True'
    else:
        pmap_str = 'False'



    testname = '%s_%s_%s_%d_%dnodes_%dtrain_rho%0.1f_sigma%0.1e_leakage%0.1f_tau%0.3f' % \
            (system, traintype, noisetype, noise_realizations, res_size, trainlen, rho, sigma, leakage,  tau)
    launch_script = "/scratch/zt1/project/edott-prj/user/awikner1/res-noise-stabilization/launch_files/slurm-launch-zaratan.py"
    run_script    = "/scratch/zt1/project/edott-prj/user/awikner1/res-noise-stabilization/src/res_reg_lmnt_awikner/reservoir_train_test.py"
    options_str = '--savepred=%s --system=%s --noisetype=%s --traintype=%s -r %d --rho=%f --sigma=%f --theta=%f --leakage=%f --win_type=%s --bias_type=%s --tau=%f -N %d -T %d --testtime=%d --res=%d --tests=%d --trains=%d --debug=%s --squarenodes=%s --metric=%s --returnall=%s --savetime=%s --noisevals=%s --regvals=%s --maxvt=%d --machine=%s --parallel=%s --resstart=%d --trainstart=%d --teststart=%d --regtraintimes=%s --discardlen=%d --prior=%s --saveeigenvals=%s --pmap=%s --datarootdir=%s --datasavedir=%s --savetruth=%s --synctime=%d'\
                  % (savepred_str, system, noisetype, traintype, noise_realizations,  rho, sigma, theta, leakage, win_type, bias_type, tau, res_size, trainlen, testlen, num_res, num_tests, num_trains, debug_str, squarenodes_str, metric, returnall_str, savetime_str, noise_values_str, reg_values_str, max_valid_time, machine, parallel_str, res_start, train_start, test_start, reg_train_times_str, discard_time, prior,save_eigenvals_str, pmap_str, root_folder, save_folder, savetruth_str, sync_time)
    input_str = 'python -u %s --ifray %s --exp-name %s --command "python -u %s %s" --num-nodes %d %s --load-env "conda activate res39" -t %s -A %s %s'\
                % (launch_script, parallel_str_bash, testname, run_script, options_str, num_nodes, cpus_str, runtime, account, debug_part_str)
    print(input_str)
    run_out = subprocess.check_output(input_str, shell=True)
    run_out_str = str(run_out)
    print(run_out_str)
    time.sleep(10)

    log_file = re.search('log_files/(.*).log', run_out_str)
    time_str = log_file.group(1)[-11:]
    print(time_str)
    print(run_out_str)

    job_group = re.search('Submitted batch job(.*)', run_out_str)
    job_id = job_group.group(1)
    if just_process:
        os.system('scancel %s' % job_id)
    template = open('/scratch/zt1/project/edott-prj/user/awikner1/res-noise-stabilization/src/res_reg_lmnt_awikner/process_test_data.py', 'r')
    lines = template.readlines()
    template.close()

    script_name = '/scratch/zt1/project/edott-prj/user/awikner1/res-noise-stabilization/data_scripts/process_test_data_%s.py' % job_id
    job_name = '%s_%s_process_data' % (testname, time_str)
    log_name = '/scratch/zt1/project/edott-prj/user/awikner1/res-noise-stabilization/log_files/%s.log' % job_name
    script = open(script_name, 'w')
    for line in lines:
        if '#SBATCH -t ' in line and debug_part:
            debug_part = False
            script.write('#SBATCH -p debug\n')
            script.write('#SBATCH -t 15:00\n')
        elif just_process and '{{JOB_ID}}' in line:
            pass
        else:
            script.write(line.replace('{{JOB_ID}}',job_id).replace('{{JOB_NAME}}', log_name).\
                         replace('{{LOG_NAME}}', log_name).replace('{{ACCOUNT}}', account))
    script.close()

    os.system('sbatch %s %s' % (script_name, options_str))

    time.sleep(1)