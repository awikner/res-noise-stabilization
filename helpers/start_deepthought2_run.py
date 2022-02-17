import os
import subproccess
import re
import time
import sys

def start_deepthought2_run(system = 'KS', traintype = 'normal', noisetype = 'gaussian', \
        noise_realizations = 1, res_size = 500, trainlen = 3000, rho = 0.5, sigma = 1.0,\
        leakage = 1.0, tau = 0.25, win_type = 'full', bias_type = 'old',\
        noise_values_array = np.logspace(-3, 0, num = 19, base = 10)[5:11], \
        alpha_values = np.append(0., np.logspace(-7, -3, 9)), num_res = 3,\
        num_trains = 4, num_tests = 4, metric = 'mss_var', returnall = False, \
        savepred = False, savetime = False, max_valid_time = 500, debug = False, \
        num_nodes = 4, cpus_per_node = 20, runtime = '2:00:00', account = 'physics-hi',\
        debug_part = False, just_process = False):

    noise_values_str = '%e' % noise_values_array[0]
    for noise in noise_values_array[1:]:
        noise_values_str += ',%e' % noise

    reg_values_str = '%e' % alpha_values[0]
    for reg in alpha_values[1:]:
        reg_values_str += ',%e' % reg

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

    if debug:
        debug_str = 'True'
    else:
        debug_str ='False'

    if savetime:
        savetime_str = 'True'
    else:
        savetime_str = 'False'




    testname = '%s_%s_%s_%d_%dnodes_%dtrain_rho%0.1f_sigma%0.1e_leakage%0.1f_tau%0.3f' % \
            (system, traintype, noisetype, no, res_size, trainlen, rho, sigma, leakage,  tau)
    options_str = '--savepred=%s --system=%s --noisetype=%s --traintype=%s -r %d --rho=%f --sigma=%f --leakage=%f --win_type=%s --bias_type=%s --tau=%f -N %d -T %d --res=%d --tests=%d --trains=%d --debug=%s --metric=%s --returnall=%s --savetime=%s --noisevals=%s --regvals=%s --maxvt=%d' % (savepred_str, system, noisetype, traintype, no,  rho,sigma, leakage, win_type, bias_type, tau, res_size, trainlen, num_res, num_tests, num_trains, debug_str, metric, returnall_str, savetime_str,
            noise_values_str, reg_values_str, max_valid_time)
    input_str = 'python slurm-launch.py --exp-name %s --command "python -u climate_replication_test.py %s" --num-nodes %d --cpus-per-node=%d --load-env "conda activate reservoir-rls" -t %s -A %s %s' % (testname, options_str, num_nodes, cpus_per_node, runtime, account, debug_part_str)
    print(input_str)
    run_out = subprocess.check_output(input_str, shell=True)
    time.sleep(1)

    log_file = re.search('log_files/(.*).log', str(run_out))
    time_str = log_file.group(1)[-11:]

    job_id = str(run_out)[-11:-3]
    if just_process:
        os.system('scancel %s' % job_id)
    template = open('data_scripts/process_test_data.py', 'r')
    lines = template.readlines()
    template.close()

    script_name = 'data_scripts/process_test_data_%s.py' % job_id
    log_name = '%s_%s_process_data' % (testname, time_str)
    script = open(script_name, 'w')
    for line in lines:
        if '#SBATCH -t ' in line and debug_part:
            debug_part = False
            script.write('#SBATCH -p debug\n')
            script.write('#SBATCH -t 15:00\n')
        elif just_process and '{{JOB_ID}}' in line:
            pass
        else:
            script.write(line.replace('{{JOB_ID}}',job_id).replace('{{JOB_NAME}}', log_name))
    script.close()

    os.system('sbatch %s %s' % (script_name, options_str))

    time.sleep(1)
