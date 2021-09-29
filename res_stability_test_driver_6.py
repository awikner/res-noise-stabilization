import os
import numpy as np
base_noise = 2.15443469e-02
noise_mults = np.logspace(-3,0,num = 10, base = 10)
q_vals = 1/noise_mults
noise_vals = base_noise*noise_mults

    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=gaussian_onestep --traintype=rplusq%e --noise=%e -r 6 -N 100 -T 50' % (q_vals[i], noise_vals[i]))

