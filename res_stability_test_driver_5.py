import os
import numpy as np
trainlen = 2250
res_size = 1000
noise_reals_array = np.arange(2, 16, dtype = int)
for noise_reals in noise_reals_array:
    os.system('sbatch optimizingAlpha_ray_recode_KS.py --system=KS --noisetype=gaussian --traintype=rplusq -r %d -T %d -N %d' %(noise_reals, trainlen, res_size))

ks = np.arange(2, 11, dtype = int)
for k in ks:
    os.system('sbatch optimizingAlpha_ray_recode_KS.py --system=KS --noisetype=gaussian%sstep --traintype=rplusq -r %d -T %d -N %d' %(k, 10,trainlen, res_size))
    #os.system('sbatch optimizingAlpha_ray_recode_KS.py --system=KS --noisetype=gaussian --traintype=rplusq -r %d -T %d -N %d' %(noise_reals, trainlen, res_size))
    #os.system('sbatch optimizingAlpha_ray_recode_KS.py --system=KS --noisetype=gaussian --traintype=rmean -r %d -T %d -N %d' %(noise_reals, trainlen,   res_size))


    #os.system('sbatch optimizingAlpha_ray_recode_KS.py --system=KS --noisetype=gaussian_onestep --traintype=rplusq -r %d -T %d -N %d' %(noise_reals, trainlen,res_size))
#os.system('sbatch optimizingAlpha_ray_recode_KS.py --system=KS --noisetype=none --traintype=gradient -r %d -T %d -N %d' %(1, trainlen,  res_size))
#os.system('sbatch optimizingAlpha_ray_recode_KS.py --system=KS --noisetype=none --traintype=gradient12 -r %d -T %d -N %d' %(1, trainlen,  res_size))
#os.system('sbatch optimizingAlpha_ray_recode_KS.py --system=KS --noisetype=none --traintype=gradient2 -r %d -T %d -N %d' %(1, trainlen,res_size))
