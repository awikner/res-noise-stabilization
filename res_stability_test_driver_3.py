import os

noisetypes = ['gaussian'+str(i)+'step' for i in range(2,11)]
for noise in noisetypes:
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=%s --traintype=normal -r 3' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=%s --traintype=rmean -r 3' % noise)
    os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=%s --traintype=rplusq -r 3' % noise)

