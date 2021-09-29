import os

os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=%s --traintype=normal -r 1' % 'gradient')
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=%s --traintype=rplusq -r 6' % 'gradient_onestep')
os.system('sbatch optimizingAlpha_ray_recode.py --noisetype=%s --traintype=rplusq -r 6' % 'perturbation_onestep')

