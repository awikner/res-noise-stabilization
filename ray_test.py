#!/homes/awikner1/anaconda3/envs/reservoir-rls/bin/python -u
#Assume will be finished in no more than 18 hours
#SBATCH -t 5:00
#Launch on 12 cores distributed over as many nodes as needed
#SBATCH --ntasks=80
#SBATCH -N 4
#Assume need 6 GB/core (6144 MB/core)
#SBATCH --mail-user=awikner1@umd.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
import ray
import time

ray.init(num_cpus = 20)

@ray.remote
def f(i):
    time.sleep(1)
    return i
tic = time.perf_counter()
futures = [f.remote(i) for i in range(80)]
print(ray.get(futures))
toc = time.perf_counter()
runtime = toc - tic
print('Runtime: %f sec.' % runtime)
