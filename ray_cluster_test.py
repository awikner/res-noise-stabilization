#!/homes/awikner1/anaconda3/envs/reservoir-rls/bin/python -u
#Assume will be finished in no more than 18 hours
#SBATCH -t 5:00
#Launch on 12 cores distributed over as many nodes as needed
#SBATCH --ntasks=10
#SBATCH -N 1
#Assume need 6 GB/core (6144 MB/core)
#SBATCH --mail-user=awikner1@umd.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
from collections import Counter
import socket
import time

import ray

ray.init()

print('''This cluster consists of
    {} nodes in total
    {} CPU resources in total
'''.format(len(ray.nodes()), ray.cluster_resources()['CPU']))

@ray.remote
def f():
    time.sleep(0.001)
    # Return IP address.
    return socket.gethostbyname(socket.gethostname())

object_ids = [f.remote() for _ in range(10000)]
ip_addresses = ray.get(object_ids)

print('Tasks executed')
for ip_address, num_tasks in Counter(ip_addresses).items():
    print('    {} tasks on {}'.format(num_tasks, ip_address))
