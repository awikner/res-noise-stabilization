# trainer.py
from collections import Counter
import os
import socket
import sys
import time
import ray

#num_cpus = int(sys.argv[1])
print("Program started")
ray.init(address=os.environ["ip_head"])
print("Connected to ray cluster")

print("Nodes in the Ray cluster:")
print(ray.nodes())
print(ray.available_resources())

num_tasks = int(sys.argv[1])
num_cpus = int(ray.available_resources()['CPU'])//4


@ray.remote
def f():
    time.sleep(1)
    return socket.gethostbyname(socket.gethostname())

@ray.remote
def get_f(num_tasks = num_tasks):
    #return f.remote()
    return [f.remote() for _ in range(num_tasks)]


# The following takes one second (assuming that
# ray was able to access all of the allocated nodes).
for i in range(60):
    start = time.time()
    ip_addresses = ray.get([get_f.remote() for _ in range(num_cpus)])
    flat_list = [item for sublist in ip_addresses for item in sublist]
    print(Counter(ray.get(flat_list)))
    #print(Counter(ray.get(ip_addresses)))
    end = time.time()
    print(end - start)
