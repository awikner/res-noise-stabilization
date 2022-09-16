# slurm-launch.py
# Usage:
# python slurm-launch.py --exp-name test \
#     --command "rllib train --run PPO --env CartPole-v0"

import argparse
import subprocess
import sys, os
import time

from pathlib import Path

folder_path = "/home/awikner1/scratch.edott-prj/res-noise-stabilization"
template_file = os.path.join(folder_path, "slurm-template-zaratan.sh")

JOB_NAME = "${JOB_NAME}"
NUM_NODES = "${NUM_NODES}"
NUM_GPUS_PER_NODE = "${NUM_GPUS_PER_NODE}"
PARTITION_OPTION = "${PARTITION_OPTION}"
COMMAND_PLACEHOLDER = "${COMMAND_PLACEHOLDER}"
GIVEN_NODE = "${GIVEN_NODE}"
LOAD_ENV = "${LOAD_ENV}"
RUNTIME = "${RUNTIME}"
ACCOUNT = "${ACCOUNT}"
MEMORY  = "${MEMORY}"
CPUS    = "${CPUS}"
PARTITION = "${PARTITION}"
SCRATCH = "${SCRATCH}"
IFRAY   = "${IFRAY}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ifray",
        "-r",
        type=str,
        default="true",
        help="Flag whether or not to use ray for parallelization.")
    parser.add_argument(
        "--runtime",
        "-t",
        type=str,
        default="15:00",
        help="Total runtime for the function.")
    parser.add_argument(
        "--account",
        "-A",
        type=str,
        default="physics-hi",
        help="Account to charge cluster time to.")
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="The job name and path to logging file (exp_name.log).")
    parser.add_argument(
        "--num-nodes",
        "-n",
        type=int,
        default=1,
        help="Number of nodes to use.")
    parser.add_argument(
        "--node",
        "-w",
        type=str,
        help="The specified nodes to use. Same format as the "
        "return of 'sinfo'. Default: ''.")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="Number of GPUs to use in each node. (Default: 0)")
    parser.add_argument(
        "--partition",
        "-p",
        type=str,
    )
    parser.add_argument(
        "--load-env",
        type=str,
        help="The script to load your environment ('module load cuda/10.1')")
    parser.add_argument(
        "--command",
        type=str,
        required=True,
        help="The command you wish to execute. For example: "
        " --command 'python test.py'. "
        "Note that the command must be a string.")
    parser.add_argument(
        "--cpus-per-node",
        type = int,
        default = 20,
        help = "The minimum number of CPUs per node.")
    parser.add_argument(
        "--tmp",
        type = int,
        default = 10240,
        help = "The minimum amount of scratch space per node.")
    args = parser.parse_args()

    if args.node:
        # assert args.num_nodes == 1
        node_info = "#SBATCH -w {}".format(args.node)
    else:
        node_info = ""

    job_name = "{}_{}".format(args.exp_name,
                              time.strftime("%m%d-%H%M%S", time.localtime()))
    if str(args.ifray) == "false":
        args.num_nodes = 1
        args.cpus_per_node = 1
        print('Number of nodes and cpus set to 1 due to ifray=false')

    partition_option = "#SBATCH --partition={}".format(
        args.partition) if args.partition else ""

    memory = str(128000)
    # ===== Modified the template script =====
    with open(template_file, "r") as f:
        text = f.read()
    text = text.replace(JOB_NAME, job_name)
    text = text.replace(NUM_NODES, str(args.num_nodes))
    text = text.replace(NUM_GPUS_PER_NODE, str(args.num_gpus))
    text = text.replace(PARTITION_OPTION, partition_option)
    text = text.replace(COMMAND_PLACEHOLDER, str(args.command))
    text = text.replace(LOAD_ENV, str(args.load_env))
    text = text.replace(GIVEN_NODE, node_info)
    text = text.replace(RUNTIME, args.runtime)
    text = text.replace(ACCOUNT, args.account)
    text = text.replace(MEMORY, memory)
    text = text.replace(CPUS, str(args.cpus_per_node))
    text = text.replace(SCRATCH, str(args.tmp))
    text = text.replace(IFRAY, str(args.ifray))
    text = text.replace(
        "# THIS FILE IS A TEMPLATE AND IT SHOULD NOT BE DEPLOYED TO "
        "PRODUCTION!",
        "# THIS FILE IS MODIFIED AUTOMATICALLY FROM TEMPLATE AND SHOULD BE "
        "RUNNABLE!")

    # ===== Save the script =====
    script_file = os.path.join(folder_path, "bash_scripts/{}.sh".format(job_name))
    with open(script_file, "w") as f:
        f.write(text)

    # ===== Submit the job =====
    print("Starting to submit job!")
    subprocess.Popen(["sbatch", script_file])
    print(
        "Job submitted! Script file is at: <{}>. Log file is at: <{}>".format(
            script_file, "log_files/{}.log".format(job_name)))
    sys.exit(0)