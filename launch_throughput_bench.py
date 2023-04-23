import argparse
import subprocess as sp 
import random
from simple_slurm import Slurm

"""
Simply Slurm interface for launching throughput benchmarking
python create_dataset.py --num_images 2000 --num_gpus 50 --start_idx 4000 --split pretrain
"""

# parser = argparse.ArgumentParser()
# parser.add_argument('--num_gpus', default=1, type=int,
#     help="Number of GPUs to parallelize over (with slurm)")
# parser.add_argument('--start_idx', default=0, type=int,)
# parser.add_argument('--split', default="pretrain", type=str,
#     help="Which split to generate images for")


# args = parser.parse_args()

# bb: largest log bs
# backbones = { # em clusteer
#     # "tiny": 8, 
#     # "small": 8,
#     # "base": 7, 
#     # "large": 5,
#     "huge": 4, 
#     "giant": 3
# }

backbones = { # em12 cluster
    "tiny": 10, 
    "small": 10,
    "base": 9, 
    "large": 8,
    "huge": 8, 
    "giant": 6 
}


def main():

    for bb, max_bs in backbones.items():
        for i in range(1, max_bs+1):
            bs = 2**i

            slurm_fbp = Slurm(
                # array=range(args.num_gpus),
                cpus_per_task=4,
                # chdir="~/clevr/image_generation/",
                exclude=['em1'],
                gres=['gpu:1'],
                job_name='clevr',
                qos='low',
                output=f'slurm/swin_{bb}_bs{bs}_fbp_{Slurm.JOB_ID}.out',
                time="1-02:03:04",
            )
            slurm_van = Slurm(
                # array=range(args.num_gpus),
                cpus_per_task=4,
                # chdir="~/clevr/image_generation/",
                exclude=['em1'],
                gres=['gpu:1'],
                job_name='clevr',
                qos='low',
                output=f'slurm/swin_{bb}_bs{bs}_van_{Slurm.JOB_ID}.out',
                time="1-02:03:04",
            )
            port = random.randint(10000, 20000)
            cmd_fbp = f"bash run_thr.sh {bs} {bb} True {port}"
            port = random.randint(10000, 20000)
            cmd_van = f"bash run_thr.sh {bs} {bb} False {port}"
            slurm_fbp.sbatch(cmd_fbp)
            slurm_van.sbatch(cmd_van)
    # slurm.sbatch(f'bash base_create_dataset.sh {num_images_per_gpu} {Slurm.SLURM_ARRAY_TASK_ID}*{args.num_images} {args.split}')

if __name__ == "__main__":
    main()