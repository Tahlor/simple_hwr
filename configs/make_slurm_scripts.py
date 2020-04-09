import os
import sys
import argparse
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def get_config_paths(config_dir):
    """Gets a list of paths for config files in the given directory

    Paths are AUTOMATICALLY given the directory of this script (currently in root/configs).  That means that you should give the config directory as a relative path (e.g. blur instead of /home/user/Projects/simple_hwr/configs/blur).

    Since this is dependent on the directory of this script, it means you should be running this script on the supercomputer or whereever you plan to submit the scripts, so that the paths are consistent.

    Contact Aaron Chan if questions.
    """

    config_paths = []
    this_script_dir = os.path.dirname(os.path.realpath(__file__))
    for subdir, dirs, files in os.walk(config_dir):
        for fname in files:
            if fname.endswith(".yaml"):
                config_paths.append(f"{this_script_dir}/{subdir}/{fname}")
    return config_paths

def make_scripts(config_paths, email, time, threads, ngpus, gpu, env):
    scripts = []
    mem = int(64000/threads)


    for path in tqdm(config_paths):
        py_script = "train_stroke_recovery.py" if "stroke" in path else "train.py"

        dir_path = os.path.dirname(path) # will not work on Windows
        base_path = os.path.basename(path)
        fname = os.path.splitext(base_path)[0]
        log_path = f"{dir_path}/log_{fname}.slurm"
        command = f"python -u {py_script} --config {path}"

        script = f"""#!/bin/bash
#SBATCH --gres=gpu:{ngpus}
#SBATCH -C 'rhel7&pascal'
#SBATCH --mem {mem}
#SBATCH --ntasks {threads}
#SBATCH --output="{log_path}"
#SBATCH --time {time}
#SBATCH --mail-user={email}
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#%Module

module purge
module load cuda/10.1
module load cudnn/7.6

export PATH="{env}:$PATH"
cd "{ROOT_DIR}"
which python
{command}"""

        # Return each script with its file name attached for writing.
        scripts.append((script, fname))
    return scripts

def write_scripts(scripts, output_dir):
    for script, name in scripts:
        with open(f"{output_dir}/{name}.sh", "w") as f:
            f.write(script)

if __name__=="__main__":
    print("Please make sure that this script is run on the supercomputer (or whereever the slurm scripts will be submitted)")
    parser = argparse.ArgumentParser(description="Create slurm script per configuration in given configuration directory")
    parser.add_argument("config_dir", type=str, help="Directory of configuration (.yaml) files to create scripts for")
    parser.add_argument("output_dir", type=str, help="Directory (already existing) to place created slurm scripts")
    parser.add_argument("email", type=str, help="Email to notify when script is finished")
    parser.add_argument("--time", type=str, default="36:00:00", help="Time per script to run")
    parser.add_argument("--threads", type=int, default=6, help="Number of threads to use")
    parser.add_argument("--ngpus", type=int, default=1, help="Number of GPUs to request")
    parser.add_argument("--gpu", type=str, default="pascal", help="GPU type (?)")
    parser.add_argument("--env", type=str, default="/panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/env/hwr4_env", help="Execution environment")
    
    args = parser.parse_args()
    print("Creating scripts with settings:")
    print("\t", f"Email: {args.email}")
    print("\t", f"Time: {args.time}")
    print("\t", f"Threads: {args.threads}")
    print("\t", f"# GPUs: {args.ngpus}")
    print("\t", f"GPU: {args.gpu}")
    
    config_paths = get_config_paths(args.config_dir)
    scripts = make_scripts(config_paths, args.email, args.time, args.threads, args.ngpus, args.gpu, args.env)
    write_scripts(scripts, args.output_dir)
