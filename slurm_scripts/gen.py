import os
import sys
from subprocess import Popen
from pathlib import Path
import socket

# GLOBALS
EMAIL = "taylornarchibald@gmail.com" # mikebbrodie@gmail.com
MEM = 100000

## Setup paths
group_path = Path("/panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr")
proj_dir = Path(group_path / "taylor_simple_hwr") # the "project" directory
sh_proj_dir = Path(group_path / "taylor_simple_hwr") # the "project" directory

# If running locally, setup path for files to be written to
if socket.gethostname() in "Galois":
    proj_dir = Path("/media/data/GitHub/simple_hwr")
    assert proj_dir.exists()

# Paths relative to project directory
src_dir = proj_dir                                # the dir containing the python script to be summoned in the .sh file
config_root = proj_dir / "configs"                # the config root folder
sh_root = proj_dir / "slurm_scripts/scripts"              # the .sh root folder


# Environment
env = '/panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/env/hwr4_env'

hardware_dict = {
    "default":{"threads":8, "time":"72:00:00", "gpu":"pascal"},
    }

def get_sh(path, ext=".sh"):
    for ds,s,fs in os.walk(path):
        if ds.lower() in ("old", "~archive"):
            continue
        for f in fs:
            if f[-len(ext):] == ext:
                yield os.path.join(ds,f)


def is_iterable(object, string_is_iterable=True):
    """Returns whether object is an iterable. Strings are considered iterables by default.

    Args:
        object (?): An object of unknown type
        string_is_iterable (bool): True (default) means strings will be treated as iterables
    Returns:
        bool: Whether object is an iterable

    """

    if not string_is_iterable and type(object) == type(""):
        return False
    try:
        iter(object)
    except TypeError as te:
        return False
    return True
def mkdir(paths, parent=False):
    if not is_iterable(paths):
        paths = paths,
    for path in paths:
        if parent:
            path, _ = os.path.split(path)
        if path is not None and len(path) > 0 and not os.path.exists(path):
            os.makedirs(path)

def gen(sh_path, log_path, env, command, hardware_dict, cd_path=None):
    time = hardware_dict["time"]
    threads = hardware_dict["threads"]
    mem = f"{int( MEM / threads )}MB"
    if cd_path is None:
        cd_path,_ = os.path.split(sh_path)
    mkdir(sh_path, parent=True)
    print(f"Writing {sh_path}")
    with open(f"{sh_path}", "w") as f:
            f.write(f"""#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -C 'rhel7&pascal'
#SBATCH --mem-per-cpu {mem}
#SBATCH --ntasks {threads}
#SBATCH --output="{log_path}"
#SBATCH --time {time}
#SBATCH --mail-user={EMAIL}   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#%Module

module purge
module load cuda/10.1
module load cudnn/7.6

export PATH="{env}:$PATH"
eval "$(conda shell.bash hook)"
conda activate /panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/env/hwr4_env

cd "{cd_path}"
which python
{command}
""")

def loop_configs(config_root, script_root, ext=".yaml"):
    p = Path(config_root)
    print(p)
    for config_path in p.rglob(f"*.yaml"):  # use rglob for recursion
        # Define paths
        subfolders = config_path.relative_to(p)  # = 'c/d'
        sh_path = Path(script_root / subfolders.parent / config_path.with_suffix('.sh').name)
        #log_path = Path(sh_path.parent / config_path.parent / ('log_'+ config_path.with_suffix('.slurm').name))
        log_path = Path(script_root / subfolders.parent / ('log_' + config_path.with_suffix('.slurm').name))
        cd_path = src_dir

        py_script = "train_stroke_recovery.py" if "stroke_config" in config_path.as_posix() else "recognition/train.py"

        command = f"python -u {py_script} --config '{config_path}'"
        #print(f"sh:{sh_path} log:{log_path} cd: {cd_path}")
        if socket.gethostname() == "Galois":
            log_path = sh_proj_dir / log_path.relative_to(proj_dir)
            cd_path = sh_proj_dir / cd_path.relative_to(proj_dir)
        gen(sh_path=sh_path, log_path=log_path, env=env, command=command, hardware_dict=hardware_dict["default"], cd_path=cd_path)

def loop_exp(config_folder):
    # Loop over all experiment variations
    pass

def delete_old_sh(path="."):
    for sh in get_sh(path):
        os.remove(sh)

if __name__=="__main__":
    delete_old_sh()
    loop_configs(config_root, sh_root)

    # Make scripts executable                            
    Popen('find . -type f -iname "*.sh" -exec chmod +x {}  \;', shell=True)
