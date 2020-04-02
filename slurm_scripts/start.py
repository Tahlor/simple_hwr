import os
import sys
from subprocess import Popen
from pathlib import Path
import argparse

def get_sh(path, ext=".sh"):
    for ds,s,fs in os.walk(path):
        if ds.lower() in ("old", "~archive"):
            continue
        for f in fs:
            if f[-len(ext):] == ext:
                yield os.path.join(ds,f)

def start_scripts():
    opts = sh_parser()
    sh_root = Path(opts.root)
    for y in get_sh(sh_root):
        if opts.keyword.lower() in y.lower():
            p = Popen(f'sbatch {y}', shell=True, close_fds=True)
            print(f'Launching {y}')
    #p.communicate(input='\n')


def sh_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=".", help='Path of sbatch files')
    parser.add_argument('--keyword', type=str, default="", help='Keyword to find in specified .sh filenames')
    opts = parser.parse_args()
    if opts.root == "." and opts.keyword=="":
        cont = input("OK to submit ALL slurm .sh job scripts? (Y/n) ")
        if cont.lower() != "y":
            sys.exit()
    return opts


if __name__=="__main__":
    start_scripts()
