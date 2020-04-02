#!/bin/python3, Author: Aaron Chan, September 26th, 2019

import yaml
import itertools
import argparse
import numpy as np
from tqdm import tqdm

def try_parse(s):
    # Return a number, otherwise string or boolean
    try:
        return float(s)
    except ValueError:
        if s == 'true':
            return True
        elif s == 'false':
            return False
        else:
            return s

def make_configs(baseline_config, output_folder, params, title):
    with open(baseline_config, "r") as f:
        baseline = yaml.load(f.read(), Loader=yaml.SafeLoader)

    # Split the names and values
    param_names = [p[0] for p in params]
    param_vals = [p[1:] for p in params]

    # Take the cartesian product of values for all possible combinations
    param_combinations = list(itertools.product(*param_vals))

    i = 0
    padding = len(str(np.prod([len(x) for x in param_vals])))
    for p in tqdm(param_combinations):
        for setting in zip(param_names, p):
            # Set the name/value in the yaml configuration for each setting in this combination
            baseline[setting[0]] = setting[1]

        # Create output file name
        output_id = str(i).rjust(padding, '0')
        output_fname = f"{output_folder}/{title}{output_id}.yaml"

        # Write output yaml configuration
        with open(output_fname, "w") as f:
            yaml.dump(baseline, f, default_flow_style=False)
        i += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create spinoffs of a baseline config with certain parameters modified")
    parser.add_argument("baseline_config", type=str, help="Baseline config to spin off from")
    parser.add_argument("output_folder", type=str, help="Output folder for configuration files")
    parser.add_argument("--title", type=str, default="generated-config")
    parser.add_argument("--param", action="append", nargs="*")
    args = parser.parse_args()

    params = [ [try_parse(x) for x in p] for p in args.param ]
    make_configs(args.baseline_config, args.output_folder, params, args.title)
