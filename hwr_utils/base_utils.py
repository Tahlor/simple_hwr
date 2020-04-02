import torch
import numpy as np
from pathlib import Path
import os
import pickle
import re
import logging

logger = logging.getLogger("root."+__name__)

def to_numpy(tensor, astype="float64"):
    if isinstance(tensor,(torch.cuda.FloatTensor, torch.FloatTensor, torch.Tensor)):
        return tensor.detach().cpu().numpy().astype(astype)
    else:
        return tensor

def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError as te:
        return False


def increment_path(name="", base_path=".", make_directory=True, ignore="partial"):
    """

    Args:
        name: Base name of directory to be preceded by numeral (e.g. 2MyLog)
        base_path: Root directory
        make_directory: Make the directory
        ignore: Don't include files matching this pattern when finding max number
    Returns:

    """

    # Check for existence
    Path(base_path).mkdir(parents=True, exist_ok=True)
    n, npath = get_max_file(base_path, ignore=ignore)

    # Create
    logdir = Path(os.path.join(base_path, "{:02d}_{}".format(n + 1, name)))
    if make_directory:
        Path(logdir).mkdir(parents=True, exist_ok=True)
    return logdir

def get_max_file(path, ignore=None):
    """ Gets the file with the highest (first) number in the string, ignoring the "ignore" string
    Args:
        path (str): Folder to search
    Returns:

    """
    if ignore:
        filtered = [p for p in os.listdir(path) if not re.search(ignore, p)]
    else:
        filtered = os.listdir(path)
    numbers = [(int(re.search("^[0-9]+", p)[0]), p) for p in filtered if re.search("^[0-9]+", p)]
    n, npath = max(numbers) if numbers else (0, "")
    # print("Last File Version: {}".format(npath))
    return n, os.path.join(path, npath)


def unpickle_it(path):
    with open(path, 'rb') as f:
        dict = pickle.load(f)  # , encoding = 'latin-1'
    return dict

def pickle_it(obj, path):
    with open(path, 'wb') as f:
        dict = pickle.dump(obj, f)  # , encoding = 'latin-1'

def print_tensor(tensor):
    logger.info(tensor, tensor.shape)