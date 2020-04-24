import itertools
import numbers
import socket
import argparse
import matplotlib.pyplot as plt
import torch
import re
import shutil
import time
import pickle
import yaml
import json
import os
import signal
from Bio import pairwise2
import numpy as np
import warnings
import glob
from pathlib import Path
from easydict import EasyDict as edict
from hwr_utils import visualize, string_utils, error_rates
from hwr_utils.stattrack import Stat, AutoStat, Counter
import traceback
from hwr_utils import hwr_logger
from subprocess import Popen, DEVNULL, STDOUT, check_output
from hwr_utils.base_utils import *
from torch.optim import lr_scheduler


def read_config(config):
    config = Path(config)
    log_print(config)
    if config.suffix.lower() == ".json":
        return json.load(config.open(mode="r"))
    elif config.suffix.lower() == ".yaml":
        return fix_scientific_notation(yaml.load(config.open(mode="r"), Loader=yaml.Loader))
    else:
        raise "Unknown Filetype {}".format(config)

_print = print

# def print(*args,**kwargs):
#     log_print(*args, **kwargs, print_statements=False)

def log_print(*args, print_statements=True):
    if print_statements:
        _print(" ".join([str(a) for a in args]))
    else:
        hwr_logger.logger.info(" ".join([str(a) for a in args]))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./configs/taylor.yaml", help='Path to the config file.')
    #parser.add_argument('--name', type=str, default="", help='Optional - special name for this run')

    opts = parser.parse_args()
    return opts
    # if "name" not in config.keys():
    #     config["name"] = opts.name
    # elif not config["name"] and opts.name:
    #     config["name"] = opts.name

def find_config(config_name, config_root="./configs"):
    # Correct config paths
    if os.path.isfile(config_name):
        return config_name

    found_paths = []
    for d,s,fs in os.walk(config_root):
        for f in fs:
            if config_name == f:
                found_paths.append(os.path.join(d, f))

    # Error handling
    if len(found_paths) == 1:
        return found_paths[0]
    elif len(found_paths) > 1:
        raise Exception("Multiple {} config were found: {}".format(config_name, "\n".join(found_paths)))
    elif len(found_paths) < 1:
        raise Exception("{} config not found".format(config_name))

def incrementer(root, base):
    new_folder = Path(root / base)
    increment = 0
    increment_string = ""

    while new_folder.exists():
        increment += 1
        increment_string = f"{increment:02d}" if increment > 0 else ""
        new_folder = Path(root / (base + increment_string))

    new_folder.mkdir(parents=True, exist_ok=True)
    return new_folder

hwr_defaults = {"load_path":False,
            "training_shuffle": False,
            "testing_shuffle": False,
            "test_only": False,
            "TESTING": False,
            "gpu_if_available": True,
            "SKIP_TESTING": False,
            "OVERFIT": False,
            "TEST_FREQ": 1,
            "SMALL_TRAINING": False,
            "images_to_load": None,
            "plot_freq": 50,
            "rnn_layers": 2,
            "nudger_rnn_layers": 2,
            "nudger_rnn_dimension": 512,
            "improve_image": False,
            "decoder_type" : "naive",
            "rnn_type": "lstm",
            "cnn": "default",
            "online_flag": True,
            "save_count": 0,
            "training_blur": False,
            "training_blur_level": 1.5,
            "training_random_distortions": False,
            "training_distortion_sigma": 6.0,
            "testing_blur": False,
            "testing_blur_level": 1.5,
            "testing_random_distortions": False,
            "testing_distortion_sigma": 6.0,
            "occlusion_size": None,
            "occlusion_freq": None,
            "logging": "info",
            "n_warp_iterations": 11,
            "testing_occlude": False,
            "testing_warp": False,
            "optimizer_type": "adam",
            "max_intensity": .4,
            "exclude_offline": False,
            "validation_jsons": [],
            "elastic_transform": False,
            "visdom_port": 9001,
            "test_freq": 1
            }

stroke_defaults = {"SMALL_TRAINING": False,
                   "TESTING": False,
                    "logging": "info",
                   "use_visdom": True,
                   "save_count": 0,
                    "coord_conv": False,
                   "data_root_fsl": "../hw_data/strokes",
                   "data_root_local":".",
                   "training_nn_loss": False,
                   "test_nn_loss": True,
                   "test_nn_loss_freq": 1,
                   "visdom_port": 9001,
                   "gpu_if_available": True,
                    "start_of_stroke_method":"normal",
                    "interpolated_sos": "normal",
                    "cumsum_window_size": 30,
                    "convolve_func": "conv_weight", # or conv_window
                    "model_name":"normal",
                    "dataset": {"img_height": 61,
                                "image_prep": "pil_with_distortion",
                                "num_of_channels": 1,
                                "include_synthetic": True,
                                "adapted_gt_path": None},
                    "coordconv_method": "y_abs",
                    "model": {"nHidden": 128, "num_layers": 2},
                    "reset_LR": True,
                    "load_optimizer": False,
                    }

def debugger(func):
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except(Exception) as e:
            traceback.print_exc()
            print(e)
            globals().update(locals())
    return wrapper

def load_config(config_path, hwr=True, testing=False, results_dir_override=None):
    config_path = Path(config_path)
    project_path = Path(os.path.realpath(__file__)).parent.parent.absolute()
    config_root = project_path / "configs"

    # Path was specified, but not found
    # if config_root not in config_path.absolute().parents:
    #     raise Exception("Could not find config!")

    # Try adding a suffix
    if config_path.suffix != ".yaml":
        config_path = config_path.with_suffix(".yaml")

    # Go search for it
    if not config_path.exists():
        print(f"{config_path} does not exist")
        config_path = find_config(config_path.name, config_root)

    config = edict(read_config(config_path))
    config["name"] = Path(config_path).stem  ## OVERRIDE NAME WITH THE NAME OF THE YAML FILE
    config["project_path"] = project_path
    config.counter = Counter()

    if results_dir_override:
        config.results_dir = results_dir_override

    if testing:
        config.TESTING = True

    defaults = hwr_defaults if hwr else stroke_defaults
    for k in defaults.keys():
        if k not in config.keys():
            config[k] = defaults[k]

    # Main output folder
    # If override specified

    if Path(config_path).stem=="RESUME":
        output_root = Path(config_path).parent
        experiment = config.experiment

        # Backup some stuff
        backup = incrementer(output_root, "backup")
        backup.mkdir(exist_ok=True, parents=True)
        print(backup)
        for f in itertools.chain(output_root.glob("*.json"),output_root.glob("*.log")):
            shutil.copy(str(f), backup)
        output_root = output_root.as_posix()
    elif results_dir_override or ("results_dir_override" in config and config.results_dir_override):
        experiment = Path(results_dir_override).stem
        output_root = results_dir_override
    # If using preloaded model AND override is not mentioned
    elif (config["load_path"] and "results_dir_override" not in config):
        _output = incrementer(Path(config["load_path"]).parent, "new_experiment") # if it has a load path, create a new experiment in that same folder!
        experiment = _output.stem
        output_root = _output.as_posix()
    else:
        try: # Try to recreate the structure in the config directory (if this is in the config directory)
            experiment = Path(config_path).absolute().relative_to(Path(config_root).absolute()).parent
            if str(experiment) == ".": # if experiment is in root directory, use the experiment specified in the yaml
                experiment = config["experiment"]
            output_root = os.path.join(config["output_folder"], experiment)

        except Exception as e: # Fail safe; just dump to "./output/CONFIG NAME"
            log_print(f"Failed to find relative path of config file {config_root} {config_path}")
            experiment = Path(config_path).stem
            output_root = os.path.join(config["output_folder"], experiment)

    # Use config folder to determine output folder
    config["experiment"] = str(experiment)
    log_print(f"Experiment: {experiment}, Results Directory: {output_root}")

    hyper_parameter_str='{}'.format(
         config["name"],
     )

    train_suffix = '{}-{}'.format(
        time.strftime("%Y%m%d_%H%M%S"),
        hyper_parameter_str)

    if config["SMALL_TRAINING"] or config["TESTING"]:
        train_suffix = "TEST_"+train_suffix

    config["full_specs"] = train_suffix

    # Directory overrides
    if 'results_dir' not in config.keys():
        config['results_dir']=os.path.join(output_root, train_suffix)
    if 'output_predictions' not in config.keys():
        config['output_predictions']=False
    if "log_dir" not in config.keys():
        config["log_dir"]=os.path.join(output_root, train_suffix)
    if "image_dir" not in config.keys():
        config["image_dir"] = os.path.join(config['results_dir'], "imgs")

    if hwr:
        config["image_test_dir"] = os.path.join(config["image_dir"], "test")
        config["image_train_dir"] = os.path.join(config["image_dir"], "train")

    # Create paths
    for path in [output_root] + [config[d] for d in config.keys() if "_dir" in d]:
        if path is not None and len(path) > 0 and not os.path.exists(path):
            Path(path).mkdir(parents=True, exist_ok=True)

    # Make a link to most recent run
    try:
        link = "./RECENT.lnk"
        old_link = "./RECENT.lnk2"
        if os.path.exists(old_link):
            os.remove(old_link)
        if os.path.exists(link):
            os.rename(link, old_link)
        symlink(config['results_dir'], link)
    except Exception as e:
        log_print("Problem with RECENT link stuff: {}".format(e))

    # Copy config to output folder
    #parent, child = os.path.split(config)
    try:
        shutil.copy(config_path, config['results_dir'])
    except Exception as e:
        log_print(f"Could not copy config file: {e}")

    if hwr:
        config = make_config_consistent_hwr(config)
    else:
        config = make_config_consistent_stroke(config)

    logger = hwr_logger.setup_logging(folder=config["log_dir"], level=config["logging"].upper())

    log_print(f"Effective logging level: {logger.getEffectiveLevel()}")
    log_print("Using config file", config_path)
    #log_print(json.dumps(config, indent=2))

    config["logger"] = logger

    config["stats"] = {}
    config = computer_defaults(config)

    if not config.gpu_if_available:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    #make_lower(config)
    return config

def is_fsl():
    return "byu.edu" in socket.gethostname()

def make_config_consistent_stroke(config):
    config.image_dir = Path(config.image_dir)

    config.coordconv_opts = {"zero_center":config.coordconv_0_center,
                             "method":config.coordconv_method}

    config.data_root = config.data_root_fsl if is_fsl() else config.data_root_local

    # if config.x_relative_positions not in (True, False):
    #     raise NotImplemented
    if config.TESTING:
        if "icdar" not in config.dataset_folder.lower():
            config.dataset_folder = "online_coordinate_data/8_stroke_vSmall_16"
        config.update_freq = 1
        config.save_freq = 1
        config.first_loss_epochs = 1 # switch to other loss fast
        config.test_nn_loss_freq = 3
        config.train_size = 35
        config.test_size = 35

        if not config.gpu_if_available:
            config.batch_size = 2
            config.train_size = 2
            config.test_size = 2

    if is_dalai():
        config.dataset_folder = "online_coordinate_data/3_stroke_vverysmallFull"

    ## Process loss functions
    config.all_losses = set()
    # for key in [k for k in config.keys() if "loss_fns" in k]:
    #     for i, loss in enumerate(config[key]):
    #         loss, coef = loss.lower().split(",")
    #
    #         # Don't use inconsistent losses
    #         if "interpolated" in config.interpolated_sos and loss=="ssl":
    #             warnings.warn("Ignoring SSL, since 'interpolated' start point method doesn't use it")
    #             del config[key][i]
    #         else:
    #             config[key][i] = (loss, float(coef))
    #             config.all_losses.add(loss)
    validate_and_prep_loss(config)

    # Update dataset params
    config.dataset.gt_format = config.gt_format
    config.dataset.batch_size = config.batch_size
    config.dataset.image_prep = config.dataset.image_prep.lower()

    # Include synthetic data
    if config.dataset.include_synthetic:
        if config.TESTING:
            config.dataset.extra_dataset = ["online_coordinate_data/MAX_stroke_vFullSynthetic100kFull/train_online_coords_sample.json"]
        else:
            config.dataset.extra_dataset = ["online_coordinate_data/MAX_stroke_vFullSynthetic100kFull/train_online_coords.json",
                                            "online_coordinate_data/MAX_stroke_vBoosted2_normal/train_online_coords.json",
                                            "online_coordinate_data/MAX_stroke_vBoosted2_random/train_online_coords.json",
                                            ]

    else:
        config.dataset.extra_dataset = []

    if "loaded" in config.dataset.image_prep:
        config.dataset.img_height = 60

    config.device = "cuda" if torch.cuda.is_available() and config.gpu_if_available else "cpu"
    return config

def make_config_consistent_hwr(config):
    if config["SMALL_TRAINING"] or config["TESTING"]:
        config["images_to_load"] = config["batch_size"]

    config["occlusion"] = config["occlusion_size"] and config["occlusion_freq"]
    if not config["testing_occlude"] and not config["testing_warp"]:
        config["n_warp_iterations"] = 0
    elif (config["testing_occlude"] or config["testing_warp"]) and config["n_warp_iterations"] == 0:
        config["n_warp_iterations"] = 7
        log_print("n_warp_iterations set to 0, changing to 11")

    if config["exclude_offline"]:
        training_data = "prepare_IAM_Lines/gts/lines/txt/training.json"
        if training_data in config["training_jsons"]:
            config["training_jsons"].remove(training_data)
            if not config.training_jsons:
                raise Exception("No training data -- check exclude_offline flag")

    # Training data should be a list
    print(type(config.training_jsons))

    if isinstance(config.training_jsons, str):
        config.training_jsons = [config.training_jsons]
    if isinstance(config.testing_jsons, str):
        config.testing_jsons = [config.testing_jsons]
    if isinstance(config.validation_jsons, str):
        config.validation_jsons = [config.validation_jsons]

    if not config["TESTING"]:
        wait_for_gpu()

    if config["style_encoder"] == "fake_encoder":
        config["detach_embedding"] = True
    else:
        config["detach_embedding"] = False

    if "scheduler_step" not in config.keys() or "scheduler_gamma" not in config.keys():
        config["scheduler_step"] = 1
        config["scheduler_gamma"] = 1

    if config["SMALL_TRAINING"]:
        config["plot_freq"] = 1

    # Removing online jsons if not using online
    for data_path in config["training_jsons"]:
        if "online" in data_path and not config["online_augmentation"]:
            log_print(f"Online flag not specified -- removing {data_path}")
            config["training_jsons"].remove(data_path)
            config["online_flag"] = False # turn off flag if no online data provided
        if not config.training_jsons:
            log_print("No training json files -- check online_augmentation flag")
            raise Exception("No training json files -- check online_augmentation flag")

    # Save images
    if config["improve_image"]:
        config["save_improved_images"] = True
        config["image_dir"] = os.path.join(config["results_dir"], "images")
        mkdir(config["image_dir"])

    config.device = "cuda" if torch.cuda.is_available() and config.gpu_if_available else "cpu"
    return config

def make_lower(config):
    exclusions=("experiment","name","output_folder","writer_id_pickles", "training_jsons",
                "testing_jsons", "training_root", "testing_root", "results_dir", "log_dir", "img_dir")
    for key,value in config.items():
        if isinstance(config[key], str) and key not in exclusions and "dir" not in key and "path" not in key:
            config[key]=value.lower()

def computer_defaults(config):
    if is_galois():
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    elif not is_taylor():
        config["use_visdom"]=False
    return config

def symlink(target, link_location):
    while True:
        try:
            os.symlink(target, link_location)
            break
        except FileExistsError:
            os.remove(link_location)

def get_computer():
    return socket.gethostname()

def is_galois():
    return get_computer() == "Galois"

def is_dalai():
    return get_computer() in ["DalaiLama", "TheServe"]

def is_taylor():
    return get_computer() in ("Galois", "brodie")

def no_gpu_testing():
    """ Quick test, no GPU

    Returns:

    """
    return is_dalai()

def choose_optimal_gpu(priority="memory"):
    import GPUtil
    log_print(GPUtil.getGPUs())
    if priority == "memory":
        best_gpu = [(x.memoryFree, -x.load, i) for i,x in enumerate(GPUtil.getGPUs())]
    else: # utilization
        best_gpu = [(-x.load, x.memoryFree, i) for i, x in enumerate(GPUtil.getGPUs())]

    try:
        best_gpu.sort(reverse=True)
        best_gpu = best_gpu[0][2]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(best_gpu)
        return best_gpu
    except:
        return None

def project_root(set_root=True):
    current_directory = Path(os.getcwd())
    rel_path = "."
    while current_directory and current_directory.stem != "handwriting-synthesis":
        current_directory = current_directory.parent
        rel_path += "/.."

    if set_root:
        os.chdir(current_directory.as_posix())
        return Path(".")
    else:
        return Path(rel_path)

def get_project_root():
    ROOT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
    while ROOT_DIR.name != "simple_hwr":
        ROOT_DIR = ROOT_DIR.parent
    return ROOT_DIR

def get_folder(folder="."):
    path = (project_root() / folder)
    if path.exists() and path.is_file():
        return path.as_posix()
    else:
        return (project_root() / folder).as_posix() + "/"


def get_gpu_utilization():
    import GPUtil
    GPUtil.showUtilization()
    GPUs = GPUtil.getGPUs()
    utilization = GPUs[0].load * 100  # memoryUtil
    memory_utilization = GPUs[0].memoryUtil * 100  #
    return utilization, memory_utilization

def wait_for_gpu():
    if get_computer() != "Galois":
        return

    ## Wait until GPU is available -- only on Galois
    utilization, memory_utilization = get_gpu_utilization()
    log_print(utilization)
    if memory_utilization > 40:
        warnings.warn("Memory utilization is high; close other GPU processes")

    while utilization > 45:
        log_print("Waiting 30 minutes for GPU...")
        time.sleep(1800)
        utilization = GPUtil.getGPUs()[0].load * 100  # memoryUtil
    torch.cuda.empty_cache()
    if memory_utilization > 40:
        pass
        # alias gpu_reset="kill -9 $(nvidia-smi | sed -n 's/|\s*[0-9]*\s*\([0-9]*\)\s*.*/\1/p' | sort | uniq | sed$

    return

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


def fix_scientific_notation(config):
    exp = re.compile("[0-9.]+e[-0-9]+")
    for key,item in config.items():
        #print(item, is_iterable(item, string_is_iterable=False))
        if type(item) is str and exp.match(item):
            config[key] = float(item)
    return config

def get_last_index(my_list, value):
    return len(my_list) - 1 - my_list[::-1].index(value)

def write_out(folder, fname, text):
    with open(os.path.join(folder, fname), "a") as f:
        f.writelines(text+"\n\n")

def validate_and_prep_loss(config):
    if not "pred_format" in config:
        config.pred_format = config.gt_format

    # Each should be the same desired_num_of_strokes
    assert len(config.gt_format) == len(config.gt_opts) or config.model_name != "normal"
    if len(config.gt_format) != len(config.pred_opts):
        warnings.warn("GT format does not match pred_opts")

    config.vocab_size = len(config.pred_format) # vocab size is the desired_num_of_strokes of the GT format

    # Process loss functions
    if "loss_fns" not in config.keys() and "loss_fns2" in config.keys():
        config["loss_fns"] = config["loss_fns2"]
        del config["loss_fns2"]
    if "loss_fns2" in config and not config.loss_fns2:
        del config["loss_fns2"]

    for loss_fn_group in [k for k in config.keys() if "loss_fns" in k]:  # [loss_fns, loss_fns2]
        for i, loss in enumerate(config[loss_fn_group]):  # [{name: , coef: } ...]
            indices = [config.gt_format.index(k) for k in loss["gts"]] # This will throw an error if the loss expected something not in the GT

            # Add to list used for AUTOSTATS
            if loss["name"] not in config.all_losses:
                config.all_losses.add(loss["name"])
            else:
                warnings.warn(f"{loss['name']} loss already added to stats")

            # Convert to a slice?? no
            loss["loss_indices"] = indices


            for subindex in "dtw_mapping_basis", "cross_entropy_indices":
                if subindex in loss.keys():
                    # Convert the strings to indexes in the GT list, only if config has them as strings
                    # e.g. gts=[x,y], dtw_mapping_basis=[x,y], =>
                    if isinstance(loss[subindex][0], str):
                        loss[subindex] = [config.gt_format.index(k) for k in loss[subindex]]

            if "subcoef" in loss.keys():
                subcoef = loss["subcoef"]
                if isinstance(loss["subcoef"], str):
                    subcoef = [float(s) for s in loss["subcoef"].split(",")]
                elif isinstance(loss["subcoef"], float):
                    subcoef = [subcoef]
                loss["subcoef"] = subcoef
                assert len(subcoef) == len(loss["gts"])

            if not "coef" in loss.keys():
                loss["coef"] = 1
            else:
                # COEF should be a number, subcoef a comma-separated list
                assert isinstance(loss["coef"], (int, float, complex))

            # Whether to include metric in backprop
            if loss_fn_group=="loss_fns_to_report":
                config[loss_fn_group][i]["monitor_only"] = True
                config[loss_fn_group][i]["coef"] = 0
            else:
                config[loss_fn_group][i]["monitor_only"] = False

    if not "loss_fns2" in config.keys():
        config.loss_fns2 = None

    # Add loss functions to report only
    if "loss_fns_to_report" in config:
        config.loss_fns += config.loss_fns_to_report
        if "loss_fns2" in config:
            config.loss_fns2 += config.loss_fns_to_report

    config.pred_relativefy = [i for i, x in enumerate(config.pred_opts) if x == "cumsum"]
    return config

class CharAcc:
    def __init__(self, char_to_idx):
        self.correct = np.zeros(len(char_to_idx.keys()))
        self.actual_counts = np.zeros(len(char_to_idx.keys()))
        self.false_positive = self.correct.copy() # thought letter was found, was not
        self.false_negative = self.correct.copy() # missed true classification

        char_to_idx = char_to_idx.copy()
        if min(char_to_idx.values())==1:
            for key, val in char_to_idx.items():
                char_to_idx[key] = val-1

        self.char_to_idx = char_to_idx
        self.char_to_idx["|"] = self.char_to_idx["-"]

    def char_accuracy(self, pred, gt):
        pred = pred.replace("-", "|")
        gt = gt.replace("-", "|")
        pred_algn, gt_algn, *_ = pairwise2.align.globalxx(pred, gt)[0]

        for i, c in enumerate(pred_algn):
            guess_char=pred_algn[i]
            true_char=gt_algn[i]
            self.actual_counts[self.char_to_idx[true_char]] += 1
            if guess_char == true_char:
                self.correct[self.char_to_idx[c]] += 1
            elif true_char == "-": # model inserted a letter incorrectly
                self.false_positive[self.char_to_idx[guess_char]] += 1
            elif guess_char == "-": # model missed letter
                self.false_negative[self.char_to_idx[true_char]] += 1
            elif guess_char != true_char: # model missed one letter, and incorrectly posited another
                self.false_positive[self.char_to_idx[guess_char]] += 1
                self.false_negative[self.char_to_idx[true_char]] += 1

def dict_to_list(d):
    idx_to_char = []
    for i in range(0, max(d.keys())+1):
        idx_to_char.append(d[i])
    return idx_to_char

def load_model(config):
    # User can specify folder or .pt file; other files are assumed to be in the same folder
    if os.path.isfile(config["load_path"]):
        old_state = torch.load(config["load_path"])
        path, child = os.path.split(config["load_path"])
    else:
        old_state = torch.load(os.path.join(config["load_path"], "baseline_model.pt"))
        path = config["load_path"]

    # Load the definition of the loaded model if it was saved
    # if "model_definition" in old_state.keys():
    #     config["model"] = old_state['model_definition']

    for key in ["idx_to_char", "char_to_idx"]:
        if key in old_state.keys():
            if key == "idx_to_char":
                old_state[key] = dict_to_list(old_state[key])
            config[key] = old_state[key]

    if "model" in old_state.keys():
        config["model"].load_state_dict(old_state["model"])
        if "optimizer" in config.keys():
            config["optimizer"].load_state_dict(old_state["optimizer"])
        config["global_counter"] = old_state["global_step"]
        config["starting_epoch"] = old_state["epoch"]
        config["current_epoch"] = old_state["epoch"]
    else:
        config["model"].load_state_dict(old_state)

    # Launch visdom
    if config["use_visdom"]:
        try:
            config["visdom_manager"].load_log(os.path.join(path, "visdom.json"))
        except:
            warnings.warn("Unable to load from visdom.json; does the file exist?")
            ## RECREATE VISDOM FROM FILE IF VISDOM IS NOT FOUND

    # Load Loss History
    stat_path = os.path.join(path, "all_stats.json")
    loss_path = os.path.join(path, "losses.json")

    if Path(loss_path).exists():
        with open(loss_path, 'r') as fh:
            losses = json.load(fh)
    else:
        print("losses.json not found in load_path folder")
    try:
        config["train_cer"] = losses["train_cer"]
        config["test_cer"] = losses["test_cer"]
        config["validation_cer"] = losses["validation_cer"]
    except:
        warnings.warn("Could not load from losses.json")
        config["train_cer"]=[]
        config["test_cer"] = []

    # Load stats
    try:
        with open(stat_path, 'r') as fh:
            stats = json.load(fh)
        # Update the counter - not implemented yet??
        counter = stats["counter"]
        config.counter.__dict__.update(counter)

        # Load stats
        stats = stats["stats"]
        for name, stat in config["stats"].items():
            if isinstance(stat, Stat):
                config["stats"][name].y = stats[name]["y"]
            else:
                for i in stats[name]: # so we don't mess up the reference etc.
                    config["stats"][name].append(i)

    except:
        warnings.warn("Could not load from all_stats.json")


def save_model_stroke(config, bsf=False):
    # Save the best model
    if bsf:
        path = os.path.join(config["results_dir"], "BSF")
        mkdir(path)
    else:
        path = config["results_dir"]

    #    'model_definition': config["model"],
    state_dict = {
        'epoch': config.counter.epochs,
        'model': config["model"].state_dict(),
        'optimizer': config["optimizer"].state_dict(),
        'global_step': config.counter.updates,
        'scheduler': config.scheduler.state_dict(),
        'batch_size': config.batch_size
    }

    # Make resume model be the BSF one
    #config["main_model_path"] = os.path.join(path, "{}_model.pt".format(config['name']))

    # Make resume model be the most recent
    config["main_model_path"] = os.path.join(config["results_dir"], "{}_model.pt".format(config['name']))

    torch.save(state_dict, config["main_model_path"])
    save_stats_stroke(config, bsf)

    # Save visdom
    if config["use_visdom"]:
        try:
            path = os.path.join(path, "visdom.json")
            config["visdom_manager"].save_env(file_path=path)
        except:
            warnings.warn(f"Unable to save visdom to {path}; is it started?")
            config["use_visdom"] = False

    # Copy BSF stuff to main directory
    if bsf:
        for filename in glob.glob(path + r"/*"):
            shutil.copy(filename, config["results_dir"])

    if config["save_count"]==0:
        create_resume_training_stroke(config)
    config["save_count"] += 1
    if "training_dataset" in config and "dtw_adaptive" in config.all_losses:
        np.save(Path(config["results_dir"]) / "training_dataset.npy", [{"gt":gt["gt"], "image_path":gt["image_path"]} for gt in config.training_dataset.data] )

def new_scheduler(optimizer, batch_size, gamma=.95, last_epoch=-1):
    print("Building new scheduler...")
    return lr_scheduler.StepLR(optimizer, step_size=int(180000 / batch_size), gamma=gamma, last_epoch=last_epoch)

def load_model_strokes(config, load_optimizer=True, device="cuda"):
    # User can specify folder or .pt file; other files are assumed to be in the same folder
    if os.path.isfile(config["load_path"]):
        old_state = torch.load(config["load_path"], map_location=torch.device(device)) # map_location=torch.device(config.device)
        path, child = os.path.split(config["load_path"])
    else:
        children = [f.name for f in Path(config["load_path"]).glob("*.pt")]
        if "baseline_model.pt" in children:
            old_state = torch.load(os.path.join(config["load_path"], "baseline_model.pt"), map_location=torch.device(device)) # map_location=torch.device(config.device)
        else:
            old_state = torch.load(os.path.join(config["load_path"], children[0]), map_location=torch.device(device))
            if len(children) > 1:
                warnings.warn(f"Multiple .pt files found, using {children[0]}")
        path = config["load_path"]
    logger.info(f"Loading MODEL from {path}")

    # Load the definition of the loaded model if it was saved
    # if "model_definition" in old_state.keys():
    #     config["model"] = old_state['model_definition']

    if "model" in old_state.keys():
        config["model"].load_state_dict(old_state["model"])
        if "optimizer" in config.keys() and load_optimizer:
            logger.info("Loading optimizer...")
            config.optimizer.load_state_dict(old_state["optimizer"])
            LR = next(iter(config.optimizer.param_groups))['lr']
            logger.info(f"LR from loaded optimizer is {LR}")
        config["global_counter"] = old_state["global_step"]
        config["starting_epoch"] = old_state["epoch"]
        config["current_epoch"] = old_state["epoch"]
    else:
        config["model"].load_state_dict(old_state)

    old_batch_size = old_state["batch_size"] if "batch_size" in old_state else config.batch_size

    # Load these counters -- will be overwritten if all_stats.json found
    config.counter.updates = config["global_counter"]
    config.counter.epochs = config["current_epoch"]

    logger.info(f"Total updates from loaded model: {config.global_counter}")
    logger.info(f"Epochs from loaded model: {config.current_epoch}")
    if "scheduler" in old_state and load_optimizer:
        print("Loading saved scheduler...")
        config.scheduler.load_state_dict(old_state["scheduler"])

    elif load_optimizer: # if there is no saved schedule state, rebuild it
        config.scheduler = new_scheduler(config.optimizer, old_batch_size, gamma=config.scheduler_gamma, last_epoch=config.counter.updates)

    # Launch visdom
    if config["use_visdom"]:
        try:
            config["visdom_manager"].load_log(os.path.join(path, "visdom.json"))
        except:
            warnings.warn("Unable to load from visdom.json; does the file exist?")
            ## RECREATE VISDOM FROM FILE IF VISDOM IS NOT FOUND

    # Load Stats History
    stat_path = os.path.join(path, "all_stats.json")

    # Load stats
    try:
        load_stats_stroke(config, stat_path)
    except:
        warnings.warn("Could not load from all_stats.json")


def mkdir(path):
    if isinstance(path, str):
        if path is not None and len(path) > 0 and not os.path.exists(path):
            try:
                os.makedirs(path)
            except Exception as e:
                print(e)
    elif isinstance(path, Path):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(e)
    else:
        warnings.warn("Unknown path type, cannot create folder")

def save_stats(config, bsf):
    if bsf:
        path = os.path.join(config["results_dir"], "BSF")
        mkdir(path)
    else:
        path = config["results_dir"]

    # Save all stats
    results = {"stats":config.stats, "counter": config.counter.__dict__}
    with open(os.path.join(path, "all_stats.json"), 'w') as fh:
        json.dump(results, fh, cls=EnhancedJSONEncoder, indent=4)

    # Save CER
    config.train_cer = config.stats[config["designated_validation_cer"]].y
    config.validation_cer = config.stats[config["designated_training_cer"]].y
    config.test_cer = config.stats[config["designated_test_cer"]].y

    #results = {"training":config["stats"][config["designated_training_cer"]], "test":config["stats"][config["designated_test_cer"]]}
    results = {"train_cer":config.train_cer, "validation_cer":config.validation_cer, "test_cer":config.test_cer}
    with open(os.path.join(path, "losses.json"), 'w') as fh:
        json.dump(results, fh, cls=EnhancedJSONEncoder, indent=4)

def save_stats_stroke(config, bsf):
    if bsf:
        path = os.path.join(config["results_dir"], "BSF")
        mkdir(path)
    else:
        path = config["results_dir"]

    # Save all stats
    results = {"stats":config.stats, "counter": config.counter.__dict__}
    with open(os.path.join(path, "all_stats.json"), 'w') as fh:
        json.dump(results, fh, cls=EnhancedJSONEncoder, indent=4)

def load_stats_stroke(config, stat_path):
    with open(stat_path, 'r') as fh:
        loaded_stats = json.load(fh)

    # Update the counter
    counter = loaded_stats["counter"]
    config.counter.__dict__.update(counter)

    loaded_stats = loaded_stats["stats"]
    for name, stat in config["stats"].items():
        if isinstance(stat, Stat):
            config["stats"][name].y = loaded_stats[name]["y"]
            config["stats"][name].x = loaded_stats[name]["x"]
        else:
            for i in loaded_stats[name]:  # so we don't mess up the reference etc.
                config["stats"][name].append(i)

def save_model(config, bsf=False):
    # Can pickle everything in config except items below
    # for x, y in config.items():
    #     print(x)
    #     if x in ("criterion", "visdom_manager", "trainer"):
    #         continue
    #     torch.save(y, "TEST.pt")

    # Save the best model
    if bsf:
        path = os.path.join(config["results_dir"], "BSF")
        mkdir(path)
    else:
        path = config["results_dir"]

    #    'model_definition': config["model"],
    state_dict = {
        'epoch': config["current_epoch"] + 1,
        'model': config["model"].state_dict(),
        'optimizer': config["optimizer"].state_dict(),
        'global_step': config["global_step"],
        "idx_to_char": config["idx_to_char"],
        "char_to_idx": config["char_to_idx"]
    }

    config["main_model_path"] = os.path.join(path, "{}_model.pt".format(config['name']))
    torch.save(state_dict, config["main_model_path"])

    if "nudger" in config.keys():
        state_dict["model"] = config["nudger"].state_dict()
        torch.save(state_dict, os.path.join(path, "{}_nudger_model.pt".format(config['name'])))

    save_stats(config, bsf)

    # Save visdom
    if config["use_visdom"]:
        try:
            path = os.path.join(path, "visdom.json")
            config["visdom_manager"].save_env(file_path=path)
        except:
            warnings.warn(f"Unable to save visdom to {path}; is it started?")
            config["use_visdom"] = False

    # Copy BSF stuff to main directory
    if bsf:
        for filename in glob.glob(path + r"/*"):
            shutil.copy(filename, config["results_dir"])

    if config["save_count"]==0:
        create_resume_training(config)
    config["save_count"] += 1

def create_resume_training_stroke(config):
    export_config = config.copy()
    export_config["load_path"] = config["main_model_path"]

    for key in config.keys():
        item = config[key]
        # Only keep items that are numbers, strings, and lists
        if item is not None \
                and not isinstance(item, (bool, list, dict, numbers.Number, str, edict)):
            del export_config[key]

    output = Path(config["results_dir"])
    with open(Path(output / 'RESUME.yaml'), 'w') as outfile:
        export_config["reset_LR"] = False # don't reset learning rate if loading from pretrained model
        export_config["load_optimizer"] = True # load the previous optimizer state
        if "adapted_gt_path" in config.dataset and config.dataset.adapted_gt_path:
            # export_config["load_path"] is probably BSF
            export_config["dataset"]["adapted_gt_path"] = Path(output) / "training_dataset.npy"
        yaml.dump(export_config, outfile, default_flow_style=False, sort_keys=False)

    with open(Path(output / 'TEST.yaml'), 'w') as outfile:
        yaml.dump(export_config, outfile, default_flow_style=False, sort_keys=False)

def create_resume_training(config):
    #export_config = copy.deepcopy(config)
    export_config = config.copy()
    export_config["load_path"] = config["main_model_path"]

    for key in config.keys():
        item = config[key]
        # Only keep items that are numbers, strings, and lists
        if not isinstance(item, str) \
                and not isinstance(item, numbers.Number) \
                and not isinstance(item, list) \
                and item is not None \
                and not isinstance(item, bool):
            del export_config[key]

    output = Path(config["results_dir"])
    with open(Path(output / 'RESUME.yaml'), 'w') as outfile:
        yaml.dump(export_config, outfile, default_flow_style=False, sort_keys=False)

    with open(Path(output / 'TEST.yaml'), 'w') as outfile:
        export_config["test_only"] = True
        if export_config["training_warp"]:
            export_config["testing_warp"] = True
        if export_config["max_intensity"]:
            export_config["testing_occlude"] = True
        if (export_config["testing_occlude"] or export_config["testing_warp"]) and not export_config["n_warp_iterations"]:
            export_config["n_warp_iterations"] = 21
        yaml.dump(export_config, outfile, default_flow_style=False, sort_keys=False)

def plt_loss(config):
    ## Plot with matplotlib
    try:
        x_axis = [(i + 1) * config["n_train_instances"] for i in range(len(config["train_cer"]))]
        plt.figure()
        plt.plot(x_axis, config["train_cer"], label='train')
        plt.plot(x_axis, config["validation_cer"], label='validation')
        if config["test_cer"]:
            plt.plot(config["test_epochs"], config["test_cer"], label='validation')
        plt.legend()
        plt.ylim(top=.2)
        plt.ylabel("CER")
        plt.xlabel("Number of Instances")
        plt.title("CER Loss")
        plt.savefig(os.path.join(config["results_dir"], config['name'] + ".png"))
        plt.close()
    except Exception as e:
        log_print("Problem graphing: {}".format(e))


class Decoder:
    def __init__(self, idx_to_char, beam=False):
        self.decode_training = self.decode_batch_naive
        self.decode_test = self.decode_batch_naive
        self.idx_to_char = idx_to_char

        if beam:
            from ctcdecode import CTCBeamDecoder
            log_print("Using beam")
            self.beam_decoder = CTCBeamDecoder(labels=idx_to_char.values(), blank_id=0, beam_width=30, num_processes=3, log_probs_input=True)
            self.decode_test = self.decode_batch_beam

    def decode_batch_naive(self, out, as_string=True):
        out = out.data.cpu().numpy()
        for j in range(out.shape[0]):
            logits = out[j, ...]
            pred, raw_pred = string_utils.naive_decode(logits)
            if as_string:
                yield string_utils.label2str(pred, self.idx_to_char, False)
            else:
                yield pred

    def decode_batch_beam(self, out, as_string=True):
        pred, scores, timesteps, out_seq_len = self.beam_decoder.decode(out)
        pred = pred.data.int().numpy()
        output_lengths = out_seq_len.data.data.numpy()
        rank = 0 # get top ranked prediction
        # Loop through batches
        for batch in range(pred.shape[0]):
            line_length = output_lengths[batch][rank]
            line = pred[batch][rank][:line_length]
            string = u""
            if as_string:
                for char in line:
                    string += self.idx_to_char[char]
                yield string
            else:
                yield line

def calculate_cer(pred_strs, gt):
    sum_loss = 0
    steps = 0
    for j, pred_str in enumerate(pred_strs):
        gt_str = gt[j]
        cer = error_rates.cer(gt_str, pred_str)
        sum_loss += cer
        steps += 1
    return sum_loss, steps


def reset_all_stats(config, keyword="", freq=None):
    """ Only update the stats that have the same frequency as the update call

    Args:
        config:
        keyword:
        freq:

    Returns:

    """

    for title, stat in config["stats"].items():
        if isinstance(stat, Stat) and stat.accumlator_active and stat.accumulator_freq == freq and keyword.lower() in stat.name.lower():
            stat.reset_accumulator(new_x=None)
            config["logger"].debug(f"{stat.name} {stat.y[-1]}")

    try:
        visualize.plot_all(config)
    except:
        print("Problem graphing")

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        try:
            return super().default(o)
        except:
            d = o.__dict__
            # Don't save out circular reference
            return {i:d[i] for i in d if i!='x_dict'}

def stat_prep(config):
    """ Prep to track statistics/losses, setup plots etc.
        Visdom must already be initialized if using

    Returns:

    """
    config["stats"]["epochs"] = []
    config["stats"]["epoch_decimal"] = []
    #config["stats"]["instances"] = []
    config["stats"]["updates"] = []
    config.counter = Counter(instances_per_epoch=config.n_train_instances,
                             test_instances=config.n_test_instances,
                             test_pred_length_static=config.n_test_points,
                             validation_pred_length_static=config.n_validation_instances)

    # Prep storage
    config_stats = []

    config_stats.append(AutoStat(counter_obj=config.counter, x_weight="instances", x_plot="epoch_decimal",
                                 x_title="Epochs", y_title="HWR Training Loss", name=f"HWR_Training_Loss",
                                 train=True))
    config_stats.append(AutoStat(counter_obj=config.counter, x_weight="instances", x_plot="epoch_decimal",
                                 x_title="Epochs", y_title="Training Error Rate", name=f"Training_Error_Rate",
                                 train=True, ymax=.2))
    config_stats.append(AutoStat(counter_obj=config.counter, x_weight="test_instances", x_plot="epoch_decimal",
                                 x_title="Epochs", y_title="Test Error Rate", name=f"Test_Error_Rate",
                                 train=False, ymax=.2))
    config_stats.append(AutoStat(counter_obj=config.counter, x_weight="validation_pred_length_static", x_plot="epoch_decimal",
                                 x_title="Epochs", y_title="Validation Error Rate", name=f"Validation_Error_Rate",
                                 train=False, ymax=.2))

    # config_stats.append(Stat(y=[], x=config["stats"]["updates"], x_title="Updates", y_title="Loss", name="HWR Training Loss"))
    # config_stats.append(Stat(y=[], x=[], x_title="Instances", y_title="CER", name="Training Error Rate", accumulator_freq="instance"))
    # config_stats.append(Stat(y=[], x=[], x_title="Instances", y_title="CER", name="Test Error Rate", ymax=.2, accumulator_freq="instance"))
    # config_stats.append(Stat(y=[], x=[], x_title="Instances", y_title="CER", name="Validation Error Rate", ymax=.2, accumulator_freq="instance"))
    config["designated_training_cer"] = "Training_Error_Rate"
    config["designated_test_cer"] = "Test_Error_Rate"
    config["designated_validation_cer"] = "Validation_Error_Rate" if config["validation_jsons"] else "Test_Error_Rate"


    for stat in config_stats:
        if config["use_visdom"]:
            config["visdom_manager"].register_plot(stat.name, stat.x_title, stat.y_title, ymax=stat.ymax)
        config["stats"][stat.name] = stat


def stat_prep_strokes(config):
    """ Visdom must already be initialized if using

    Args:
        config:

    Returns:

    """
    config_stats = []
    # config.stats["epoch"] = 0
    # config.stats["epoch_decimal"] = 0
    # config.stats["instances"] = 0
    # config.stats["updates"] = 0

    config.counter = Counter(instances_per_epoch=config.n_train_instances, test_instances=config.n_test_instances, test_pred_length_static=config.n_test_points)

    for variant in ("train", "test"):
        is_training = variant == "train"
        # Not quite right
        #x_weight = "updates" if is_training else config.n_test_instances/config.batch_size

        #x_weight = "instances" if is_training else config.n_test_instances
        x_weight = "training_pred_count" if is_training else "test_pred_length_static" # should be a key in the counter object

        # # Always include L1 loss
        # config_stats.append(AutoStat(counter_obj=config.counter, x_weight=x_weight, x_plot="epoch_decimal",
        #                              x_title="Epochs", y_title="Loss", name=f"l1_{variant}", train=is_training))

        # TOTAL ACTUAL LOSS
        config_stats.append(AutoStat(counter_obj=config.counter, x_weight=x_weight, x_plot="epoch_decimal",
                                     x_title="Epochs", y_title="Loss", name=f"Actual_Loss_Function_{variant}", train=is_training))

        # NN Loss
        config_stats.append(AutoStat(counter_obj=config.counter, x_weight=x_weight, x_plot="epoch_decimal",
                                     x_title="Epochs", y_title="Loss", name=f"nn_{variant}", train=is_training))

        # Include how many GT points there were for this update
        config_stats.append(AutoStat(counter_obj=config.counter, x_weight=x_weight, x_plot="epoch_decimal",
                                     x_title="Epochs", y_title="Points Predicted", name=f"point_count_{variant}", train=is_training))

        # All other loss functions
        for loss in config.all_losses:
            if loss!="l1_default":
                config_stats.append(AutoStat(counter_obj=config.counter, x_weight=x_weight, x_plot="epoch_decimal",
                                             x_title="Epochs", y_title="Loss", name=f"{loss}_{variant}", train=is_training))
    for stat in config_stats:
        if config["use_visdom"]:
            config["visdom_manager"].register_plot(stat.name, stat.x_title, stat.y_title, ymax=stat.ymax)
        config["stats"][stat.name] = stat
    print(config.stats)

# class Number:
#     def __init__(self, value):
#         self.value = value
# 
#     
    
def plot_tensors(tensor):
    for i in range(0, tensor.shape[0]):
        plot_tensor(tensor[i,0])

def plot_tensor(tensor):
    #print(tensor.shape)
    t = tensor.cpu()
    assert not np.isnan(t).any()
    plt.figure(dpi=400)
    plt.imshow(t, cmap='gray')
    plt.show()

def kill_gpu_hogs(force=False):
    ## Try to kill just nvidia ones first; ask before killing everything; try to restart Visdom
    if is_taylor():
        utilization, memory_utilization = get_gpu_utilization()
        if memory_utilization > 50 or force:
            kill_all = input("GPU memory utilization over 50%; kill all python scripts? Y/n")
            if kill_all.lower()!="y":
                return
        else:
            return
        hwr_logger.logger.info("Killing GPU hogs")
        ## KILL ALL OTHER PYTHON SCRIPTS
        pid = os.getpid()
        if False:
            # All GPU processes
            # find_processes_command = "nvidia-smi | sed -n 's/|\s*[0-9]*\s*\([0-9]*\)\s*.*/\1/p' | sort | uniq | sed '/^\$/d'"
            # All python commands - this works a little better, but will kill visdom
            find_processes_command = f"pgrep -fl python"
            command = find_processes_command + f" | awk '!/{pid}/{{print $1}}' | xargs kill"
            result = Popen(command, shell=True)
        else:
            exclusion_words = "visdom", "jupyter", "grep"
            find_processes_command = f"ps all | grep python"  + f" | awk '!/{pid}/'"
            x = check_output([find_processes_command], shell=True)
            all_python_processes = x.decode().split("\n")[:-1]
            for process in all_python_processes:
                if not any([ew in process.lower() for ew in exclusion_words]):
                    hwr_logger.logger.info(f"killing {process}")
                    try:
                        os.kill(int(process.split()[2]), signal.SIGTERM)
                    except:
                        hwr_logger.logger.info("Didn't work")
                        pass
        return

def start_visdom(port=9001, suppress_output=True, suppress_errors=False):
    # Error is "OSError: [Errno 98] Address already in use"
    my_env = os.environ.copy()
    my_env["PATH"] = "/usr/sbin:/sbin:" + my_env["PATH"]
    stderr = DEVNULL if suppress_errors else STDOUT
    stdout = DEVNULL if suppress_output else STDOUT
    Popen("python -m visdom.server -p {}".format(port), shell=True, env=my_env, stdout=stdout, stderr=stderr)

def get_index(l, item):
    if item in l:
        return l.index(item)
    else:
        return -1

def npy_loader(path):
    if Path(path).suffix == ".json":
        numpy_path = Path(path).with_suffix(".npy")
        if numpy_path.exists() and (not Path(path).exists() or os.path.getmtime(numpy_path) > os.path.getmtime(path)):
            print("Loading .npy version")
            return np.load(numpy_path, allow_pickle=True)
        else: # if json was updated, re-make the numpy version
            print(f"No npy file detected or too old, saving {numpy_path}")
            new_data = json.load(Path(path).open("r"))
            np.save(numpy_path, new_data)
            return new_data
    elif Path(path).suffix == ".npy":
        return np.load(path)
    else:
        raise Exception(f"Unexpected file type {Path(path).suffix}")

if __name__=="__main__":
    from hwr_utils.visualize import Plot
    viz = Plot(port=9001)
    viz.viz.close()
    viz.load_all_env("./results")

