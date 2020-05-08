import sys 
sys.path.append("..")
sys.path.append(".")
import pickle, json 
from pathlib import Path 
import os
import socket

if not "byu.edu" in socket.gethostname():
    from visdom import Visdom

import numpy as np 
import torch 
import warnings 
import json 
from hwr_utils.stattrack import Stat 
import traceback

## Some functions stolen from https://github.com/theevann/visdom-save
DEFAULT_PORT = 9001

class Plot(object):
    def __init__(self, title="", env_name="", config=None, port=8080):
        self.env_name = env_name if env_name else title
        self.viz = Visdom(port=port, env=self.env_name)
        #self.viz.close()
        self.windows = {}
        self.title = title
        self.config = config

    def register_plot(self, name, xlabel, ylabel, plot_type="line", ymax=None):
        self.windows[name] = {"xlabel":xlabel, "ylabel":ylabel, "title":name, "plot_type":plot_type}
        self.windows[name]["opts"] = dict(title=name, markersize=5, xlabel=xlabel, ylabel=ylabel)

        if ymax is not None:
            self.windows[name]["opts"]["layoutopts"] = dict(plotly=dict(yaxis=dict(range=[0, ymax])))


    def update_plot(self, plot_name, x, y, **kwargs):
        # Create plot if not registered
        try:
            plot_d = self.windows[plot_name]
        except:
            warnings.warn("Plot not found, creating new plot")
            plot_d = {"xlabel":"X", "ylabel":"Y", "plot_type":"scatter"}

        plotter = self.viz.scatter if plot_d["plot_type"] == "scatter" else self.viz.line

        # WHY WAS "Y" A NESTED LIST???
        #data = {"X": np.asarray(x), "Y": np.asarray([y])} if plot_d["plot_type"] == "line" else {"X": np.asarray([x, y])}

        x = np.asarray(x)
        if len(x) < len(y):
            warnings.warn("X coords not found, interpolating")
            if x[0] == 0 and len(x)>1:
                x[0] = x[1]-.001
            additional_x = np.linspace(0, x[0], len(y)-len(x))

            x = np.r_[additional_x, np.asarray(x)]
        data = {"X": x, "Y": np.asarray(y)} if plot_d["plot_type"] == "line" else {"X": np.asarray([x, y])}

        ## Update plot
        if "plot" in plot_d.keys():
            plotter(
                **data,
                win=plot_d["plot"],
                update="append"
            )
        else: # Create new plot
            win = plotter(
                **data,
                opts=plot_d["opts"], **kwargs
            )
            plot_d["plot"] = win
            self.windows["name"] = plot_d

    # LOADING
    def load_log(self, path):
        self.viz.replay_log(path)

    def load_all_env(self, root, keyword="visdom"):
        for d, ss, fs in os.walk(root):
            for f in fs:
                full_env = os.path.join(d, f)
                # Don't load "BSF" graphs, just complete graphs
                if full_env[-5:]==".json" and keyword in full_env and f != "losses.json" and "BSF_" not in full_env:
                    print("Loading {}".format(full_env))
                    self.viz.replay_log(full_env) # viz.load load the environment to viz

    def save_env(self, file_path=None, current_env=None, new_env=None):
        if file_path is None:
            file_path = os.path.join(self.config["results_dir"], "visdom.json")
        if current_env is None:
            current_env = self.env_name
            
        new_env = current_env if new_env is None else new_env
        #self.viz = Visdom(env=current_env) # get current env
        data = json.loads(self.viz.get_window_data())
        if len(data) == 0:
            print("NOTHING HAS BEEN SAVED: NOTHING IN THIS VISDOM ENV - DOES IT EXIST ?")
            return
    
        file = open(file_path, 'w+')
        for datapoint in data.values():
            output = {
                'win': datapoint['id'],
                'eid': new_env,
                'opts': {}
            }
    
            if datapoint['type'] != "plot":
                output['data'] = [{'content': datapoint['content'], 'type': datapoint['type']}]
                if datapoint['height'] is not None:
                    output['opts']['height'] = datapoint['height']
                if datapoint['width'] is not None:
                    output['opts']['width'] = datapoint['width']
            else:
                output['data'] = datapoint['content']["data"]
                output['layout'] = datapoint['content']["layout"]
    
            to_write = json.dumps(["events", output])
            file.write(to_write + '\n')
        file.close()

def initialize_visdom(env_name, config):
    if not config["use_visdom"]:
        return
    try:
        config["visdom_manager"] = Plot("Loss", env_name=env_name, config=config, port=config.visdom_port)
        return config["visdom_manager"]
    except:
        config["use_visdom"] = False
        config["logger"].warning("Unable to initialize visdom, is the visdom server started?")

def plot_all(config):
    """
    ADD SMOOTHING
    Args:
        config:

    Returns:

    """
    if not config["use_visdom"]:
        return

    visdom_manager = config["visdom_manager"]
    for title, stat in config["stats"].items():
        if isinstance(stat, Stat) and stat.plot and stat.updated_since_plot:
            #print("updating {}".format(stat.name), stat.x, stat.y)
            visdom_manager.update_plot(stat.name, stat.x[-stat.plot_update_length:], stat.y[-stat.plot_update_length:])
            stat.updated_since_plot = False

def test():
    plot = Plot("Test")
    plot.register_scatterplot("Loss", "Epoch", "Loss")

    for i in range(0,10):
        plot.update_plot("Loss", i, 2)

def close_all_env(plotter):
    for env in plotter.viz.get_env_list():
        plotter.viz.delete_env(env)

def load_all_hwr(path, key='test_cer', clear=True, keywords=""):
    # python -m visdom.server -p 8080
    plotter = Plot("NewEnv", port=DEFAULT_PORT)
    close_all_env(plotter)

    for p in Path(path).rglob("losses.json"):
        if "BSF" not in p.as_uri() and keywords in p.as_uri():
            print(p)
            name = p.parent.name.replace("-", "_")
            split_name = name.split("_")
            #name = "_".join((name[0],name[2],name[3],name[4], name[1]))
            name = "_".join((split_name[2], split_name[3], split_name[0]))
            if len(split_name) > 4:
                name += "_".join(split_name[4:])

            plotter = Plot(name, port=DEFAULT_PORT)
            losses = json.loads(p.read_text())[key]
            x = list(range(len(losses)))
            plotter.register_plot(key, "Epoch", key, plot_type="line", ymax=.1)
            try:
                plotter.update_plot(plot_name=key, x=x, y=losses, name=name)
            except:
                print(f"Problem with {name, key}")

def load_all(path, key=None, clear=True, keywords=""):
    # python -m visdom.server -p 8080
    if False:
        plotter = Plot("NewEnv", port=DEFAULT_PORT)
        close_all_env(plotter)

    for p in Path(path).rglob("all_stats.json"):
        if "BSF" not in p.as_uri() and keywords in p.as_uri():
            print(p)
            name = p.parent.name.replace("-", "_")

            plotter = Plot(name, port=DEFAULT_PORT)
            stats = json.loads(p.read_text())["stats"]
            print(stats.keys())
            for key in stats.keys():
                if key in ["epochs", 'epoch_decimal', "updates"]:
                    continue
                losses = stats[key]

                try:
                    xlen, ylen = len(losses["x"]), len(losses["y"]) 
                    print(xlen,ylen)
                    m = min(xlen, ylen)
                    if xlen != ylen:
                        print("UNEQAL LENGTHS")
                        losses["x"] = losses["x"][:m]
                        losses["y"] = losses["y"][:m] 
                    plotter.register_plot(key, losses["x_title"], key, plot_type="line") # , ymax=.1
                    plotter.update_plot(plot_name=key, x=losses["x"], y=losses['y'], name=name)
                except:
                    traceback.print_exc()
                    print(losses, key)
                    print(f"Problem with {name, key}")
                    input()

def prep_path(foreign_paths):
    from hwr_utils.base_utils import increment_path, is_iterable
    import shutil
    """ If path is not local, copy them to the TEMP path

    Returns:

    """

    new_path = increment_path(base_path=Path("RESULTS/COMPARISON"), make_directory=True)
    if not is_iterable(foreign_paths):
        foreign_paths = [foreign_paths]

    for foreign_path in foreign_paths:
        for stats_file in Path(foreign_path).rglob("all_stats.json"):
            if "BSF" in stats_file.parent.name:
                continue
            experiment_folder_name = stats_file.parent.name
            experiment_folder_name2 = stats_file.parent.parent.name
            print(experiment_folder_name)
            new_sub = Path(new_path) / (experiment_folder_name2+"_"+experiment_folder_name)
            new_sub.mkdir(parents=True)
            shutil.copy(stats_file, new_sub)

    return new_path.absolute()

def load_bashrc():
    import os
    import pprint
    import shlex
    import subprocess

    command = shlex.split("env -i bash -c 'source init_env && env'")
    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
    for line in proc.stdout:
        (key, _, value) = line.partition("=")
        os.environ[key] = value
    proc.communicate()


if __name__=="__main__":
    # python -m visdom.server -p 9001
    from subprocess import Popen
    import time
    load_bashrc()
    p = Popen(f'pkill -f visdom', shell=True, close_fds=True)
    visdom_command = "/home/taylor/anaconda3/envs/hwr5/bin/python -m visdom.server -p 9001 &>/dev/null &"
    if True:
        path = r"/home/taylor/shares/Super/SuperComputerGroups/fslg_hwr/taylor_simple_hwr/results/stroke_config/ver3/"
        #path = r"fish://tarch@rhel7ssh.fsl.byu.edu/zhome/tarch/fsl_groups/fslg_hwr/compute/taylor_simple_hwr/results/stroke_config/ver3/"
        paths = [r"/media/SuperComputerGroups/fslg_hwr/taylor_simple_hwr/RESULTS/ver3",
                 "/home/taylor/shares/brodie/github/simple_hwr/RESULTS/ver3/",
                 "/home/taylor/shares/brodie/github/simple_hwr/RESULTS/ver4/"]
        #path = r"/media/data/GitHub/simple_hwr/RESULTS/COMPARISON"
        paths = ["/media/SuperComputerGroups/fslg_hwr/taylor_simple_hwr/RESULTS/indic/first_attempt"]
        # "/media/data/GitHub/simple_hwr/RESULTS/ver4",
        # "/media/SuperComputerGroups/fslg_hwr/taylor_simple_hwr/results/stroke_config/ver4/20200305_072609-stroke_number_with_BCE_lil_bigger/"]
        # "/media/SuperComputerGroups/fslg_hwr/taylor_simple_hwr/results/stroke_config/20200301-PRETRAIN/"
        # "/home/taylor/shares/brodie/github/simple_hwr/RESULTS/ver4"

        paths = ["/home/taylor/shares/brodie/github/simple_hwr/RESULTS/ver4/20200229_223630-stroke_number_with_BCE_RESUME",
                  "/media/SuperComputerGroups/fslg_hwr/taylor_simple_hwr/RESULTS/20200301-PRETRAIN/",
                    "/media/SuperComputerGroups/fslg_hwr/taylor_simple_hwr/results/stroke_config/ver5"]
        # "/media/SuperComputerGroups/fslg_hwr/taylor_simple_hwr/RESULTS/20200301-PRETRAIN"]
        paths = ["/home/taylor/shares/brodie/github/simple_hwr/RESULTS/ver5",
                 "/media/SuperComputerGroups/fslg_hwr/taylor_simple_hwr/results/stroke_config/ver5",
                 '/media/SuperComputerGroups/fslg_hwr/taylor_simple_hwr/results/stroke_config/20200301-PRETRAIN']

        paths = ["/simple_hwr/results/recognition/strokes_v1"]
        path = prep_path(paths)
    else:
        path = Path("/media/data/GitHub/simple_hwr/RESULTS/COMPARISON/11_")


    p = Popen(visdom_command, shell=True, close_fds=True)
    time.sleep(2)

    #path = r"./results/stroke_config"
    #path = r"/media/SuperComputerGroups/fslg_hwr/taylor_simple_hwr/results/long/variants"
    load_all(path, keywords="") #, key="validation_cer")
