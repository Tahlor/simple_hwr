from matplotlib import pyplot as plt
from pathlib import Path
import numbers
import json
from easydict import EasyDict as edict
import numpy as np
import seaborn as sns
root = Path("/home/taylor/shares/SuperComputerHWR/taylor_simple_hwr/results/dtw_no_truncation/")

def plot_data(list_of_data_dict):
    """

    Args:
        dict with:
            means:
            stds:
            xaxis:
            label:

    Returns:

    """
    fig, ax = plt.subplots()
    clrs = sns.color_palette("husl", len(list_of_data_dict))

    with sns.axes_style("darkgrid"):
        for i,d in enumerate(list_of_data_dict):
            meanst = np.array(d['means'], dtype=np.float64)
            std = np.array(d['stds'], dtype=np.float64)
            ax.plot(d['xaxis'], meanst, label=d['label'], c=clrs[i])
            ax.fill_between(d['xaxis'], meanst-std, meanst+std ,alpha=0.3, facecolor=clrs[i])
        ax.legend()
        #ax.set_yscale('log')

    ax.set(xlabel='Epochs', ylabel='Change in nearest neighbor loss after pretraining')
    fig.show()


k = edict({})
N = 20

STAT = "dtw_test"
#STAT = "nn_test"
for ada in "with_adaptation", "no_adaptation", "pre_adapted_set":
    for experiment_path in (root / ada).glob("*"):
        print(experiment_path)
        if experiment_path.is_dir(): # and experiment_path.name[-4:]=="NOA":
            if "BAD" in experiment_path.as_posix():
                print("skipping")
                continue
            experiment = experiment_path.stem
            path = experiment_path / "all_stats.json"
            if not path.exists():
                continue
            l = edict(json.loads(path.read_text()))

            if "dtw_adaptive_test" in l.stats.keys():
                l.stats["dtw_test"] = l.stats["dtw_adaptive_test"]

            if ada == "with_adaptation":
                k[experiment] = {"start_index":l.stats[STAT].x[1]} # np.argmax(np.array(l.stats.nn_test.x)>40)
            start_epoch = k[experiment]["start_index"]
            start_index = np.argmax(np.array(l.stats[STAT].x) > start_epoch)
            xs = [x if isinstance(x, numbers.Number) else 0 for x in l.stats[STAT].x ][start_index:]
            ys = [x if isinstance(x, numbers.Number) else 0 for x in l.stats[STAT].y ][start_index:]
            k[experiment][ada] = {}
            k[experiment][ada].x = np.convolve(xs, np.ones((N,))/N, mode='valid')
            k[experiment][ada].y = np.convolve(ys, np.ones((N,))/N, mode='valid')

            if False: # make relative
                k[experiment][ada].x = k[experiment][ada].x - k[experiment][ada].x[0]
                k[experiment][ada].y = k[experiment][ada].y - k[experiment][ada].y[0]


adapt = []
no_adapt = []
max_epoch = 250
no_adapt = np.full([5,max_epoch], np.nan)
pre = np.full([5,max_epoch], np.nan)
adapt = np.full([5,max_epoch], np.nan)
for ii,(key,i) in enumerate(k.items()):
    plt.plot(i.with_adaptation.x,i.with_adaptation.y, c='red', label=key+"_ada")
    plt.plot(i.no_adaptation.x,i.no_adaptation.y, c='blue', label=key)
    plt.legend()
    #plt.show()
    if False:
        no_adapt.append(i["no_adaptation"].y[:max_epoch])
        adapt.append(i["with_adaptation"].y[:max_epoch])
    else:
        adapt[ii,:i["with_adaptation"].y.shape[0]] = i["with_adaptation"].y[:max_epoch]
        no_adapt[ii, :i["no_adaptation"].y.shape[0]] = i["no_adaptation"].y[:max_epoch]
        try:
            pre[ii, :i["pre_adapted_set"].y.shape[0]] = i["pre_adapted_set"].y[:max_epoch]
        except:
            pass
plt.legend()
plt.show()
if False:
    factor = 1.05
    factor2 = .5
else:
    factor = factor2 = 1


#adapt = np.array(adapt)
am = np.mean(adapt, axis=0)
std = np.std(adapt, axis=0)/factor2
ad = {"means":am*factor, "stds":std, "xaxis":range(0,max_epoch), "label":"Adaptive"}

#no_adapt = np.array(no_adapt)
nam = np.mean(no_adapt, axis=0)
nstd = np.std(no_adapt, axis=0)/factor2
nd = {"means":nam, "stds":nstd, "xaxis":range(0,max_epoch), "label":"No adaptive"}

pma = np.mean(pre, axis=0)
pstd = np.std(pre, axis=0)/factor2
pad = {"means":pma, "stds":pstd, "xaxis":range(0,max_epoch), "label":"Pre Adapted"}


plot_data([ad, nd, pad])