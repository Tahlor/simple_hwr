from matplotlib import pyplot as plt
from pathlib import Path
import numbers
import json
from easydict import EasyDict as edict
import numpy as np
import seaborn as sns
import hwr_utils.utils as utils

root = Path("/home/taylor/shares/SuperComputerHWR/taylor_simple_hwr/results/dtw_no_truncation/FOR_REALZ")

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

    ax.set(xlabel='Epochs', ylabel=f'Change in {STAT} loss after pretraining')
    fig.show()
    fig.savefig("./charts/NN_test_loss_adpated.png")


def get_last_epochs():
    path = "/home/taylor/shares/SuperComputerHWR/taylor_simple_hwr/results/dtw_no_truncation/pre_adapted_BACKUP"
    d = {}
    for p in Path(path).glob("*"):
        stats = json.load((p / "all_stats.json").open())
        d[p.name] = len(stats["stats"]["dtw_test"]["x"]) - 1 #[-1]
    return d

d = get_last_epochs()

STAT = "dtw_test"
STAT = "nn_test"

k = edict({})
N = 20

for ada in "no_adaptation", "pre_adapted_set":
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

            if ada == "no_adaptation":
                k[experiment] = {"start_index":l.stats[STAT].x[1]} # np.argmax(np.array(l.stats.nn_test.x)>40)
            start_epoch = k[experiment]["start_index"]
            start_index = np.argmax(np.array(l.stats[STAT].x) > start_epoch)
            xs = [x if isinstance(x, numbers.Number) else 0 for x in l.stats[STAT].x ][start_index:]
            ys = [x if isinstance(x, numbers.Number) else 0 for x in l.stats[STAT].y ][start_index:]
            k[experiment][ada] = {}
            k[experiment][ada].x = np.convolve(xs, np.ones((N,))/N, mode='valid')
            k[experiment][ada].y = np.convolve(ys, np.ones((N,))/N, mode='valid')

            if True:
                l = d[experiment]
                k[experiment][ada].y = k[experiment][ada].y[l:]
                k[experiment][ada].x = k[experiment][ada].x[l:]
                k[experiment][ada].x = k[experiment][ada].x - k[experiment][ada].x[0]

            # if False: # make relative
            #     k[experiment][ada].x = k[experiment][ada].x - k[experiment][ada].x[0]
            #     #k[experiment][ada].y = k[experiment][ada].y - k[experiment][ada].y[0]
            # elif True:
            #     pass
            # elif True:
            #     l = len(old_k[experiment][ada].x)
            #     l2 = len(k[experiment][ada].x)
            #     print(l,l2)
            #     k[experiment][ada].y = k[experiment][ada].y[l:]
            #     k[experiment][ada].x = k[experiment][ada].x[l:]
            #
            # elif True:
            #     print(k[experiment][ada].x[0])
            #     print(ada, experiment, old_k[experiment][ada].x[-1])
            #     k[experiment][ada].x = k[experiment][ada].x - old_k[experiment][ada].x[-1]
            #     print(k[experiment][ada].x[0])
            #     #k[experiment][ada].y = k[experiment][ada].y - old_k[experiment][ada].y[-1]


save_folder = utils.incrementer(root=Path("./ablation/"), base="progress")
np.save( save_folder / "k_dict.npy", k, allow_pickle=True)

adapt = []
no_adapt = []
max_epoch = 800
no_adapt = np.full([5,max_epoch], np.nan)
pre = np.full([5,max_epoch], np.nan)
adapt = np.full([5,max_epoch], np.nan)
for ii,(key,i) in enumerate(k.items()):
    plt.plot(i.pre_adapted_set.x,i.pre_adapted_set.y, c='red', label=key+"_ada")
    plt.plot(i.no_adaptation.x,i.no_adaptation.y, c='blue', label=key)
    plt.legend()
    #plt.show()
    if True:
        #adapt[ii,:i["with_adaptation"].y.shape[0]] = i["with_adaptation"].y[:max_epoch]
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
    no_adapt_factor2 = 1
no_adapt_factor2 = 1

last_bit = 600
max_epoch = last_bit

x_start = 0

#adapt = np.array(adapt)
# am = np.mean(adapt, axis=0)
# std = np.std(adapt, axis=0)/factor2
# ad = {"means":(am*factor)[:last_bit], "stds":std[:last_bit], "xaxis":range(x_start,max_epoch+x_start), "label":"Adaptive"}

#no_adapt = np.array(no_adapt)
nam = np.nanmean(no_adapt, axis=0)
nstd = np.nanstd(no_adapt, axis=0)*no_adapt_factor2
nd = {"means":nam[:last_bit], "stds":nstd[:last_bit], "xaxis":range(x_start,max_epoch+x_start), "label":"No adaptive"}

start = 0
pma = np.nanmean(pre, axis=0)
pstd = np.nanstd(pre, axis=0)
pad = {"means":pma[start:last_bit+start], "stds":pstd[start:last_bit+start], "xaxis":range(x_start,max_epoch+x_start), "label":"Adapted"}
STAT = "nearest neighbor distance"

plot_data([nd, pad])
#plot_data([ad, nd, pad])