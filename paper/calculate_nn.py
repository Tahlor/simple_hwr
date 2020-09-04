import torch
from scipy import interpolate
from gen_preds_online import load_online_dataloader
from tqdm import tqdm
from scipy.spatial import KDTree
from easydict import EasyDict as edict
import loss_metrics
import os
from pathlib import Path
import random
import numpy as np
import shutil
from matplotlib import pyplot as plt
import cv2
import hwr_utils.stroke_recovery as stroke_recovery
# Online
from torch.utils.data import DataLoader
from hwr_utils.stroke_dataset import BasicDataset, StrokeRecoveryDataset
from losses import DTWLoss
#matplotlib.use("TkAgg")

# Offline
# data
offline_path = "/media/data/GitHub/simple_hwr/RESULTS/OFFLINE_PREDS/RESUME_model/imgs/current/eval/data/all_data.npy"
# distance to nearest pixel - assume 0 if less than .5
TESTING=False

def sample():
    # Randomly choose 10 samples
    root = Path("/media/data/GitHub/simple_hwr/RESULTS/OFFLINE_PREDS/RESUME_model/imgs/current/eval")

    all_files = list(Path(root).glob("overlay*"))
    n_files = len(all_files)
    n_samples = 10
    choices = np.random.choice(n_files,n_samples)

    sampled_dir = root / "sampled"
    sampled_dir.mkdir(parents=True, exist_ok=True)

    for i in choices:
        shutil.copy(str(all_files[i]), str(sampled_dir))

def plot(preds, targs, targs2=None, name="suck"):
    # plt.axis('off')
    # plt.axis('square')
    plt.figure(figsize=[4, 1], dpi=180)
    plt.plot(preds[:, 0], preds[:, 1])
    plt.plot(targs[:, 0], targs[:, 1])
    if not targs2 is None:
        plt.plot(targs2[:, 0], targs2[:, 1])
    plt.savefig(f"/media/data/GitHub/simple_hwr/RESULTS/ONLINE_PREDS/RESUME_model/{name}.png")
    plt.show()


def test():
    # 0,0 to 1,1 = -1,0
    # 1,1 to 2,2 = -2,1
    # Count it as 0 if the floor and ceiling are also in reference
    # Just round all the predictions

    img = "/home/taylor/Desktop/X.png"
    #x = plt.imread(img) # bottom left is x[-1,0]
    x = cv2.imread(img, 0) # origin is top left W=8, H=7

    # Tests
    nearest_points, distances, kd, reference = stroke_recovery.calculate_distance(x, np.asarray([[0,0],[1,1]]), reference_is_image=True)
    nearest_points, distances, kd, reference = stroke_recovery.calculate_distance(x, np.asarray([[.2,1.1],[1,1]]), reference_is_image=True)
    nearest_points, distances, kd, reference = stroke_recovery.calculate_distance(x, np.asarray([[1.2,0],[1,1]]), reference_is_image=True)

    return x, nearest_points, distances, kd

#x, nearest_points, distances, kd = test()

def fast_resample(preds, output_sample_factor=8, resample_sz=None):
    sz = preds.shape[0]
    resample_sz = resample_sz if resample_sz else output_sample_factor*sz
    rng = 1./(sz-2) * np.array(range(sz)).astype(float)
    x = interpolate.interp1d(rng, preds[:,0])
    y = interpolate.interp1d(rng, preds[:,1])
    linspace = np.linspace(0,1,int(resample_sz))
    #np.concatenate([x(linspace),y(linspace)])
    return np.c_[x(linspace), y(linspace)]

if __name__=='__main__':
    ONLINE = True
    PROJ_ROOT= Path(os.path.dirname(os.path.realpath(__file__))).parent

    if ONLINE:
        pred_data_path = "/media/data/GitHub/simple_hwr/results/stroke_config/ONLINE_PREDS/RESUME_model/new_experiment01/all_data.npy"  # change to 11
        test_dataloader, test_dataset = load_online_dataloader(batch_size=1, shuffle=False, resample=True)
        test_dataloader2, test_dataset2 = load_online_dataloader(batch_size=1, shuffle=False, resample=False)
        plot(test_dataset[0]["raw_gt"], test_dataset[0]["gt"], name="suck10")

    elif False:
        config = edict({"dataset":
                            {"resample": True,
                             "linewidth": 1,
                             "kdtree": False,
                             "image_prep": "pil",
                             },
                        "loss_fns": [{"name": ""}],
                        "warp": False,

                        })

        # DTW and NN # new_experiment15
        pred_data_path = "/media/data/GitHub/simple_hwr/results/stroke_config/ONLINE_PREDS/RESUME_model/new_experiment48/all_data.npy" # change to 11
        train_data_path = "/media/data/GitHub/simple_hwr/data/online_coordinate_data/MAX_stroke_vlargeTrnSetFull/test_online_coords.npy"
        extension = ".tif"
        folder = PROJ_ROOT / Path("data/prepare_online_data/")
        test_dataset=StrokeRecoveryDataset([train_data_path],
                                root=PROJ_ROOT / "data",
                                max_images_to_load=None,
                                cnn_type="default64",
                                test_dataset = True,
                                config=config,
                                )
        cs = lambda x: test_dataset.collate(x, alphabet_size=test_dataset.alphabet_size)
        test_dataloader = DataLoader(test_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=5,
                                      collate_fn=cs,
                                      pin_memory=False)

    else:
        # just NN stuff
        folder = PROJ_ROOT / Path("data/prepare_IAM_Lines/lines/")
        if TESTING:
            pred_data_path = "/media/data/GitHub/simple_hwr/results/stroke_config/OFFLINE_PREDS/RESUME_model/new_experiment07/all_data.npy"
        else:
            train_data_path = "/media/data/GitHub/simple_hwr/data/online_coordinate_data/MAX_stroke_vlargeTrnSetFull/test_online_coords.npy"
        
        if False:
            extension = ".png"
            eval_dataset = BasicDataset(root=folder, cnn="default64", rebuild=False, extension=extension, crop=True)
            eval_loader = DataLoader(eval_dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=6,
                                     collate_fn=eval_dataset.collate,  # this should be set to collate_stroke_eval
                                     pin_memory=False)

    def calc_nn(moving, tree):
        """ Naive nn calculation

        Args:
            moving:
            tree:

        Returns:

        """
        if isinstance(tree, torch.Tensor):
            tree = tree.detach().numpy()
        kd = KDTree(tree[:, :2])
        dist = kd.query(moving[:, :2], p=2)[0]  # How far do we have to move the GT's to match the predictions?
        return dist

    #next(iter(test_dataloader))

    # % of GT start points with adjacent predicted start point (recall)
    # % of predicted start points not near GT start point (precision)

    dtw_losses = []
    dtw_losses_l2 = []
    nn_losses_pred_moves = []
    nn_losses_gt_moves = []
    nn_online = []
    RESAMPLE_FACTOR = 0 # resample to make X times as many predictions; if 0, use GT length
    D_RESAMPLE = True # use the distance resampled version
    NO_RESAMPLE = True

    if RESAMPLE_FACTOR:
        ignore_last = 20 * RESAMPLE_FACTOR  # ignore last 20 preds for calculating loss
        opts = {"output_sample_factor": RESAMPLE_FACTOR}

    PLOT = False

    if ONLINE:
        # Resample???
        # Calculate DTW
        # Open the dataset
        predicted_dataset = np.load(pred_data_path, allow_pickle=True)
        d2 = {}
        for i in predicted_dataset:
            d2[i['id']] = i

        loss_obj = DTWLoss(loss_indices=[0, 1], window_size=2000)

        for ii, item in enumerate(tqdm(test_dataloader)):
            #targs = targs_loader = item['raw_gt'][0] #

            try:
                id = item['id'][0]
                preds = d2[id]['raw_pred'] # 'stroke' has been post_processed and resampled

                if D_RESAMPLE:
                    targs = targs1 =  d2[id]['gt_list']
                else:
                    targs = targs2 =  d2[id]['raw_gt'] # no resample

                img = d2[id]['img']
            except:
                #print(f"{id} NOT FOUND")
                continue

            if not RESAMPLE_FACTOR:
                ignore_last = int(20 * preds.shape[0]/targs.shape[0])  # ignore last 20 preds for calculating loss
                opts = {"resample_sz": targs.shape[0]}

            if NO_RESAMPLE:
                ignore_last=20
                opts = {}
                fast_resample = lambda x: x.numpy()

            #print(preds.shape)
            preds = torch.from_numpy(fast_resample(preds, **opts))

            #print(preds.shape)

            # raw_gt is the unresampled gt
            # an unresampled dataloader will not generate the same width GT because stuff gets resized based on the resampled dimensions
            # plot(targs1, targs2, test_dataset2[ii]["gt"]) # non-resample data still doesn't really match
            # plot(test_dataset2[ii]["gt"], test_dataset2[ii]["raw_gt"], name="suck81")
            # input()
            # print(id)
            # print(next(iter(test_dataloader))["gt_list"][:5])

            # pred, distance, kd = post_process(p[:remove_end], gt, calculate_distance=NN_DISTANCES, kd=kd)
            # p = stroke_recovery.resample(pred)
            # preds = preds[:-7]
            loss =    loss_obj.dtw([preds],[targs],label_lengths=item['label_lengths'], item=item, dont_sum=True)
            loss_l2 = loss_obj.dtw([preds],[targs],label_lengths=item['label_lengths'], item=item, dont_sum=True, exponent=2)
            dtw_losses.append(np.average(loss[:-ignore_last]))  # just ignore the loss at the end
            dtw_losses_l2.append(np.average(loss_l2[:-ignore_last]))

            nearest_points, distances, kd, reference = stroke_recovery.calculate_distance(item["line_imgs"].squeeze(),
                                                                                          preds[:-ignore_last, :2],
                                                                                          reference_is_image=True
                                                                                          )
            nn_online.append(np.mean(distances))
            if PLOT:
               plot(preds, targs)
            # Calculate NN BOTH WAYS
            if True:
                l_gt_moves = calc_nn(targs, tree=preds)[:-ignore_last]
                l_pred_moves = calc_nn(preds, tree=targs)[:-ignore_last]
                nn_losses_gt_moves.append(np.mean(l_gt_moves))
                nn_losses_pred_moves.append(np.mean(l_pred_moves))

        print("D RESAMPLE", D_RESAMPLE)

        print(np.average(nn_losses_pred_moves))
        print(np.average(nn_losses_gt_moves))
        print(np.average(dtw_losses))
        print(np.average(dtw_losses_l2))
        print(np.average(nn_online))

        d={}
        for i, id in enumerate(d2.keys()):
            d[id] = nn_online[i]
        s = sorted(d.items(), key=lambda x: x[1])


    else:

        # Open the dataset
        pred_data_path = Path("/media/data/GitHub/simple_hwr/results/stroke_config/OFFLINE_PREDS/RESUME_model/new_experiment09/")

        nn_distances = []
        for i in range(10):
            d2 = np.load(pred_data_path / f"all_data_{i}.npy", allow_pickle=True)

            for i, item in enumerate(tqdm(d2)):
                # Calculate NN
                #print(item["raw_pred"].shape[0])
                nearest_points, distances, kd, reference = stroke_recovery.calculate_distance(item["img"],
                                                                                              item["raw_pred"][:-50,:2],
                                                                                              #item["stroke"][:-10,:2],
                                                                                              reference_is_image=True
                                                                                              )
                nn_distances.append(np.average(distances))
                # if i > 10:
                #     break
            print(np.average(nn_distances))
        print(np.average(nn_distances))

        # create dataset
        # feed images and predictions into distance calculator

        ids = []
        for i in range(10):
            d2 = np.load(pred_data_path / f"all_data_{i}.npy", allow_pickle=True)
            for i, item in enumerate(tqdm(d2)):
                ids.append(item["id"])

        d = {}
        for i, id in enumerate(ids):
            d[id] = nn_distances[i]

        #s = {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}
        s = sorted(d.items(), key=lambda x: x[1])
