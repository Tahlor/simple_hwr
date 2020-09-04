import shutil
from models.basic import CNN, BidirectionalRNN
from torch import nn
from models.CoordConv import CoordConv
from hwr_utils import visualize
from torch.utils.data import DataLoader
from loss_module.stroke_recovery_loss import StrokeLoss
from trainers import TrainerStrokeRecovery
from hwr_utils.stroke_dataset import BasicDataset, StrokeRecoveryDataset
from hwr_utils.stroke_recovery import *
import hwr_utils.stroke_recovery as stroke_recovery
from hwr_utils import utils
from torch.optim import lr_scheduler
from models.stroke_model import StrokeRecoveryModel
from train_stroke_recovery import parse_args, graph
from hwr_utils.hwr_logger import logger
from pathlib import Path
import os
from tqdm import tqdm
from subprocess import Popen
import h5py
import math

pid = os.getpid()

def load_online_dataloader(batch_size=1, shuffle=True, resample=True):
    train_data_path = "/media/data/GitHub/simple_hwr/data/online_coordinate_data/MAX_stroke_vlargeTrnSetFull/test_online_coords.json"
    extension = ".tif"
    folder = PROJ_ROOT / Path("data/prepare_online_data/")

    config1 = edict({"dataset":
                         {"resample": resample,
                          "linewidth": 1,
                          "kdtree": False,
                          "image_prep": "pil",
                          },
                     "loss_fns": [{"name": ""}],
                     "warp": False,

                     })

    eval_dataset = StrokeRecoveryDataset([train_data_path],
                                         root=Path(PROJ_ROOT) / "data",
                                         max_images_to_load=None,
                                         cnn_type="default64",
                                         test_dataset=True,
                                         config=config1,
                                         deterministic=True,
                                         force_recreate=False,
                                         )

    cs = lambda x: eval_dataset.collate(x, alphabet_size=eval_dataset.alphabet_size)
    eval_loader = DataLoader(eval_dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=5,
                             collate_fn=cs,
                             pin_memory=False)

    # print(SHUFFLE, "shuffle")
    # x = next(iter(eval_loader))
    # print(x["id"][0])
    # print(x["gt_list"][0])
    # stop
    return eval_loader, eval_dataset

#@debugger
def main(config_path):
    global epoch, device, trainer, batch_size, output, loss_obj, x_relative_positions, config, LOGGER, load_path_override
    torch.cuda.empty_cache()

    for load_path_override in [load_path_override
                               ]:

        #load_path_override = PROJ_ROOT + "RESULTS/pretrained/new_best/good.pt"

        #load_path_override = PROJ_ROOT + "RESULTS/OFFLINE_PREDS/all_data.npy"
        _load_path_override = Path(load_path_override)

        OUTPUT = PROJ_ROOT / Path(f"RESULTS/{'ONLINE' if ONLINE else 'OFFLINE'}_PREDS/") / _load_path_override.stem

        _t = utils.increment_path(name="eval", base_path=OUTPUT / "imgs/current")
        model_output_dir = OUTPUT / _t / "data"
        model_output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(_load_path_override, model_output_dir) # save the model ya hoser

        # Make these the same as whereever the file is being loaded from; make the log_dir and results dir be a subset
        # main_model_path, log_dir, full_specs, results_dir, load_path


        config = utils.load_config(config_path, hwr=False, results_dir_override=OUTPUT.as_posix())
        config.use_visdom = False
        # Free GPU memory if necessary
        if config.device == "cuda":
            utils.kill_gpu_hogs()

        batch_size = 28 #config.batch_size

        vocab_size = VOCAB_OVERRIDE if VOCAB_OVERRIDE else config.feature_map_dim

        device=torch.device(config.device)
        #device=torch.device("cpu")

        # OLD OUTPUT MAKER
        #output = utils.increment_path(name="Run", base_path=Path(load_path_override).parent)
        #output = Path(config.results_dir)
        #output.mkdir(parents=True, exist_ok=True)

        folder = Path(config.dataset_folder)

        if True:
            if not ONLINE:
                folder = PROJ_ROOT / Path("data/prepare_IAM_Lines/lines/")
                gt_path = PROJ_ROOT / Path("data/prepare_IAM_Lines/gts/lines/txt")
                extension = ".png"
                meta_data_files = "*.json"
            else:
                folder = PROJ_ROOT / Path("data/prepare_online_data/lineImages/")
                gt_path = PROJ_ROOT / Path("data/prepare_online_data/")
                extension = ".tif"
                meta_data_files = "*online_augmentation.json"

        else:
            folder = Path("/media/data/GitHub/simple_hwr/data/prepare_IAM_Lines/words")
            gt_path = PROJ_ROOT / Path("data/prepare_IAM_Lines/gts/words")

        model = StrokeRecoveryModel(vocab_size=vocab_size,
                                    first_conv_op=config.coordconv,
                                    first_conv_opts=config.coordconv_opts,
                                    **config.model_definition).to(device)

        ## Loader
        logger.info(("Current dataset: ", folder))

        # Dataset - just expecting a folder
        if not ONLINE:
            eval_dataset=BasicDataset(root=folder, cnn=model.cnn, rebuild=False, extension=extension, crop=True, vertical_pad=False if ONLINE else True)

            if MASTER_LIST:
                for i in range(len(eval_dataset.data),0,-1):
                    if eval_dataset.data[i-1]["id"] not in MASTER_LIST:
                         eval_dataset.data.pop(i-1)

            eval_loader=DataLoader(eval_dataset,
                                      batch_size=batch_size,
                                      shuffle=SHUFFLE,
                                      num_workers=6,
                                      collate_fn=eval_dataset.collate, # this should be set to collate_stroke_eval
                                      pin_memory=False)
        else:
            eval_loader, eval_dataset = load_online_dataloader(batch_size=batch_size, shuffle=SHUFFLE)

            # Remove items from BasicDataset not in Test set
            fpath = "/media/data/GitHub/simple_hwr/data/online_coordinate_data/MAX_stroke_vlargeTrnSetFull/test_online_coords.npy"
            test_data = np.load(fpath, allow_pickle=True)
            test_set = {Path(item['xml_path']).stem: 0 for item in test_data}
            for i in range(len(eval_dataset.data)-1,-1,-1):
                if Path(eval_dataset.data[i]["image_path"]).stem.split("_")[0] not in test_set:
                    eval_dataset.data.pop(i)
                #eval_dataset.data["label_length"] =

        #next(iter(eval_dataset))

        config.n_train_instances = None
        config.n_test_instances = len(eval_loader.dataset)
        config.n_test_points = None

        ## Stats
        if config.use_visdom:
            visualize.initialize_visdom(config["full_specs"], config)
        utils.stat_prep_strokes(config)

        # Create loss object
        config.loss_obj = StrokeLoss(loss_names=config.loss_fns, loss_stats=config.stats, counter=config.counter)
        optimizer = torch.optim.Adam(model.parameters(), lr=.0005 * batch_size/32)
        config.scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=.95)
        trainer = TrainerStrokeRecovery(model, optimizer, config=config, loss_criterion=config.loss_obj)

        config.model = model
        config.load_path = load_path_override if "load_path_override" in {**locals(), **globals()} else config.load_path

        config.sigmoid_indices = TrainerStrokeRecovery.get_indices(config.pred_opts, "sigmoid")

        # Load the GTs
        GT_DATA = load_all_gts(gt_path, extension=meta_data_files)
        print("Number of images: {}".format(len(eval_loader.dataset)))
        print("Number of GTs: {}".format(len(GT_DATA)))

        ## LOAD THE WEIGHTS
        config = utils.load_model_strokes(config) # should be load_model_strokes??????
        for parameter in model.parameters():
            parameter.requires_grad = False
        model = model.to(device)
        model.eval()

        # OUTPUT PATH
        if ONLINE:
            config.output_root = config.output_root.replace("OFFLINE", "ONLINE")
            Path(config.output_root).mkdir(exist_ok=True, parents=True)

        _output = utils.incrementer(Path(config.output_root), "new_experiment")

        eval_only(eval_loader, model, output_path=_output)
        globals().update(locals())

def post_process(pred,gt, calculate_distance=True, kd=None):
    #return make_more_starts(move_bad_points(reference=gt, moving_component=pred, reference_is_image=True), max_dist=.15)
    if calculate_distance:
        _, distances, kd = stroke_recovery.get_nearest_point(gt, pred, reference_is_image=False, kd=kd)
    else:
        distances = 0

    if True:
        return make_more_starts(pred, max_dist=.18), distances, kd
        # Move single points that are far from everything to nearest part on image
            # If that nearest point is far from other points
            # Else delete that point
    else:
        return pred.numpy(), distances, kd


PROJ_ROOT= os.path.dirname(os.path.realpath(__file__))
config_path = "/media/data/GitHub/simple_hwr/~RESULTS/20191213_155358-baseline-GOOD_long/TEST.yaml"
config_path = "/media/SuperComputerGroups/fslg_hwr/taylor_simple_hwr/RESULTS/ver1/RESUME.yaml"
config_path = PROJ_ROOT + "RESULTS/OFFLINE_PREDS/good/normal_preload.yaml"
config_path = "/media/data/GitHub/simple_hwr/RESULTS/pretrained/dtw_v3/normal_preload.yaml"
config_path = PROJ_ROOT + "/configs/stroke_configs/ver8/dtw_adaptive.yaml"
config_path = "/media/data/GitHub/simple_hwr/results/stroke_config/pretrained/with_EOS/RESUME.yaml"

load_path_override = "/media/SuperComputerGroups/fslg_hwr/taylor_simple_hwr/results/stroke_config/GOOD/baseline_model.pt"
load_path_override = "/media/data/GitHub/simple_hwr/~RESULTS/20191213_155358-baseline-GOOD_long"
load_path_override = "/media/SuperComputerGroups/fslg_hwr/taylor_simple_hwr/RESULTS/ver1/20200215_014143-normal/normal_model.pt"
load_path_override = "/media/SuperComputerGroups/fslg_hwr/taylor_simple_hwr/RESULTS/ver2/20200217_033031-normal2/normal2_model.pt"
load_path_override = "/media/data/GitHub/simple_hwr/RESULTS/pretrained/dtw_train_2.9/normal_preload_model.pt"
load_path_override = "/media/data/GitHub/simple_hwr/RESULTS/pretrained/dtw_train_v2/v2.pt"
load_path_override = PROJ_ROOT + "/RESULTS/pretrained/dtw_adaptive_new_model.pt"
load_path_override = PROJ_ROOT + "/RESULTS/pretrained/adapted_v2/"
load_path_override = "/home/taylor/shares/brodie/home/taylor/github/simple_hwr/RESULTS/ver8/20200406_131747-dtw_adaptive_new2_restartLR_RESUME/RESUME_model.pt"
load_path_override = "/home/taylor/shares/brodie/home/taylor/github/simple_hwr/RESULTS/ver8/20200406_131747-dtw_adaptive_new2_restartLR_RESUME/RESUME_model.pt"
load_path_override = "/media/data/GitHub/simple_hwr/results/stroke_config/pretrained/with_EOS/dtw_adaptive_no_truncation_model.pt"
load_path_override = "/media/data/GitHub/simple_hwr/results/stroke_config/pretrained/with_EOS/RESUME_Bigger_Window_model.pt"

# MAIN GOOD ONE
vers="normal"
load_path_override = "/media/data/GitHub/simple_hwr/RESULTS/OFFLINE_PREDS/RESUME_model/imgs/current/eval/data/RESUME_model.pt"
config_path = "/media/data/GitHub/simple_hwr/RESULTS/OFFLINE_PREDS/RESUME_model/dtw_adaptive.yaml"
VOCAB_OVERRIDE = 0

if False:
    # BIG ONE
    vers = "v5BIG"
    load_path_override = "/media/data/GitHub/simple_hwr/results/stroke_config/OFFLINE_PREDS/v5BIG/RESUME_modelv5BIG.pt"
    config_path = "/media/data/GitHub/simple_hwr/results/stroke_config/OFFLINE_PREDS/v5BIG/RESUME.yaml"
    VOCAB_OVERRIDE = 5

if True:
    # BIG ONE - live
    vers = "v6BIG"
    load_path_override = f"/media/data/GitHub/simple_hwr/results/stroke_config/OFFLINE_PREDS/{vers}/RESUME_model{vers}.pt"
    config_path = f"/media/data/GitHub/simple_hwr/results/stroke_config/OFFLINE_PREDS/{vers}/RESUME{vers}.yaml"
    VOCAB_OVERRIDE = 5

if False:
    # V4
    vers="v4.1"
    load_path_override = "/media/data/GitHub/simple_hwr/results/stroke_config/OFFLINE_PREDS/v4.1/RESUME_modelv4.1.pt"
    config_path =        "/media/data/GitHub/simple_hwr/results/stroke_config/OFFLINE_PREDS/v4.1/RESUME.yaml"

    if False:
        vers="v4"
        load_path_override = "/media/data/GitHub/simple_hwr/results/stroke_config/OFFLINE_PREDS/v4/RESUME_modelv4.pt"
        config_path =        "/media/data/GitHub/simple_hwr/results/stroke_config/OFFLINE_PREDS/v4/RESUME.yaml"

    TRUNCATE=False
    VOCAB_OVERRIDE = 5 ## IDK WHY THIS IS 5, should have been 4

# OVERLOAD

ONLINE = False
TESTING = False # TESTING
TRUNCATE = False # use EOS token if available
TRUNCATE_DISTANCES = True # cut off last 20 if EOS token not available

if not ONLINE:
    TRUNCATE=True


ONLINE_OR_OFFLINE = "ONLINE" if ONLINE else "OFFLINE"
KDTREE_PATH = "/media/data/GitHub/simple_hwr/RESULTS/{}_PREDS/kd_trees{}.npy".format(ONLINE_OR_OFFLINE, vers)
LOAD_KDTREE = False
SAVE_KD = False
NN_DISTANCES = False
SAVE_ALL_IMGS = False # but it always saves a sample
SHUFFLE = True if not SAVE_ALL_IMGS else False

MASTER_LIST = ["b04-162-05","m01-090-04","g04-017-01","g06-026r-04","g06-037e-02","e04-091-01","a04-103-08","a02-124-03","e04-103-01","a01-132-07"]
#MASTER_LIST = []

def eval_only(dataloader, model, output_path=None):
    distances = []
    final_out = []
    if Path(KDTREE_PATH).exists() and LOAD_KDTREE:
        kd_trees = np.load(KDTREE_PATH, allow_pickle=True).item()
    else:
        kd_trees = {}
    for i, item in enumerate(tqdm(dataloader)):
        preds = TrainerStrokeRecovery.eval(item["line_imgs"],
                                           model,
                                           label_lengths=item["label_lengths"], #if not ONLINE else item['img_widths'],
                                           relative_indices=config.pred_relativefy,
                                           sigmoid_activations=config.sigmoid_indices)

        output = []
        names = [Path(p).stem.lower() for p in item["paths"]]

        # Pred comes out of eval WIDTH x VOCAB
        preds_to_graph = []
        for ii, p in enumerate(preds): # Loop within batch
            item_number = i*config.batch_size+ii
            name = names[ii]
            kd = kd_trees[name] if name in kd_trees else None

            # MOVE PRED TO MATCH GT
            gt = convert_reference(item["line_imgs"][ii], threshold=0)
            if TRUNCATE:
                remove_end = -min(40 if not ONLINE else 1, int(p.shape[0]/2))
            elif p.shape[-1]>=4:
                eos = np.argmax(p[:,3]>.7)
                remove_end = eos +1 if eos >= 300 else None
            else:
                remove_end = None

            pred, distance, kd = post_process(p[:remove_end], gt, calculate_distance=NN_DISTANCES, kd=kd)

            # MOVE GT TO MATCH PRED - this will improve as you increase the number of samples, not what we want right now
            if NN_DISTANCES and False: #item_number < 0:
                _, distances2, _ = stroke_recovery.get_nearest_point(p, gt, reference_is_image=False)
                avg_distance2 = np.average(distances2)
            else:
                distances2 = []
                avg_distance2 = 0

            # Warning if the name already exists in the dictionary and it wasn't loaded
            if name in kd_trees and not Path(KDTREE_PATH).exists():
                print(f"{name} already in KDTree dict")
            if SAVE_KD:
                kd_trees[name] = kd

            preds_to_graph.append(pred.transpose([1, 0])) # Convert to VOCAB X WIDTH
            path = item['paths'][ii]

            if TRUNCATE_DISTANCES and not TRUNCATE:
                avg_distance = np.average(distance[:-20])
            else:
                avg_distance = np.average(distance) #- (2*.5**2)**.5 / 61

            new_stem = path.stem + f"_{str(avg_distance)[:8].replace('.',',')}"
            #print(distance, new_stem)
            item["paths"][ii] = (path.parent / new_stem).with_suffix(path.suffix)

            if "_" in name:
                name = name[:name.find("_")]
            if name in GT_DATA:
                #p = preds[ii].detach().numpy()
                # Resample p
                pred[0,2]=1 # make the first point a start point
                p = stroke_recovery.resample(pred)

                # _, distance = stroke_recovery.get_nearest_point(item["line_imgs"][ii], p, reference_is_image=True)
                output_entry = {"stroke": p,
                               "raw_pred": preds[ii],
                               "text": GT_DATA[name],
                               "id": name,
                               "distance": avg_distance,
                               "pts": p.shape,
                               "distance2": avg_distance2,
                               "pts2": gt.shape,
                               "img": item["line_imgs"][ii],
                               }

                if ONLINE:
                    output_entry.update({"gt_list": item['gt_list'][ii],
                                          "raw_gt": item['raw_gt'][ii]})

                output.append(output_entry)

            else:
                print(f"{name} not found")

        print("image number:", item_number)

        # Get GTs, save to file
        output_path = Path(output_path)
        if not output_path is None:
            img_folder = output_path / "imgs"
            img_folder.mkdir(exist_ok=True, parents=True)
        else:
            img_folder = "auto"

        if i<4 or SAVE_ALL_IMGS:
            # Save a sample
            save_folder = graph(item, preds=preds_to_graph,
                                _type="eval",
                                epoch="current",
                                config=config,
                                save_folder=img_folder,
                                max_plots=50)
            if output_path is None:
                output_path = (Path(save_folder) / "data")
                output_path.mkdir(exist_ok=True, parents=True)

        #utils.pickle_it(output, output_path / f"{i}.pickle")
        #np.save(output_path / f"{i}.npy", output)
        final_out += output

        distances += [x["distance"] for x in output]
        #print(np.sum(distances < .1) / len(distance))

        # IF TESTING
        if TESTING:
            break
    #utils.pickle_it(final_out, output_path / f"all_data.pickle")

    if True:
        l = len(final_out)
        m = math.ceil(l/1000)
        for i in range(m):
            np.save(output_path / f"all_data_{i}.npy", final_out[i*1000:(i+1)*1000])

    # else:
    #     out = output_path / f"all_data.hdf5"
    #     print(out)
    #     h = h5py.File(out)
    #     for k in final_out:
    #         h.create_dataset(k["id"], data=k)

    # Compute stats
    #distances = np.asarray([x["distance"] for x in final_out])
    distances = np.asarray(distances)
    avg = np.average(distances)
    sd = np.std(distances)
    threshold = np.sum(distances < .01)
    print(f"Average distance: {avg}")
    print(f"SD: {sd}")
    print(f"Count below .01: {threshold}")

    with open(output_path / f"stats.txt", "w") as ff:
        ff.write(f"{avg}, {sd}, {threshold}")
    plt.xlim(0,.015)
    plt.hist(distances)
    plt.savefig(output_path / f"stats.png")
    plt.close()

    if kd_trees and kd_trees[next(iter(kd_trees))]: # make sure it's not None
        np.save(KDTREE_PATH, kd_trees)
    print("Files saved to: ", output_path)
    logger.info(f"Output size: {len(final_out)}")
    logger.info("ALL DONE")

# Loading GTs for offline data
def load_all_gts(gt_path, extension="*.json"):
    global GT_DATA
    from hwr_utils.hw_dataset import HwDataset
    data = HwDataset.load_data(data_paths=gt_path.glob(extension))
    #{'gt': 'He rose from his breakfast-nook bench', 'image_path': 'prepare_IAM_Lines/lines/m01/m01-049/m01-049-00.png',
    GT_DATA = {}
    for i in data:
        key = Path(i["image_path"]).stem.lower()
        assert not key in GT_DATA
        GT_DATA[key] = i["gt"]
    #print(f"GT's found: {GT_DATA.keys()}")
    np.save(rf"./RESULTS/{ONLINE_OR_OFFLINE}_PREDS/TEXT.npy", GT_DATA)
    return GT_DATA

if __name__=="__main__":
    opts = parse_args()
    config_path = config_path if True else opts.config
    main(config_path=config_path)
    # gt_path = Path("./data/prepare_IAM_Lines/gts/lines/txt")
    # load_all_gts(gt_path)
