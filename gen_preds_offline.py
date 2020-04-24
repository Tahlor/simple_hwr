from models.basic import CNN, BidirectionalRNN
from torch import nn
from models.CoordConv import CoordConv
from hwr_utils import visualize
from torch.utils.data import DataLoader
from loss_module.stroke_recovery_loss import StrokeLoss
from trainers import TrainerStrokeRecovery
from hwr_utils.stroke_dataset import BasicDataset
from hwr_utils.stroke_recovery import *
from hwr_utils import utils
from torch.optim import lr_scheduler
from models.stroke_model import StrokeRecoveryModel
from train_stroke_recovery import parse_args, graph
from hwr_utils.hwr_logger import logger
from pathlib import Path
import os
from tqdm import tqdm
from subprocess import Popen

pid = os.getpid()

#@debugger
def main(config_path):
    global epoch, device, trainer, batch_size, output, loss_obj, x_relative_positions, config, LOGGER
    torch.cuda.empty_cache()

    PROJ_ROOT= os.path.dirname(os.path.realpath(__file__))
    config_path = "/media/data/GitHub/simple_hwr/~RESULTS/20191213_155358-baseline-GOOD_long/TEST.yaml"
    config_path = "/media/SuperComputerGroups/fslg_hwr/taylor_simple_hwr/RESULTS/ver1/RESUME.yaml"
    config_path = PROJ_ROOT + "RESULTS/OFFLINE_PREDS/good/normal_preload.yaml"
    config_path = "/media/data/GitHub/simple_hwr/RESULTS/pretrained/dtw_v3/normal_preload.yaml"
    config_path = PROJ_ROOT + "/configs/stroke_configs/ver8/dtw_adaptive.yaml"

    load_path_override = "/media/SuperComputerGroups/fslg_hwr/taylor_simple_hwr/results/stroke_config/GOOD/baseline_model.pt"
    load_path_override = "/media/data/GitHub/simple_hwr/~RESULTS/20191213_155358-baseline-GOOD_long"
    load_path_override = "/media/SuperComputerGroups/fslg_hwr/taylor_simple_hwr/RESULTS/ver1/20200215_014143-normal/normal_model.pt"
    load_path_override = "/media/SuperComputerGroups/fslg_hwr/taylor_simple_hwr/RESULTS/ver2/20200217_033031-normal2/normal2_model.pt"
    load_path_override = "/media/data/GitHub/simple_hwr/RESULTS/pretrained/dtw_train_2.9/normal_preload_model.pt"
    load_path_override = "/media/data/GitHub/simple_hwr/RESULTS/pretrained/dtw_train_v2/v2.pt"
    load_path_override = PROJ_ROOT + "/RESULTS/pretrained/dtw_adaptive_new_model.pt"
    load_path_override = PROJ_ROOT + "/RESULTS/pretrained/adapted_v2/"
    load_path_override = "/home/taylor/shares/brodie/home/taylor/github/simple_hwr/RESULTS/ver8/20200406_131747-dtw_adaptive_new2_restartLR_RESUME/RESUME_model.pt"

    for load_path_override in [load_path_override
                               ]:

        #load_path_override = PROJ_ROOT + "RESULTS/pretrained/new_best/good.pt"

        #load_path_override = PROJ_ROOT + "RESULTS/OFFLINE_PREDS/all_data.npy"
        _load_path_override = Path(load_path_override)

        OUTPUT = PROJ_ROOT / Path("RESULTS/OFFLINE_PREDS/") / _load_path_override.stem
        model_output_dir = OUTPUT / "imgs/current/eval/data"
        model_output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(_load_path_override, model_output_dir) # save the model ya hoser

        # Make these the same as whereever the file is being loaded from; make the log_dir and results dir be a subset
        # main_model_path, log_dir, full_specs, results_dir, load_path


        config = utils.load_config(config_path, hwr=False, results_dir_override=OUTPUT.as_posix())
        config.use_visdom = False
        # Free GPU memory if necessary
        if config.device == "cuda":
            utils.kill_gpu_hogs()

        batch_size = config.batch_size

        vocab_size = config.vocab_size

        device=torch.device(config.device)
        #device=torch.device("cpu")

        output = utils.increment_path(name="Run", base_path=Path(load_path_override).parent)
        #output = Path(config.results_dir)
        output.mkdir(parents=True, exist_ok=True)
        folder = Path(config.dataset_folder)

        # OVERLOAD
        if True:
            folder = PROJ_ROOT / Path("data/prepare_IAM_Lines/lines/")
            gt_path = PROJ_ROOT / Path("data/prepare_IAM_Lines/gts/lines/txt")
        else:
            folder = Path("/media/data/GitHub/simple_hwr/data/prepare_IAM_Lines/words")
            gt_path = PROJ_ROOT / Path("data/prepare_IAM_Lines/gts/words")

        model = StrokeRecoveryModel(vocab_size=vocab_size, device=device, cnn_type=config.cnn_type, first_conv_op=config.coordconv, first_conv_opts=config.coordconv_opts).to(device)

        ## Loader
        logger.info(("Current dataset: ", folder))
        # Dataset - just expecting a folder
        eval_dataset=BasicDataset(root=folder, cnn=model.cnn, )
        next(iter(eval_dataset))
        eval_loader=DataLoader(eval_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=6,
                                      collate_fn=eval_dataset.collate, # this should be set to collate_stroke_eval
                                      pin_memory=False)
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
        config.load_path = load_path_override if ("load_path_override" in locals()) else config.load_path

        config.sigmoid_indices = TrainerStrokeRecovery.get_indices(config.pred_opts, "sigmoid")

        # Load the GTs
        GT_DATA = load_all_gts(gt_path)
        print("Number of images: {}".format(len(eval_loader.dataset)))
        print("Number of GTs: {}".format(len(GT_DATA)))

        ## LOAD THE WEIGHTS
        utils.load_model_strokes(config) # should be load_model_strokes??????
        model = model.to(device)
        model.eval()
        eval_only(eval_loader, model)
        globals().update(locals())

def post_process(pred,gt, calculate_distance=True, kd=None):
    #return make_more_starts(move_bad_points(reference=gt, moving_component=pred, reference_is_image=True), max_dist=.15)
    if calculate_distance:
        _, distances, kd = stroke_recovery.get_nearest_point(gt, pred, reference_is_image=True, kd=kd)
    else:
        distances = 0

    if True:
        return make_more_starts(pred, max_dist=.18), distances, kd
        # Move single points that are far from everything to nearest part on image
            # If that nearest point is far from other points
            # Else delete that point
    else:
        return pred.numpy(), distances, kd

KDTREE_PATH = "/media/data/GitHub/simple_hwr/RESULTS/OFFLINE_PREDS/kd_trees.npy"
def eval_only(dataloader, model):
    final_out = []
    if Path(KDTREE_PATH).exists():
        kd_trees = np.load(KDTREE_PATH, allow_pickle=True).item()
    else:
        kd_trees = {}
    for i, item in enumerate(tqdm(dataloader)):
        preds = TrainerStrokeRecovery.eval(item["line_imgs"], model,
                                           label_lengths=item["label_lengths"],
                                           relative_indices=config.pred_relativefy,
                                           sigmoid_activations=config.sigmoid_indices)

        output = []
        names = [Path(p).stem.lower() for p in item["paths"]]

        # Pred comes out of eval WIDTH x VOCAB
        preds_to_graph = []
        for ii, p in enumerate(preds): # Loop within batch
            name = names[ii]
            kd = kd_trees[name] if name in kd_trees else None

            pred, distance, kd = post_process(p, item["line_imgs"][ii], kd=kd)

            # Warning if the name already exists in the dictionary and it wasn't loaded
            if name in kd_trees and not Path(KDTREE_PATH).exists():
                print(f"{name} already in KDTree dict")
            kd_trees[name] = kd

            preds_to_graph.append(pred.transpose([1, 0])) # Convert to VOCAB X WIDTH
            path = item['paths'][ii]
            avg_distance = np.average(distance)
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
                output.append({"stroke": p,
                               "text": GT_DATA[name],
                               "id": name,
                               "distance": np.average(distance)
                               })
            else:
                print(f"{name} not found")


        # Get GTs, save to file
        if i < 4:
            # Save a sample
            save_folder = graph(item, preds=preds_to_graph, _type="eval", epoch="current", config=config)
            output_path = (Path(save_folder) / "data")
            output_path.mkdir(exist_ok=True, parents=True)

        #utils.pickle_it(output, output_path / f"{i}.pickle")
        #np.save(output_path / f"{i}.npy", output)
        final_out += output

    #utils.pickle_it(final_out, output_path / f"all_data.pickle")
    np.save(output_path / f"all_data.npy", final_out)

    # Compute stats
    distances = np.asarray([x["distance"] for x in final_out])
    avg = np.average(distances)
    sd = np.std(distances)
    threshold = np.sum(distances < .01)
    print(f"Average distance: {avg}")
    print(f"SD: {sd}")
    print(f"Count below .01: {threshold}")

    with open(output_path / f"stats.txt", "w") as ff:
        ff.write(f"{avg}, {sd}, {threshold}")
    plt.hist(distances)
    plt.savefig(output_path / f"stats.png")
    plt.close()

    if kd_trees[next(iter(kd_trees))]: # make sure it's not None
        np.save(KDTREE_PATH, kd_trees)
    logger.info(f"Output size: {len(final_out)}")
    logger.info("ALL DONE")

def load_all_gts(gt_path):
    global GT_DATA
    from hwr_utils.hw_dataset import HwDataset
    data = HwDataset.load_data(data_paths=gt_path.glob("*.json"))
    #{'gt': 'He rose from his breakfast-nook bench', 'image_path': 'prepare_IAM_Lines/lines/m01/m01-049/m01-049-00.png',
    GT_DATA = {}
    for i in data:
        key = Path(i["image_path"]).stem.lower()
        assert not key in GT_DATA
        GT_DATA[key] = i["gt"]
    #print(f"GT's found: {GT_DATA.keys()}")
    np.save(r"./RESULTS/OFFLINE_PREDS/TEXT.npy", GT_DATA)
    return GT_DATA

if __name__=="__main__":
    opts = parse_args()
    main(config_path=opts.config)
    # gt_path = Path("./data/prepare_IAM_Lines/gts/lines/txt")
    # load_all_gts(gt_path)
