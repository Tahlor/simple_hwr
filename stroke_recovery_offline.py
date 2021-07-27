from collections.abc import Iterable

import shutil
from models.basic import CNN, BidirectionalRNN
from torch import nn
from models.CoordConv import CoordConv
from hwr_utils import visualize
from torch.utils.data import DataLoader
from loss_module.stroke_recovery_loss import StrokeLoss
from trainers import TrainerStrokeRecovery
from hwr_utils.stroke_dataset import BasicDataset
from hwr_utils.stroke_recovery import *
from  hwr_utils import stroke_recovery
from hwr_utils import utils
from torch.optim import lr_scheduler
from models.stroke_model import StrokeRecoveryModel
from train_stroke_recovery import parse_args, graph
from hwr_utils.hwr_logger import logger
from pathlib import Path
import os
from tqdm import tqdm
from subprocess import Popen

# Define root eval_img_path_override where the offline_data is and where the GT's are
PROJ_ROOT = os.path.dirname(os.path.realpath(__file__))
eval_img_path_override = PROJ_ROOT / Path("data/prepare_IAM_Lines/lines/")
eval_gt_path_override = PROJ_ROOT / Path("data/prepare_IAM_Lines/gts/lines/txt")
config_path = "/media/data/GitHub/simple_hwr/example_weights/config.yaml"
model_load_path_override = ""

pid = os.getpid()

#@debugger
def main(config_path):
    global epoch, device, trainer, batch_size, output, loss_obj, x_relative_positions, config, LOGGER
    torch.cuda.empty_cache()

    config = utils.load_config(config_path, hwr=False)
    if model_load_path_override:
        load_path_override = model_load_path_override
    else:
        load_path_override = config.load_path
    _load_path_override=Path(load_path_override)
    OUTPUT =Path(config.results_dir)

    _t = utils.increment_path(name="eval", base_path=OUTPUT / "imgs/current")
    model_output_dir = OUTPUT / _t / "data"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(_load_path_override, model_output_dir) # save the model ya hoser

    # Make these the same as whereever the file is being loaded from; make the log_dir and results dir be a subset
    # main_model_path, log_dir, full_specs, results_dir, load_path

    config.use_visdom = False
    # Free GPU memory if necessary
    if config.device == "cuda":
        utils.kill_gpu_hogs()

    batch_size = config.batch_size

    vocab_size = config.feature_map_dim

    device=torch.device(config.device)
    #device=torch.device("cpu")

    output = utils.increment_path(name="Run", base_path=Path(load_path_override).parent)
    #output = Path(config.results_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Image data path overrides
    folder = Path(config.dataset_folder) if not eval_img_path_override else eval_img_path_override
    gt_path = Path(config.dataset_folder) if not eval_gt_path_override else eval_gt_path_override


    #model = StrokeRecoveryModel(vocab_size=vocab_size, device=device, cnn_type=config.cnn_type, first_conv_op=config.coordconv, first_conv_opts=config.coordconv_opts).to(device)
    model = StrokeRecoveryModel(**config.model_definition).to(device)
    ## Loader
    logger.info(("Current dataset: ", folder))
    # Dataset - just expecting a eval_img_path_override
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
    config.load_path = load_path_override if ("load_path_override" in locals() and load_path_override) else config.load_path

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

LOAD_KDTREE = False
NO_KD = True

def eval_only(dataloader, model):
    distances = []
    final_out = []
    kd_trees = {}
    for i, item in enumerate(tqdm(dataloader)):
        preds = TrainerStrokeRecovery.eval(item["line_imgs"],
                                           model,
                                           label_lengths=item["label_lengths"],
                                           relative_indices=config.pred_relativefy,
                                           sigmoid_activations=config.sigmoid_indices)

        output = []
        names = [Path(p).stem.lower() for p in item["paths"]]

        # Pred comes out of eval WIDTH x VOCAB
        preds_to_graph = []
        for ii, p in enumerate(preds): # Loop within batch
            item_number = i*config.batch_size+ii
            print("image number:", item_number)

            name = names[ii]
            kd = kd_trees[name] if name in kd_trees else None

            # MOVE PRED TO MATCH GT
            gt = convert_reference(item["line_imgs"][ii], threshold=-.25)
            pred, distance, kd = post_process(p, gt, calculate_distance=not NO_KD, kd=kd)

            # MOVE GT TO MATCH PRED - too expensive
            if item_number < 3:
                _, distances2, _ = stroke_recovery.get_nearest_point(p, gt, reference_is_image=False)
                avg_distance2 = np.average(distances2)
            else:
                distances2 = []
                avg_distance2 = 0

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
                               "distance": avg_distance,
                               "pts": len(distance) if isinstance(distance, Iterable) else 0,
                               "distance2": avg_distance2,
                               "pts2": len(distances2),
                               })
            else:
                print(f"{name} not found")


        # Get GTs, save to file
        if True or i<4:
            # Save a sample
            save_folder = graph(item, preds=preds_to_graph, _type="eval", epoch="current", config=config)
            output_path = (Path(save_folder) / "data")
            output_path.mkdir(exist_ok=True, parents=True)

        #utils.pickle_it(output, output_path / f"{i}.pickle")
        #np.save(output_path / f"{i}.npy", output)
        final_out += output

        distances += [x["distance"] for x in output]
        #print(np.sum(distances < .1) / len(distance))
    #utils.pickle_it(final_out, output_path / f"all_data.pickle")
    np.save(output_path / f"all_data.npy", final_out)

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

    if kd_trees[next(iter(kd_trees))]: # make sure it's not None
        np.save(KDTREE_PATH, kd_trees)
    logger.info(f"Output size: {len(final_out)}")
    logger.info("ALL DONE")

# Loading GTs for offline data
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
    main(config_path=config_path)