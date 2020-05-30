from collections import defaultdict
import shutil
import traceback
import time
from models.basic import CNN, BidirectionalRNN
from torch import nn
from models.CoordConv import CoordConv
from hwr_utils import visualize
from torch.utils.data import DataLoader
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
from hwr_utils.stroke_dataset import *
from tqdm import tqdm
from subprocess import Popen

pid = os.getpid()

#@debugger
def main(config_path):
    global epoch, device, trainer, batch_size, output, loss_obj, x_relative_positions, config, LOGGER
    torch.cuda.empty_cache()

    PROJ_ROOT= Path(os.path.dirname(os.path.realpath(__file__)))
    config_path = PROJ_ROOT / "server/RESUME.yaml"
    load_path_override = PROJ_ROOT /  "server/RESUME_model.pt"

    for load_path_override in [load_path_override]:
        _load_path_override = Path(load_path_override)

        # Make these the same as whereever the file is being loaded from; make the log_dir and results dir be a subset
        # main_model_path, log_dir, full_specs, results_dir, load_path
        config = utils.load_config(config_path, hwr=False, results_dir_override=None)

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
        folder = Path(config.dataset_folder)

        # OVERLOAD
        if True:
            folder = PROJ_ROOT / Path("data/prepare_IAM_Lines/lines/")
            gt_path = PROJ_ROOT / Path("data/prepare_IAM_Lines/gts/lines/txt")
        else:
            folder = Path("/media/data/GitHub/simple_hwr/data/prepare_IAM_Lines/words")
            gt_path = PROJ_ROOT / Path("data/prepare_IAM_Lines/gts/words")

        model = StrokeRecoveryModel(vocab_size=vocab_size, device=device, cnn_type=config.cnn_type, first_conv_op=config.coordconv, first_conv_opts=config.coordconv_opts).to(device)

        trainer = TrainerStrokeRecovery(model, optimizer=None, config=config, loss_criterion=None)
        config.model = model
        config.load_path = load_path_override if ("load_path_override" in locals()) else config.load_path
        config.sigmoid_indices = TrainerStrokeRecovery.get_indices(config.pred_opts, "sigmoid")

        ## LOAD THE WEIGHTS
        utils.load_model_strokes(config, device="cpu", load_optimizer=False) # should be load_model_strokes??????
        model = model.to(device)
        model.eval()
        wait(model)


OUTPUT_PATH = "/home/mason/Desktop/redis_stroke_recovery/results"
INPUT_PATH = "/home/mason/Desktop/redis_stroke_recovery/raw"
failed = defaultdict(int)
give_up = []
def wait(model):
    #     output_image_path = "/home/mason/Desktop/redis_stroke_recovery/data/a01-000u-00.png"
    while True:
        try:
            completed_files = [x.stem for x in Path(OUTPUT_PATH).rglob("*.png")]
            for item in Path(INPUT_PATH).rglob("*"):
                if item.is_file() and item not in give_up:
                    if not f"0_{item.stem}" in completed_files:
                        print(item)
                        do_one(model, str(item))
        except:
            print(f"{item.stem} failed")
            failed[item.stem] += 1
            traceback.print_exc()
            if not f"0_{item.stem}" in completed_files and failed[item.stem] > 5:
                shutil.copy("./server/Well-that-didn-t-work.png", Path(OUTPUT_PATH) / f"0_{item.stem}.png")
                shutil.copy("./server/Well-that-didn-t-work.png", Path(OUTPUT_PATH) / f"overlay_0_{item.stem}.png")
                give_up.append(item)
        time.sleep(1)

def do_one(model, img_path):
    # Prep image
    # token = "ABC"
    # output_path = f"{token}.png"
    item = BasicDataset.get_item_from_path(image_path=img_path, output_path=img_path)
    item['line_imgs'] = Tensor(item['line_img']).unsqueeze(0).permute(0,3,1,2)
    eval_only(item, model)

def eval_only(item, model):
    final_out = []

    preds = TrainerStrokeRecovery.eval(item["line_imgs"], model,
                                       label_lengths=None,
                                       relative_indices=config.pred_relativefy,
                                       sigmoid_activations=config.sigmoid_indices,
                                       device='cpu')

    output = []

    # Pred comes out of eval WIDTH x VOCAB
    pred = preds[0].numpy()
    output.append(pred.transpose([1, 0])) # Convert to VOCAB X WIDTH
    path = item['path']
    item['paths'] = [path]
    save_folder = graph(item, preds=output, _type="eval", epoch="current", config=config, save_folder=OUTPUT_PATH)
    output_path = (Path(save_folder) / "data")
    output_path.mkdir(exist_ok=True, parents=True)

    #utils.pickle_it(output, output_path / f"{i}.pickle")
    #np.save(output_path / f"{i}.npy", output)
    final_out += output


if __name__=="__main__":
    opts = parse_args()
    main(config_path=opts.config)
    # gt_path = Path("./data/prepare_IAM_Lines/gts/lines/txt")
    # load_all_gts(gt_path)
