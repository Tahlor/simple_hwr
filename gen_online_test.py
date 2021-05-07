from hwr_utils import visualize
from torch.utils.data import DataLoader
from torch import nn
from loss_module.stroke_recovery_loss import StrokeLoss
from trainers import *
from hwr_utils.stroke_dataset import StrokeRecoveryDataset, collate_stroke
from hwr_utils.stroke_recovery import *
from hwr_utils import utils
from torch.optim import lr_scheduler
from timeit import default_timer as timer
import argparse
from hwr_utils.hwr_logger import logger
from loss_module import losses
from models import start_points, stroke_model
from models.AlexGraves import AlexGravesCombined
from hwr_utils.stroke_plotting import *
from hwr_utils.utils import update_LR, reset_LR
from hwr_utils.stroke_plotting import draw_from_gt
import train_stroke_recovery
from train_stroke_recovery import *

def graph_procedure(preds, item, epoch=None, _type="train", other=None):
    # GRAPH
    if epoch is None:
        epoch = config.counter.epochs
    preds_to_graph = [p.permute([1, 0]) for p in preds]
    save_folder = graph(item, config=config, preds=preds_to_graph, _type=_type, epoch=epoch, max_plots=np.inf)

def test(dataloader):
    for i, item in enumerate(dataloader):
        loss, preds, y_hat, *_ = trainer.test(item, return_preds= i == 0) #
        graph_procedure(preds, item, epoch=None, _type="test",other=y_hat)

# train_stroke_recovery.test = test
# train_stroke_recovery.graph_procedure = graph_procedure

if __name__=="__main__":
    opts = parse_args()
    main(config_path=opts.config, testing=opts.testing, eval_once=True, eval_only=True)
    
    # TO DO:
        # logging
        # Get running on super computer - copy the data!