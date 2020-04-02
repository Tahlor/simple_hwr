import tempfile
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from models.basic import CNN, BidirectionalRNN
from torch import nn
from loss_module.stroke_recovery_loss import StrokeLoss
import torch
from models.CoordConv import CoordConv
from trainers import TrainerStrokeRecovery
from hwr_utils.stroke_dataset import StrokeRecoveryDataset, read_img
from hwr_utils.stroke_recovery import *
from hwr_utils import utils, stroke_recovery
from torch.optim import lr_scheduler
from timeit import default_timer as timer
from train_stroke_recovery import graph
from hwr_utils.distortions import *
from hwr_utils.stroke_plotting import *

## DRAW THE IMAGE
# Pillow-simd

from PIL import Image, ImageDraw


def loader():
    global test_dataloader, test_dataset
    folder = Path("online_coordinate_data/3_stroke_vSmall")
    folder = Path("online_coordinate_data/8_stroke_vSmall_16")
    # folder = Path("online_coordinate_data/MAX_stroke_vTEST_AUGMENTFull")

    print(os.getcwd())
    x_relative_positions = True
    test_size = 5
    train_size = None
    batch_size = 32

    test_dataset = StrokeRecoveryDataset([folder / "test_online_coords.json"],
                                         img_height=61,
                                         num_of_channels=1.,
                                         max_images_to_load=test_size,
                                         root=r"data",
                                         cnn=None,

                                         )

    # test_dataloader = DataLoader(test_dataset,
    #                              batch_size=batch_size,
    #                              shuffle=True,
    #                              num_workers=3,
    #                              collate_fn=test_dataset.collate,
    #                              pin_memory=False)
    #
    # device = "cuda"
    # globals().update(locals())
    # return test_dataloader


def reload_and_get_new_example():
    loader()
    return next(iter(test_dataset))


def get_instance():
    instance = next(iter(test_dataset))["gt"]
    return instance


loader()
# test_drawing()

# example = next(iter(test_dataloader)) # BATCH, WIDTH, VOCAB
# instance = example["gt_list"][0].numpy()
instance = get_instance()

new_gt = random_pad(instance, 15, 15)
draw_from_gt(new_gt, show=True, right_padding="random", linewidth=13, color=(0,0,0))#max_width=12)
