import numpy as np
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

# What if we only target i's?
# Very short strokes
# Swap over if the furtherest left stroke of next stroke is further right than the current one

ALREADY_SWAPPED = []

def check_already_swapped(stroke):
    global ALREADY_SWAPPED

    for i in ALREADY_SWAPPED:
        if len(i)==len(stroke) and np.allclose(i, stroke):
            return True
    ALREADY_SWAPPED.append(stroke)
    return False

def test_drawing_swaps(filename, gt1, original_gt, show=False):
    # print(gt.shape)
    data = draw_from_gt(gt1, show=False, right_padding="random", color=(0, 0, 0), linewidth=3, use_stroke_number=False)[
           ::-1]
    img = Image.fromarray(data, "RGB")
    # img.show()

    # Create an image
    swapped_strokes = Image.new("RGB", (data.shape[:2][::-1]), (255, 255, 255))
    swapped_strokes.paste(img)
    draw = ImageDraw.Draw(swapped_strokes)
    gt1[:, :2] *= 61
    gt_pil = list(gt_to_pil(gt1, stroke_number=False))

    original_gt[:, :2] *= 61
    new_gt_pil = list(gt_to_pil(original_gt, stroke_number=False))
    adaptation = False
    for i in range(len(new_gt_pil)):
        if len(gt_pil[i]) != len(new_gt_pil[i]) or not np.allclose(gt_pil[i], new_gt_pil[i]):
            # print(gt_pil[i][:10])
            # Draw reversals in red
            if len(gt_pil[i]) == len(new_gt_pil[i]) and np.allclose(gt_pil[i], new_gt_pil[i][::-1]):
                draw.line(gt_pil[i], fill=(255, 0, 0), width=2)
                draw_ellipse(new_gt_pil[i], draw, color=(0, 255, 0), width=2) # draw green circle on start point of original GT
            else:
                if not check_already_swapped(gt_pil[i]):
                    draw.line(gt_pil[i], fill=(0, 0, 255), width=2)
                    draw_ellipse(gt_pil[i], draw, color=(0, 0, 255), width=2)
                if not check_already_swapped(new_gt_pil[i]):
                    draw.line(new_gt_pil[i], fill=(0, 255, 0), width=1)
                    draw_ellipse(new_gt_pil[i], draw, color=(0, 255, 0), width=2)

            adaptation = True
    if adaptation:
        final = Image.fromarray((np.array(swapped_strokes)[::-1]), "RGB")
        if show:
            final.show()
        else:
            final.save(f"./swaps/{Path(filename).name}")

def draw_ellipse(point, draw, color=(0, 0, 255), width=2):
    """ Draw on first point

    Args:
        point:
        draw:
        color:
        width:
    Returns:

    """
    line1 = [x - width / 2 for x in point[0]]
    line2 = [x + width / 2 for x in point[0]]
    point = np.r_[line1, line2].flatten().tolist()
    draw.ellipse(point, fill=color, outline=color)

    # overlay_images(background, instance)

## Make sure these aren't the same data
import numpy as np
x = np.load(utils.get_project_root() / "RESULTS/pretrained/adapted_v2/training_dataset.npy", allow_pickle=True)
y = np.load("/media/data/GitHub/simple_hwr/results/stroke_config/ver8/20200408_143832-dtw_adaptive_new2_restartLR_RESUME/training_dataset.npy", allow_pickle=True)

adapted, correct = {}, {}
for i in y:
	correct[i["image_path"]] = i["gt"]

for i in x:
	adapted[i["image_path"]] = i["gt"]

for i in adapted:
    try:
    	np.testing.assert_allclose(adapted[i][:5], correct[i][:5])
    except:
        test_drawing_swaps(i, gt1=adapted[i], original_gt=correct[i])

print(adapted[i][:20])
print(correct[i][:20])

if False:
    preds_combo = [[0, 1, 1, 11],
                   [4, 5, 0, 7],
                   [8, 9, 0, 3],
                   [20, 21, 1, 23],
                   [12, 13, 1, 15],
                   [16, 17, 0, 19],
                   [32, 33, 1, 27],
                   [28, 29, 0, 31],
                   [24, 25, 0, 35]]

    preds_combo = np.asarray(preds_combo)
    k = []
    preds_combo[:, 2] = np.cumsum(preds_combo[:, 2])

    for i in range(0, 32):
        k.append({"gt": np.asarray(preds_combo), "image_path": f"{i}"})

    np.save("training_dataset.npy", k)

    gt = np.array(range(36)).reshape(9, 4).astype(np.float64)
    gt[:, 2] = np.cumsum([1, 0, 0, 1, 0, 1, 1, 0, 0])