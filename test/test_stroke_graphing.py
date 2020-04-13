from torch.utils.data import DataLoader
from hwr_utils.stroke_dataset import StrokeRecoveryDataset
from hwr_utils.stroke_recovery import *
from train_stroke_recovery import graph
from hwr_utils.stroke_plotting import *

folder = Path("online_coordinate_data/3_stroke_vSmall")
folder = Path("online_coordinate_data/8_stroke_vSmall_16")


x_relative_positions = True
test_size = 2000
train_size = None
batch_size=1
config = {"model_name":"normal", "image_dir":Path("."), "gt_format":["x","y","sos","eos"], "dataset":{}}
config = edict(config)
config.dataset.image_prep = "pil"

test_dataset=StrokeRecoveryDataset([folder / "test_online_coords.json"],
                        img_height = 60,
                        num_of_channels = 1.,
                        max_images_to_load = test_size,
                        root=r"../data",
                        x_relative_positions=x_relative_positions,
                        gt_format = config.gt_format
                        )

test_dataloader = DataLoader(test_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=3,
                              collate_fn=test_dataset.collate,
                              pin_memory=False)

device="cuda"
example = next(iter(test_dataloader)) # BATCH, WIDTH, VOCAB

### PLOT
gt = example["gt_list"][0].numpy()
#pil = gt_to_pil_format(example["gt"][0])
draw_from_gt(gt)

## Normal
preds = [x.transpose(1,0) for x in example["gt_list"]]
graph(example, config, preds=preds, save_folder=None)


# Stroke number
example["gt_list"][0][:,2] = torch.cumsum(example["gt_list"][0][:,2], dim=0)
graph(example, config, preds=preds, save_folder=None)

#graph(example, save_folder=output)