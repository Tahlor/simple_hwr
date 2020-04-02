from stroke_dataset import StrokeRecoveryDataset, read_img
from stroke_plotting imoprt *
from torch.utils.data import DataLoader
from pathlib import Path
import os

def loader():
    folder = Path("online_coordinate_data/3_stroke_vSmall")
    folder = Path("online_coordinate_data/8_stroke_vSmall_16")
    folder = Path("online_coordinate_data/MAX_stroke_vTEST_AUGMENTFull")

    print(os.getcwd())
    x_relative_positions = True
    test_size = 5
    train_size = None
    batch_size = 32

    test_dataset = StrokeRecoveryDataset([folder / "test_online_coords.json"],
                                         img_height=60,
                                         num_of_channels=1.,
                                         max_images_to_load=test_size,
                                         root=r"../../data",
                                         )

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=3,
                                 collate_fn=test_dataset.collate,
                                 pin_memory=False)

    device = "cuda"
    globals().update(locals())
    return test_dataloader

def test_drawing(test_dataloader):
    example = next(iter(test_dataloader))  # BATCH, WIDTH, VOCAB

    instance = example["gt_list"][0].numpy()
    draw_from_gt(instance)
    draw_from_raw(gt_to_raw(example["gt_list"][0]))


if __name__=='__main__':
    test_dataloader = loader()
    test_drawing(test_dataloader)