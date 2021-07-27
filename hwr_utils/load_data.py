from pathlib import Path
import sys, os, numpy as np
from hwr_utils.hw_dataset import HwDataset

DIR = Path(os.path.dirname(os.path.realpath(__file__)))
PROJ_ROOT = DIR.parent
sys.path.append(DIR)

gt_paths = [PROJ_ROOT / Path("data/prepare_IAM_Lines/gts/lines/txt"), PROJ_ROOT / Path("data/prepare_online_data/online_augmentation.json")]

def load_all_gts(gt_paths, PROJ_ROOT=PROJ_ROOT):
    gt_paths = [gt_paths] if isinstance(gt_paths, str) else gt_paths
    GT_DATA = {}
    for gt_path in gt_paths:
        if gt_path.suffix == ".json":
            data = HwDataset.load_data(data_paths=[gt_path])
        else:
            data = HwDataset.load_data(data_paths=gt_path.glob("*.json"))
        #{'gt': 'He rose from his breakfast-nook bench', 'image_path': 'prepare_IAM_Lines/lines/m01/m01-049/m01-049-00.png',
        for i in data:
            key = Path(i["image_path"]).stem.lower()
            if key in GT_DATA:
                assert i["gt"] == GT_DATA[key]
            GT_DATA[key] = i["gt"]
        #print(f"GT's found: {GT_DATA.keys()}")
    np.save((PROJ_ROOT / "RESULTS/OFFLINE_PREDS/TEXT.npy"), GT_DATA)
    return GT_DATA

if __name__=='__main__':
    load_all_gts(gt_paths)