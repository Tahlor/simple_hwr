## WHY ARE THERE UNEQUAL RECONSTRUCTIONS / ORIGINALS

import cv2
from pathlib import Path
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import re


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
TYPE=np.float32

class OfflineImages(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.root_dir = Path(root_dir)
        self.ys, self.y_hats, self.idx_of_ids = self.build_index()

    def build_index(self):
        original = {}
        reconstructed = {}
        regex_id = re.compile("[a-z][0-9]+-[0-9]+[0-9a-z]+-[0-9]+")

        for f in (self.root_dir).glob("original*.png"):
            id = regex_id.search(f.name)[0]
            original[id] = f

        for f in (self.root_dir).glob("reconstruction*.png"):
            id = regex_id.search(f.name)[0]
            reconstructed[id] = f

        for k in original.keys():
            assert k in reconstructed.keys()
        for k in reconstructed.keys():
            try:
                assert k in original.keys()
            except:
                print(k)

        #assert len(original.keys()) == len(reconstructed.keys())
        idx = list(original.keys())
        idx.sort()
        return reconstructed, original, idx

    def __len__(self):
        return len(self.idx_of_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        id = self.idx_of_ids[idx]
        original_path = self.ys[id].as_posix()
        recon_path = self.y_hats[id].as_posix()
        original_image = cv2.cvtColor(np.array(io.imread(original_path)), cv2.COLOR_BGR2GRAY)[:,:,None].astype(TYPE)
        recon_image = _recon_image = cv2.cvtColor(np.array(io.imread(recon_path)), cv2.COLOR_BGR2GRAY)[:,:,None].astype(TYPE)
        y, x, c = original_image.shape
        yr,xr,cr = recon_image.shape
        if x > xr:
            recon_image = np.ones(original_image.shape).astype(TYPE)
            recon_image[:yr,:xr] = _recon_image
        return original_image.transpose([2,0,1])/255.0, recon_image.transpose([2,0,1])[:,:y,:x]/255.0, id

if __name__=='__main__':
    dtw = "/media/data/GitHub/simple_hwr/RESULTS/OFFLINE_PREDS/RESUME_Bigger_Window_model/imgs/current/eval"
    o = OfflineImages(dtw)
    dataloader = DataLoader(o, batch_size=1,
                            shuffle=True, num_workers=5)
    m = next(iter(dataloader))
