# Calculate image scores - compare actual offline image to DTW, no DTW
# Add image of no DTW image

# Get rid of NN scores, keep DTW scores for online

from pathlib import Path
import numpy as np
import re
from custom_dataset import OfflineImages, DataLoader
import sys
sys.path.append("./Pytorch-metrics")
import metrics
import torch
from tqdm import tqdm

def calculate_folder(data_loader, baseline=False):
    metric_dict = {"MSE": metrics.MSE(),
                   "PSNR": metrics.PSNR(),
                   "SSIM": metrics.SSIM(),
                   "LPIPS": metrics.LPIPS("cuda"),
                   "F1Score": metrics.F1Score()}
    scores = {}
    for m in metric_dict.keys():
        scores[m] = []
    for y,yhat,id in tqdm(data_loader):
        # y_pred << 4D tensor in [batch_size, channels, img_rows, img_cols]
        # y_true << 4D tensor in [batch_size, channels, img_rows, img_cols]
        if baseline:
            yhat = torch.Tensor(np.ones(y.shape).astype(np.float32))
        for metric in metric_dict.keys():
            metric_fn = metric_dict[metric]
            scores[metric].append(metric_fn(yhat, y).item())

    print("Number of images", len(scores["MSE"]))
    for metric in metric_dict.keys():
        print(metric, np.average(scores[metric]))


if __name__=="__main__":
    no_dtw = "/media/data/GitHub/simple_hwr/results/NO_DTW/20210508_092011-NO_DTW/imgs/current/eval"
    dtw = "/media/data/GitHub/simple_hwr/RESULTS/OFFLINE_PREDS/RESUME_Bigger_Window_model/imgs/current/eval"
    l = no_dtw, dtw
    l = []
    for path in l:
        print(path)
        o = OfflineImages(path)
        dataloader = DataLoader(o, batch_size=1,
                                shuffle=False, num_workers=5)
        calculate_folder(dataloader)

    # Calculate scores with just white images
    o = OfflineImages(no_dtw)
    dataloader = DataLoader(o, batch_size=1,
                            shuffle=False, num_workers=5)

    calculate_folder(dataloader, baseline=True)