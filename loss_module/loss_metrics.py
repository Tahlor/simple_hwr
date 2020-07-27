import torch
import robust_loss_pytorch
import numpy as np
import torch.nn as nn
from torch import Tensor
from pydtw import dtw
from scipy import spatial
from robust_loss_pytorch import AdaptiveLossFunction
#from sdtw import SoftDTW
import torch.multiprocessing as multiprocessing
from hwr_utils.utils import to_numpy, Counter
from hwr_utils.stroke_recovery import relativefy
from hwr_utils.stroke_dataset import pad, create_gts_from_fn
from scipy.spatial import KDTree
import time
from hwr_utils.stroke_recovery import distance_metric

def start_point_is_correct(preds, gts):
    distance_metric(preds[0,:2], gts[0,:2])
    pass


def juncture_is_correct(gts, preds):
    # Algorithm: make segments along the function, then O(n^2) check
    # the segments against each other for intersection.  If an intersection is
    # found, move along the path until finding a postprocessed prediction point,
    # then ensure that the postprocessed prediction points are in the same order
    # in time as the ground truth points.
    def postprocess(preds, kd, gts):
        _, closest = kd.query(preds)
        return [gts[i] for i in closest], set(closest), closest

    def seek(i, move, corrected, which):
        while i not in corrected:
            i = move(i)
        first_pred_match = next((j for j in range(len(which)) if which[j] == i), None)
        return first_pred_match

    def line_line(p1, q1, p2, q2):
        line1 = (p1, q1)
        line2 = (p2, q2)
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

    def intersect(p1, q1, p2, q2):
        def on_segment(p, q, r):
            return q[0] < max(p[0], r[0]) and q[1] > min(p[0], r[0]) and \
                   q[1] < max(p[1], r[1]) and q[1] > min(p[1], r[1])

        def orientation(p, q, r):
            v = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            return 0 if v == 0 else (1 if v > 0 else 2)

        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        if o1 != o2 and o3 != o4:
            return line_line(p1, q1, p2, q2)

        if o1 == 0 and on_segment(p1, p2, q1):
            return line_line(p1, q1, p2, q2)
        if o2 == 0 and on_segment(p1, q2, q1):
            return line_line(p1, q1, p2, q2)
        if o3 == 0 and on_segment(p2, p1, q2):
            return line_line(p1, q1, p2, q2)
        if o4 == 0 and on_segment(p2, q1, q2):
            return line_line(p1, q1, p2, q2)

    kd = KDTree(gts)
    gt_seg = [(gts[i - 1], gts[i]) for i in range(1, len(gts))]
    post, corrected, which_corrected = postprocess(preds, kd, gts)
    for i in range(len(gt_seg) - 1):
        for j in range(i + 2, len(gt_seg)):
            correct = False
            gt_int = intersect(gt_seg[i][0], gt_seg[i][1], gt_seg[j][0], gt_seg[j][1])
            if gt_int is not None:
                # move forward and back on the line until get corrected
                a = seek(i - 1, lambda x: x - 1, corrected, which_corrected)
                b = seek(i, lambda x: x + 1, corrected, which_corrected[a + 1:])
                if b is not None:
                    c = seek(j - 1, lambda x: x - 1, corrected, which_corrected[a + b + 1:])
                    if c is not None:
                        d = seek(j, lambda x: x + 1, corrected, which_corrected[a + b + c + 1:])
                        if d is not None:
                            correct = True
                if not correct:
                    return False
    return True

def calculate_nn_distance(item, preds):
    """ Can this be done differentiably?

    Args:
        item:
        preds:

    Returns:

    """
    # calculate NN distance
    n_pts = 0
    cum_dist = 0
    gt = item["gt_list"]
    batch_size = len(gt)
    for i in range(batch_size):
        # TODO binarize line images and do dist based on that
        kd = KDTree(preds[i][:, :2].detach().numpy())
        cum_dist = sum(kd.query(gt[i][:, :2])[0])  # How far do we have to move the GT's to match the predictions?
        n_pts += gt[i].shape[0]

    return cum_dist  # THIS WILL BE DIVIDED BY THE NUMBER OF PTS!! LATER
    # OLD METHOD: (cum_dist / n_pts) * batch_size
    # print("cum_dist: ", cum_dist, "n_pts: ", n_pts)
    # print("Distance: ", cum_dist / n_pts)
