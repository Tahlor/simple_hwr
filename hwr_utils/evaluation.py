import pydtw
import numpy as np
from pydtw import dtw
from hwr_utils.stroke_recovery import relativefy_numpy

def start_point_eval(pred, targ, already_stroke_number=False):
    """

    Args:
        pred:
        targ:

    Returns:

    """
    if not already_stroke_number:
        pred = relativefy_numpy(pred, reverse=True)
        targ = relativefy_numpy(targ, reverse=True)

    # give preds/targs
    # dynamic time warp them
    # assign the pred the "true" stroke number
    # compare first item in each stroke

    x1 = np.ascontiguousarray(pred[:,0:2])  # time step, (x,y)
    x2 = np.ascontiguousarray(targ[:,0:2])
    dist, cost, a, b = dtw.dtw2d(x1, x2)  # dist, cost, a, b

    pred = pred[a, 0:3]
    targ = targ[b, 0:3]
