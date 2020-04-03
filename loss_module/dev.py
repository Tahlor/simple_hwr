import numpy as np
import sys
# sys.path.append("../../")
# sys.path.append("../")
# sys.path.append("/media/data/GitHub/simple_hwr")
import os
import numpy as np
from hwr_utils import *
from hwr_utils.stroke_plotting import *
from hwr_utils.stroke_recovery import get_number_of_stroke_pts_from_gt
from hwr_utils.stroke_recovery import *
import json
from matplotlib import pyplot as plt
from taylor_dtw import custom_dtw as dtw
np.set_printoptions(precision=1)

### POSSIBLY GO BACK TO ORIGINAL HANDLING OF COST_MAT, AND USE cost_mat.base

# Original DTW
# Look at worst match - based on average distance
# Randomly sample among worst matches, based on how bad they are
# Attempt reversing/swapping
# Choose new GT
# Add some buffer
# Choose downstream/upstream baseline points
# Get distance of downstream point
# Refill cost matrix
    # Reuse upstream portion
    # Limit downstream portion
# Traceback
    # Finds new path
    # if better, replace path through window+buffer of original DTW
# Stroke to Dataloader GT
from forbiddenfruit import curse

def push_back(self, a):
    self.insert(0, a)

curse(list, "push_back", push_back)

# DTW each stroke iteratively
# DTW until stroke+1 is chosen
# stroke + i is the min of stroke/reverse stroke
# check whether the stroke or reverse stroke does better
INFINITY = 1000


def euclidean_distance(a, b):
    d = 0
    a = np.asarray(a)
    b = np.asarray(b)
    # print(a.shape,b)
    for i in range(a.shape[0]):
        tmp = a[i] - b[i]
        d += tmp * tmp
    return np.sqrt(d)


def d_argmin(a, b, c):
    if a <= b and a <= c:
        return 0
    elif b <= c:
        return 1
    else:
        return 2


def d_min(a, b, c):
    if a < b and a < c:
        return a
    elif b < c:
        return b
    else:
        return c


def traceback_partial(cost_mat, ilen, jlen, imin=0, jmin=0):
    cost = 0.0
    i = ilen - 1
    j = jlen - 1
    #     cdef vector[int] a
    #     cdef vector[int] b
    #     a.push_back(i)
    #     b.push_back(j)

    a = []
    b = []
    a.append(i)
    b.append(j)
    # cdef int match
    while (i > imin or j > jmin):
        match = d_argmin(cost_mat[i - 1, j - 1], cost_mat[i - 1, j], cost_mat[i, j - 1])
        if match == 0:
            i -= 1
            j -= 1
            cost += cost_mat[i - 1, j - 1]
        elif match == 1:
            i -= 1
            cost += cost_mat[i - 1, j]
        else:
            j -= 1
            cost += cost_mat[i, j - 1]
        a.push_back(i)
        b.push_back(j)
    return a, b, cost


def create_cost_mat_2d(a, b, constraint, dist_func=euclidean_distance):
    cost_mat = np.empty((a.shape[0] + 1, b.shape[0] + 1), dtype=np.float64)
    cost_mat[:] = INFINITY
    cost_mat[0, 0] = 0
    for i in range(1, cost_mat.shape[0]):
        for j in range(max(1, i - constraint), min(cost_mat.shape[1], i + constraint + 1)):
            cost_mat[i, j] = dist_func(a[i - 1], b[j - 1]) + \
                             d_min(cost_mat[i - 1, j], cost_mat[i, j - 1], cost_mat[i - 1, j - 1])

    return cost_mat[1:, 1:]


def traceback(cost_mat, ilen, jlen):
    i = ilen - 1
    j = jlen - 1
    #     cdef vector[int] a
    #     cdef vector[int] b
    #     a.push_back(i)
    #     b.push_back(j)

    cost_mat = cost_mat #[1:, 1:]
    cost = cost_mat[i, j]
    a = []
    b = []
    a.append(i)
    b.append(j)
    # cdef int match
    while (i > 0 or j > 0):
        match = d_argmin(cost_mat[i - 1, j - 1], cost_mat[i - 1, j], cost_mat[i, j - 1])
        if match == 0:
            i -= 1
            j -= 1
        elif match == 1:
            i -= 1
        else:
            j -= 1
        a.push_back(i)
        b.push_back(j)
    return a, b, cost

#from pydtw import dtw
# Original DTW
# Look at worst match - based on average distance
# Randomly sample among worst matches, based on how bad they are
# Attempt reversing/swapping
# Choose new GT
# Add some buffer
# Choose downstream/upstream baseline points
# Get distance of downstream point
# Refill cost matrix
# Reuse upstream portion
# Limit downstream portion
# Traceback
# Finds new path
# if better, replace path through window+buffer of original DTW
# Stroke to Dataloader GT

"""
for i in range(1, cost_mat.shape[0]):
    for j in range(max(1, i-constraint), min(cost_mat.shape[1], i+constraint+1)):
"""


def refill_cost_matrix_dev(a, b, cost_mat, start_a, end_a, start_b, end_b, constraint, metric=euclidean_distance):
    # Include some buffer beyond just the strokes being flipped
    # To get improvement, compare the cost at this point before and after
    # cost_mat = np.empty((a.shape[0] + 1, b.shape[0] + 1), dtype=np.float64)
    # cost_mat[:] = INFINITY
    # cost_mat[0, 0] = 0

    #     start_a = max(start_a - 1, 0)
    #     start_b = max(start_b - 1, 0)
    #     end_a = max(end_a - 1, 0)
    #     end_b = max(end_b - 1, 0)
    dist_func = euclidean_distance
    for i in range(start_a + 1, end_a + 1):
        for j in range(max(start_b + 1, i - constraint), min(end_b + 1, i + constraint + 1)):
            x = dist_func(a[i - 1], b[j - 1]) + \
                            d_min(cost_mat[i - 1, j], cost_mat[i, j - 1], cost_mat[i - 1, j - 1])

            cost_mat[i, j] = dist_func(a[i - 1], b[j - 1]) + \
                            d_min(cost_mat[i - 1, j], cost_mat[i, j - 1], cost_mat[i - 1, j - 1])

    return cost_mat[1:, 1:]


def get_worst_match(gt, preds, a, b, sos):
    """ Return the stroke number with the worst match
    """
    error = abs(gt[a] - preds[b]) ** 2
    # print("error", error)
    # for x in np.split(gt, sos)[1:]:
    #     print(x)

    strokes = np.split(error, sos)[1:]

    ## ACTUAL DISTANCE
    if True:
        m = np.sum(np.sum(strokes[-1], axis=1) ** .5)

    x = [np.sum(x) for x in strokes]

    #print("average mismatch cost", x)
    return np.argmax(x)


# Try reversing the stroke
# Refill ONLY the stroke of the matrix + end buffer
# Traceback from end buffer
# These are all GT indices
# First ROW/COL of cost matrix are NULL!

def adaptive_dtw(preds, gt, constraint=5, buffer=0):
    # traceback(mat, x1.shape[0], x2.shape[0])
    # create_cost_mat_2d
    # constrained_dtw2d
    _gt, _preds = np.ascontiguousarray(gt[:, :2]), np.ascontiguousarray(preds[:, :2])
    cost_mat, costr, a, b = dtw.constrained_dtw2d(_gt,_preds,
                                                  constraint=constraint)
    # print(a)
    # print(b)
    # print("full cost: ", costr)
    sos = get_sos_args(gt[:, 2], stroke_numbers=False)

    # Consider sampling from among the worst matches
    # Consider balancing worst total/average match
    worst_match_idx = get_worst_match(gt[:, :2], preds[:, :2], a, b, sos)

    # Convert the stroke number to indices in GT
    start_idx = sos[worst_match_idx]
    start_idx_buffer = max(start_idx - buffer, 0)  # double check -1

    end_idx = gt.shape[0] if worst_match_idx + 1 >= sos.size or sos[worst_match_idx + 1] > gt.shape[0] else sos[worst_match_idx + 1]
    end_idx_buffer = gt.shape[0] if end_idx + buffer >= gt.shape[0] else end_idx + buffer # too many strokes OR too many stroke points

    # Reverse the line
    _start_idx = start_idx-1 if start_idx > 0 else None
    _reversed = np.ascontiguousarray(_gt[end_idx-1:_start_idx:-1, :2])
    _new_gt = _gt.copy() # optimization: make a copy with the dataloader
    _new_gt[start_idx:end_idx] = _reversed

    # Old Cost
    if end_idx_buffer < gt.shape[0]:
        alignment_end_idx = np.argmax(a == end_idx_buffer)  # first GT point
        old_cost = cost_mat[a[alignment_end_idx], b[alignment_end_idx]]  # where we will start the traceback later

        # The indices in the preds that DTW with the GTs
        pred_start_buffer = b[np.argmax(a == start_idx_buffer)]
        pred_end_buffer = b[np.argmax(a == end_idx_buffer)]

    else: # last stroke
        old_cost = cost_mat[a[-1], b[-1]]
        pred_end_buffer = None
        pred_start_buffer = b[np.argmax(a == start_idx_buffer)]

    # Refill - end is with buffer

    # PREDS AND GTS MUST BE SAME LENGTH TO BE CONSISTENT; need to recalculate distance to end_idx buffer
    #print(np.asarray(cost_mat))
    cost_mat = dtw.refill_cost_matrix(_new_gt, _preds, cost_mat.base, start_idx, end_idx_buffer, start_idx, end_idx_buffer, constraint=constraint, metric="euclidean")
    #print(cost_mat.base.shape, cost_mat.shape)
    # cost_mat = refill_cost_matrix_dev(_new_gt, _preds, cost_mat.base, start_idx, end_idx_buffer, start_idx,
    #                                   end_idx_buffer, constraint=constraint, metric="euclidean")

    # Truncate the cost matrix to be to the designated start and end
    #print(start_idx_buffer,end_idx_buffer,pred_start_buffer,pred_end_buffer)
    cost_mat_truncated = cost_mat[start_idx_buffer:end_idx_buffer,pred_start_buffer:pred_end_buffer] # the first point in the pred]
    #print(np.asarray(cost_mat_truncated))

    #print("old cost (partial): ", old_cost)
    #print("cost (partial): ", cost_mat_truncated[-1,-1])

    if cost_mat_truncated[-1,-1]+.001 < old_cost and False:
        #print("BETTER MATCH!!!")
        # Optimize later - don't need to retrace entire matrix, just the recalc + buffer
        a,b,cost = dtw.traceback2(np.ascontiguousarray(cost_mat.base), cost_mat.shape[0], cost_mat.shape[1])
        # print("new")
        # print(a)
        # print(b)
        return b,a,_new_gt, {"swaps": None, "reverse": (slice(start_idx, end_idx), slice(end_idx-1,_start_idx, -1))}
    else:
        return b,a, None, None

    # Traceback - all the way to before the buffer
        # Finds new path
        # if better, replace path through window+buffer of original DTW
    # OVERWRITE ORIGINAL STROKE
    # OVERWRITE CURRENT STROKE
    # IF SWAPPING STROKES, NEED TO SWAP SOS TOO; SHOULD NOT SWAP SOS FOR REVERSING

## Test cases:
    # last stroke
    # first stroke
    # middle stroke

def test():
    gt = np.array(range(36)).reshape(9, 4).astype(np.float64)
    gt[:, 2] = [1, 0, 0, 1, 0, 1, 1, 0, 0]

    # Reverse first and last
    preds_first_last = [[8, 9, 1, 3],
             [4, 5, 0, 7],
             [0, 1, 0, 11],
             [12, 13, 1, 15],
             [16, 17, 0, 19],
             [20, 21, 1, 23],
             [32, 33, 1, 27],
             [28, 29, 0, 31],
             [24, 25, 0, 35]]

    # Normal
    preds = [[0, 1, 1, 11],
            [4, 5, 0, 7],
            [8, 9, 0, 3],
            [12, 13, 1, 15],
            [16, 17, 0, 19],
            [20, 21, 1, 23],
            [24, 25, 1, 35],
            [28, 29, 0, 31],
            [32, 33, 0, 27]]

    # Inverted middle
    preds_middle = [[0, 1, 1, 11],
            [4, 5, 0, 7],
            [8, 9, 0, 3],
            [16, 17, 1, 19],
            [12, 13, 0, 15],
            [20, 21, 1, 23],
            [24, 25, 1, 35],
            [28, 29, 0, 31],
            [32, 33, 0, 27]]

    # Add a stroke and reverse last
    # preds = [[0, 1, 0, 11],
    #         [4, 5, 0, 7],
    #         [8, 9, 1, 3],
    #         [12, 13, 1, 15],
    #         [16, 17, 0, 19],
    #         [20, 21, 1, 23],
    #         [21, 21, 1, 23],
    #         [32, 33, 1, 27],
    #         [28, 29, 0, 31],
    #         [24, 25, 0, 35]]


    # Revere last stroke
    preds_last = [[0, 1, 0, 11],
            [4, 5, 0, 7],
            [8, 9, 1, 3],
            [12, 13, 1, 15],
            [16, 17, 0, 19],
            [20, 21, 1, 23],
            [32, 33, 1, 27],
            [28, 29, 0, 31],
            [24, 25, 0, 35]]

    test_cases = {}
    test_cases["preds_first_last"] = {"preds":np.asarray(preds_first_last).astype(np.float64),
                           "final_cost": 0}
    test_cases["preds_last"] = {"preds":np.asarray(preds_last).astype(np.float64),
                           "final_cost": 0}
    test_cases["preds_middle"] = {"preds":np.asarray(preds_middle).astype(np.float64),
                           "final_cost": 0}
    test_cases["preds"] = {"preds":np.asarray(preds).astype(np.float64),
                           "final_cost": 0}

    for key, pred in test_cases.items():
        print(f"TESTING {key}")
        adaptive_dtw(pred["preds"], gt, constraint=5, buffer=0)

if __name__=='__main__':
    test()