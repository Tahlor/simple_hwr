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
from torch.nn.functional import softmax
from collections import defaultdict, Counter
from matplotlib import pyplot as plt
np.set_printoptions(precision=1)

COUNTER ={"swap_prev":[0,0], "swap_next":[0,0], "reverse":[0,0]}
WORST = {"strokes": Counter(), "worst": Counter(), 'percentile':[]}
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


def get_worst_match(gt, preds, a, b, stroke_numbers=True):
    """ Return the stroke number with the worst match

    Args:
        gt: Must be L * X,Y,SOS
        preds: L * X,Y
        a:
        b:
        stroke_numbers:

    Returns:

    """
    sos_stroke_numbers = np.cumsum(gt[:,2])  if not stroke_numbers else gt[:,2]
    error = abs(gt[a][:,:2] - preds[b]) ** 2
    # print("error", error)
    # for x in np.split(gt, sos)[1:]:
    #     print(x)

    sos = get_sos_args(sos_stroke_numbers[a], stroke_numbers=True)
    strokes = np.split(error, sos)[1:]

    ## ACTUAL DISTANCE
    x = [np.sum(np.sum(x, axis=1)) for x in strokes]

    #print("average mismatch cost", x)
    return np.argmax(x), sos


# Try reversing the stroke
# Refill ONLY the stroke of the matrix + end buffer
# Traceback from end buffer
# These are all GT indices
# First ROW/COL of cost matrix are NULL!


## Check pre, post, and reverse
    ## Function that recalculates given a swap
    ## Return indices to be swapped
## Use either a sampler based on mean and sampler based on absolute error - use softmax to assign probabilities
    ##

def swap_strokes(instruction_dict, gt, stroke_numbers):
    """ Doesn't support negative; MODIFIES the GT

    Args:
        instruction_dict:
        gt:
        stroke_numbers:

    Returns:

    """
    if "first_stroke_number" in instruction_dict and "second_stroke_number" in instruction_dict:
        sos_args = instruction_dict["sos"] if "sos" in instruction_dict else get_sos_args(gt[:,2], stroke_numbers=stroke_numbers)

        first_stroke_idx = sos_args[instruction_dict["first_stroke_number"]]
        second_stroke_idx = sos_args[instruction_dict["second_stroke_number"]]
        end_stroke_idx = gt.shape[0] if instruction_dict["second_stroke_number"]+1>=len(sos_args) else sos_args[instruction_dict["second_stroke_number"]+1]
    else:
        first_stroke_idx = instruction_dict["first_stroke_idx"]
        second_stroke_idx = instruction_dict["second_stroke_idx"]
        end_stroke_idx = instruction_dict["end_stroke_idx"]
    print(first_stroke_idx, second_stroke_idx, end_stroke_idx, gt.shape)
    first_backup = gt[first_stroke_idx:second_stroke_idx].copy()
    second_backup = gt[second_stroke_idx:end_stroke_idx].copy()
    gt[first_stroke_idx:first_stroke_idx+end_stroke_idx-second_stroke_idx] = second_backup
    new_middle_point = first_stroke_idx + end_stroke_idx - second_stroke_idx
    gt[first_stroke_idx:new_middle_point] = second_backup
    gt[new_middle_point:end_stroke_idx] = first_backup

    #second_stroke_idx-first_stroke_idx == end_stroke_idx-new_middle_point
    #second_stroke_idx-first_stroke_idx == end_stroke_idx-(first_stroke_idx + end_stroke_idx - second_stroke_idx)

    if stroke_numbers: # Stroke numbers are now out of order; find where the strokes change, then re-add
        sos = relativefy(gt[:,2])!=0
        gt[:,2] = np.cumsum(sos) # Regenerate stroke numbers

    return gt, first_stroke_idx, end_stroke_idx

def adaptive_dtw_original(preds, gt, constraint=5, buffer=20, testing=False, verbose=False):
    _print = print if verbose else lambda *a, **k: None
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


    worst_match_str_num = get_worst_match(gt[:, :2], preds[:, :2], a, b, sos)

    # Convert the stroke number to indices in GT
    start_idx = sos[worst_match_str_num]
    start_idx_buffer = max(start_idx - buffer, 0)  # double check -1

    # end_idx already include range
    end_idx = gt.shape[0] if worst_match_str_num + 1 >= sos.size or sos[worst_match_str_num + 1] > gt.shape[0] else sos[worst_match_str_num + 1]
    end_idx_buffer = gt.shape[0] if end_idx + buffer >= gt.shape[0] else end_idx + buffer # too many strokes OR too many stroke points

    # Reverse the line
    _new_gt = _gt.copy()  # optimization: make a copy with the dataloader
    _start_idx = start_idx-1 if start_idx > 0 else None

    if not testing:
        _reversed = np.ascontiguousarray(_gt[end_idx-1:_start_idx:-1, :2])
        _new_gt[start_idx:end_idx] = _reversed

    # Old Cost
    if end_idx_buffer < gt.shape[0]:
        alignment_end_idx = np.argmax(a == end_idx_buffer)  # get end arg of alignment sequence (first time it appears)
        alignment_start_idx = np.argmax(a == start_idx_buffer)
        # The indices in the preds that DTW with the GTs
        pred_start_buffer = b[alignment_start_idx] # get the corresponding pred indices
        pred_end_buffer = b[alignment_end_idx] # at this point, we could have a rectangle cost matrix!!
        #assert a[alignment_end_idx] == end_idx_buffer
        old_cost = cost_mat[end_idx_buffer, pred_end_buffer]  # cost mat is 1-indexed +1, but these are ranges -1

    else: # last stroke
        old_cost = cost_mat[-1,-1]
        pred_end_buffer = preds.shape[0]
        pred_start_buffer = b[np.argmax(a == start_idx_buffer)]

    # Refill - end is with buffer

    # PREDS AND GTS MUST BE SAME LENGTH TO BE CONSISTENT; need to recalculate distance to end_idx buffer
    #_print(np.asarray(cost_mat))
    cost_mat = dtw.refill_cost_matrix(_new_gt, _preds, cost_mat.base, start_idx, end_idx_buffer, pred_start_buffer, pred_end_buffer, constraint=constraint, metric="euclidean")
    _print(cost_mat.base.shape, cost_mat.shape)
    #_print(np.asarray(cost_mat))

    # Truncate the cost matrix to be to the designated start and end
    _print(start_idx_buffer,end_idx_buffer,pred_start_buffer,pred_end_buffer)
    cost_mat_truncated = cost_mat[start_idx_buffer+1:end_idx_buffer+1,pred_start_buffer+1:pred_end_buffer+1] # +1 since 1-indexed
    _print(np.asarray(cost_mat_truncated))

    _print("old cost (partial): ", old_cost)
    _print("cost (partial): ", cost_mat_truncated[-1,-1])
    _print(cost_mat_truncated[-1,-1], old_cost)
    if testing:
        assert cost_mat_truncated[-1,-1] == old_cost
    if cost_mat_truncated[-1,-1]+.00001 < old_cost and not testing:
        _print("BETTER MATCH!!!")
        # Optimize later - don't need to retrace entire matrix, just the recalc + buffer
        a,b,cost = dtw.traceback2(np.ascontiguousarray(cost_mat))
        _print("new")
        _print(a)
        _print(b)
        return b,a,_new_gt, {"swaps": None, "reverse": (slice(start_idx, end_idx), slice(end_idx-1,_start_idx, -1))}
    else:
        return b,a, None, None

    # Traceback - all the way to before the buffer
        # Finds new path
        # if better, replace path through window+buffer of original DTW
    # OVERWRITE ORIGINAL STROKE
    # OVERWRITE CURRENT STROKE
    # IF SWAPPING STROKES, NEED TO SWAP SOS TOO; SHOULD NOT SWAP SOS FOR REVERSING

def check(new_gt, _preds, cost_mat, a, b, start_idx, end_idx, constraint, buffer, verbose, testing):
    start_idx_buffer = max(start_idx - buffer, 0)  # double check -1
    end_idx_buffer = new_gt.shape[0] if end_idx + buffer >= new_gt.shape[0] else end_idx + buffer # too many strokes OR too many stroke points
    # Old Cost
    if end_idx_buffer < new_gt.shape[0]:
        alignment_end_idx = np.argmax(a == end_idx_buffer)  # get end arg of alignment sequence (first time it appears)
        alignment_start_idx = np.argmax(a == start_idx_buffer)
        # The indices in the preds that DTW with the GTs
        pred_start_buffer = b[alignment_start_idx] # get the corresponding pred indices
        pred_end_buffer = max(b[alignment_end_idx], b[alignment_start_idx]+1) # at this point, we could have a rectangle cost matrix!!; make sure at least 1 box long!
        #assert a[alignment_end_idx] == end_idx_buffer
        old_cost = cost_mat[end_idx_buffer, pred_end_buffer]  # cost mat is 1-indexed +1, but these are ranges -1

    else: # last stroke
        old_cost = cost_mat[-1,-1]
        pred_end_buffer = _preds.shape[0]
        pred_start_buffer = b[np.argmax(a == start_idx_buffer)]

    _print = print if verbose else lambda *a, **k: None
    # PREDS AND GTS MUST BE SAME LENGTH TO BE CONSISTENT; need to recalculate distance to end_idx buffer
    cost_mat = dtw.refill_cost_matrix(new_gt, _preds, cost_mat.base, start_idx, end_idx_buffer, pred_start_buffer, pred_end_buffer, constraint=constraint, metric="euclidean")
    _print(cost_mat.base.shape, cost_mat.shape)
    #_print(np.asarray(cost_mat))

    # Truncate the cost matrix to be to the designated start and end
    _print(start_idx_buffer,end_idx_buffer,pred_start_buffer,pred_end_buffer)
    cost_mat_truncated = cost_mat[start_idx_buffer+1:end_idx_buffer+1,pred_start_buffer+1:pred_end_buffer+1] # +1 since 1-indexed
    _print(np.asarray(cost_mat_truncated))

    _print("old cost (partial): ", old_cost)
    _print("cost (partial): ", cost_mat_truncated[-1,-1])
    _print(cost_mat_truncated[-1,-1], old_cost)
    if testing:
        assert cost_mat_truncated[-1,-1] == old_cost
    #if cost_mat_truncated[-1, -1] + .00001 < old_cost and not testing:
    return old_cost - cost_mat_truncated[-1, -1], cost_mat


def check_reverse(_gt, _preds, cost_mat, a, b, worst_match_stroke_num, sos_args, constraint, buffer, testing=False, verbose=False):
    _print = print if verbose else lambda *a, **k: None

    # Convert the stroke number to indices in GT
    start_idx = sos_args[worst_match_stroke_num]

    # end_idx already include range
    end_idx = _gt.shape[0] if worst_match_stroke_num + 1 >= sos_args.size or sos_args[worst_match_stroke_num + 1] > _gt.shape[0] else sos_args[worst_match_stroke_num + 1]


    # Reverse the line
    _new_gt = _gt.copy()  # optimization: make a copy with the dataloader
    _start_idx = start_idx-1 if start_idx > 0 else None

    if not testing:
        _reversed = np.ascontiguousarray(_gt[end_idx-1:_start_idx:-1, :2])
        _new_gt[start_idx:end_idx] = _reversed


    cost_savings, cost_mat = check(new_gt=_new_gt, _preds=_preds, cost_mat=cost_mat, a=a, b=b, start_idx=start_idx, end_idx=end_idx,
          constraint=constraint, buffer=buffer, verbose=verbose, testing=testing)

    return {"gt":_new_gt, "cost_savings":cost_savings, "cost_mat":cost_mat, "instruction":{"reverse": (slice(start_idx, end_idx), slice(end_idx - 1, _start_idx, -1))}}


def check_swap(_gt, _preds, cost_mat, a, b, worst_match_stroke_num, sos_args, constraint, buffer, testing=False, verbose=False):
    """ Always swap with previous; assumes you did not give it the first stroke

    Returns:

    """
    assert worst_match_stroke_num > 0
    _print = print if verbose else lambda *a, **k: None

    # Convert the stroke number to indices in GT
    start_idx = sos_args[worst_match_stroke_num-1]
    second_stroke_idx = sos_args[worst_match_stroke_num]

    # end_idx already include range
    end_idx = _gt.shape[0] if worst_match_stroke_num + 1 >= sos_args.size or sos_args[worst_match_stroke_num + 1] > _gt.shape[0] else sos_args[worst_match_stroke_num + 1]

    #instruction = {"first_stroke_number": worst_match_stroke_num - 1, "second_stroke_number": worst_match_stroke_num, "sos": sos_args}
    instruction = {"first_stroke_idx": start_idx, "end_stroke_idx": end_idx,
                   "second_stroke_idx": second_stroke_idx}

    # Swap the strokes
    _new_gt = _gt.copy()
    if not testing:
        _new_gt, _start_idx, _end_idx = swap_strokes(instruction_dict=instruction, gt=_new_gt, stroke_numbers=False) # isn't even getting 3rd dimension

    cost_savings, cost_mat = check(new_gt=_new_gt, _preds=_preds, cost_mat=cost_mat, a=a, b=b, start_idx=start_idx, end_idx=end_idx,
          constraint=constraint, buffer=buffer, verbose=verbose, testing=testing)

    return {"gt":_new_gt, "cost_savings":cost_savings, "cost_mat":cost_mat, "instruction": instruction}

def sample_worst_matches():
    pass


def adaptive_dtw(preds, gt, constraint=5, buffer=20, stroke_numbers=True, testing=False, verbose=False):
    global COUNTER
    _print = print if verbose else lambda *a, **k: None
    _gt, _preds = np.ascontiguousarray(gt[:, :2]), np.ascontiguousarray(preds[:, :2])
    cost_mat, costr, a, b = dtw.constrained_dtw2d(_gt,_preds,
                                                  constraint=constraint)

    sos_args = get_sos_args(gt[:, 2], stroke_numbers=stroke_numbers)

    worst_match_stroke_num, sos2 = get_worst_match(gt, _preds, a, b, stroke_numbers=stroke_numbers)
    assert len(sos_args) == len(sos2)
    # WORST["worst"].update({worst_match_stroke_num:1})
    # WORST["strokes"].update({len(sos_args):1})
    # WORST["percentile"].append((worst_match_stroke_num+1)/len(sos_args))
    results = {}
    #cost_mat2 = np.asarray(cost_mat).copy()

    # SWAP
    if worst_match_stroke_num+1 < len(sos_args):
        # Swap with next element
        results["swap_next"] = check_swap(_gt, _preds, cost_mat, a, b, worst_match_stroke_num+1, sos_args, constraint, buffer, testing=testing, verbose=verbose)
        #COUNTER["swap_next"][1] += 1
    # Swap with previous
    if worst_match_stroke_num > 0 and len(sos_args) > 1:
        results["swap_prev"] = check_swap(_gt, _preds, cost_mat, a, b, worst_match_stroke_num, sos_args, constraint, buffer, testing=testing, verbose=verbose)
        #COUNTER["swap_prev"][1] += 1


    # Reverse
    results["reverse"] = check_reverse(_gt, _preds, cost_mat, a, b, worst_match_stroke_num, sos_args, constraint, buffer, testing=testing, verbose=verbose)

    #np.testing.assert_allclose(cost_mat2, cost_mat)
    key_max = max(results.keys(), key=(lambda k: results[k]["cost_savings"]))

    _print(results)
    if results[key_max]["cost_savings"] > 0 and not testing:
        #COUNTER[key_max][0] +=1
        chosen_one = results[key_max]
        _print(f"BETTER MATCH!!! using {key_max}")
        # Optimize later - don't need to retrace entire matrix, just the recalc + buffer
        a,b,cost = dtw.traceback2(np.ascontiguousarray(chosen_one["cost_mat"]))
        _print("new")
        _print(a)
        _print(b)
        return b,a, chosen_one["gt"], chosen_one["instruction"]
    else:
        return b,a, None, None


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
    preds_last = [[0, 1, 1, 11],
            [4, 5, 0, 7],
            [8, 9, 0, 3],
            [12, 13, 1, 15],
            [16, 17, 0, 19],
            [20, 21, 1, 23],
            [32, 33, 1, 27],
            [28, 29, 0, 31],
            [24, 25, 0, 35]]

    # Solution: swap 1 with 2 - stroke 2 has higher loss
    preds_swap = [[0, 1, 1, 11],
            [4, 5, 0, 7],
            [8, 9, 0, 3],
            [20, 21, 1, 23],
            [12, 13, 1, 15],
            [16, 17, 0, 19],
            [24, 25, 1, 35],
            [28, 29, 0, 31],
            [32, 33, 0, 27]]

    preds_combo = [[0, 1, 1, 11],
            [4, 5, 0, 7],
            [8, 9, 0, 3],
            [20, 21, 1, 23],
            [12, 13, 1, 15],
            [16, 17, 0, 19],
            [32, 33, 1, 27],
            [28, 29, 0, 31],
            [24, 25, 0, 35]]


    random_preds = [[5, 1, 0, 11],
            [6, 5, 0, 7],
            [7, 9, 1, 3],
            [24, 13, 1, 15],
            [3, 17, 0, 19],
            [19, 21, 1, 23],
            [2, 33, 1, 27],
            [4, 29, 0, 31],
            [18, 25, 0, 35]]


    test_cases = {}
    test_cases["preds_first_last"] = {"preds":np.asarray(preds_first_last).astype(np.float64),
                           "final_cost": 0}
    test_cases["preds_last"] = {"preds":np.asarray(preds_last).astype(np.float64),
                           "final_cost": 0}
    test_cases["preds_middle"] = {"preds":np.asarray(preds_middle).astype(np.float64),
                           "final_cost": 0}
    test_cases["preds"] = {"preds":np.asarray(preds).astype(np.float64),
                           "final_cost": 0}

    test_cases["preds_swap"] = {"preds": np.asarray(preds_swap).astype(np.float64),
                           "final_cost": 0, "solution":"swap"}

    def gen_random(l=40):
        random_preds = np.random.randint(0, 30, 3 * l).reshape(l, 3).astype(np.float64)
        random_preds[:, 2] = np.round(np.random.random(l) - .2)
        random_gt = np.random.randint(0, 30, 3 * l).reshape(l, 3).astype(np.float64)
        random_gt[:, 2] = np.round(np.random.random(l) - .2)
        random_gt[0,2]=1
        random_preds[0, 2]=1
        return random_preds, random_gt

    for key, pred in test_cases.items():
        print(f"TESTING {key}")
        adaptive_dtw(pred["preds"], gt, constraint=5, buffer=0, stroke_numbers=False, verbose=True)
    #stop
    print("TESTING MODE -- doesn't find better outcomes, just makes sure the cost matrix is refilled the same way")
    for i in range(1000):
        random_preds, random_gt = gen_random()
        adaptive_dtw(random_preds, random_gt, constraint=5, buffer=0, stroke_numbers=False, testing=True, verbose=False)
    print(COUNTER)

    print("TESTING MODE -- doesn't find better outcomes, just makes sure the cost matrix is refilled the same way")
    for i in range(1000):
        random_preds, random_gt = gen_random()
        adaptive_dtw(random_preds, random_gt, constraint=5, buffer=5, stroke_numbers=False, testing=True, verbose=False)
    print(COUNTER)

    print("FIND SOME BETTER OUTCOMES!")
    for i in range(1000):
        random_preds, random_gt = gen_random()
        adaptive_dtw(random_preds, random_gt, constraint=5, buffer=0, stroke_numbers=False, testing=False, verbose=False)

    print("FIND SOME BETTER OUTCOMES!")
    for i in range(2400):
        random_preds, random_gt = gen_random()
        adaptive_dtw(random_preds, random_gt, constraint=5, buffer=5, stroke_numbers=False, testing=False, verbose=False)


def test_swap_strokes():
    preds_last = [[0, 1, 1, 11],
            [4, 5, 0, 7],
            [8, 9, 0, 3],
            [12, 13, 1, 15],
            [16, 17, 0, 19],
            [20, 21, 1, 23],
            [32, 33, 1, 27],
            [28, 29, 0, 31],
            [24, 25, 0, 35]]

    preds_last = np.asarray(preds_last)
    sos = get_sos_args(preds_last[:,2], stroke_numbers=False)
    instruction_dict = {"first_stroke_number":0, "second_stroke_number": 1, "sos":sos}
    swapped, s, e = swap_strokes(instruction_dict=instruction_dict, gt=preds_last.copy(), stroke_numbers=False)
    print(swapped)

    instruction_dict = {"first_stroke_number":1, "second_stroke_number": 2, "sos":sos}
    swapped, s, e = swap_strokes(instruction_dict=instruction_dict, gt=preds_last.copy(), stroke_numbers=False)
    print(swapped)

    instruction_dict = {"first_stroke_number":2, "second_stroke_number": 3, "sos":sos}
    swapped, s, e = swap_strokes(instruction_dict=instruction_dict, gt=preds_last.copy(), stroke_numbers=False)
    print(swapped)

    # instruction_dict = {"first_stroke_number":-2, "second_stroke_number": -1, "sos":sos}
    # swapped = swap_strokes(instruction_dict=instruction_dict, gt=preds_last.copy(), stroke_numbers=False)
    # print(swapped)

if __name__=='__main__':
    test()
    print(COUNTER)
    print(WORST)
    plt.bar(WORST["strokes"].keys(), WORST["strokes"].values(), color='g')
    plt.show()
    plt.bar(WORST["worst"].keys(), WORST["worst"].values(), color='g')
    plt.show()
    plt.hist(WORST["percentile"])
    plt.show()
    #test_swap_strokes()