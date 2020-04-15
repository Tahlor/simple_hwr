import warnings
import os
import torch
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from pathlib import Path
import os
import torch
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path
from scipy import interpolate
#from sklearn.preprocessing import normalize
from pathlib import Path
import json
from easydict import EasyDict as edict
import logging
from hwr_utils.stroke_plotting import *
from torch.nn import functional
from torch import Tensor, tensor
from scipy.spatial import KDTree

## Other loss functions and variations
# Distance from actual point
# Distance to nearest point on image
# Minimal penalty for first/last stroke points
# Make it probabilistic? I.e. there's some probability it starts at the top or bottom
# Compare rendered images! How to make it differentiable?
# Handle empty time differently - have computer guess "in between" stroke value? could work
# Stroke level penalties - perfect inverse/backward patterns have minimal penalty
# Need an end of stroke and end of sequence token - small penalty if end of stroke is close

## Other ideas
# Feed in current "step" to RNN, scaled to same scale as width

# Add more instances -- otherwise make it so the first instance is at the start of the letter

logger = logging.getLogger("root."+__name__)

EPSILON = 1e-8

def distance_metric(x,y):
    """ Returns same sized array with 0 the first distance
        Euclidean distance metric between x and x-1; first item in stroke has distance of epsilon
    Args:
        x: array-like
        y: array-like

    Returns:

    """

    output = np.zeros(x.size)
    output[1:] = ((x[:-1] - x[1:]) ** 2 + (y[:-1] - y[1:]) ** 2) ** (1 / 2)
    output[0] = 0
    return output

def read_stroke_xml(path, start_stroke=None, end_stroke=None):
    """
    Args:
        path: XML path to stroke file OR XML parsing of it

    Returns:
        list of lists of dicts: each dict contains a stroke, keys: x,y, time
    """
    if isinstance(path, Path):
        root = ET.parse(path).getroot()
        all_strokes = root[1]
    else:
        all_strokes = path

    stroke_list = []
    strokes = all_strokes[start_stroke:end_stroke]
    start_times = []

    for stroke in strokes:
        x_coords = []
        y_coords = []
        time_list = []

        for i, point in enumerate(stroke):

            x, y, time = point.attrib["x"], point.attrib["y"], float(point.attrib["time"])
            if not stroke_list and i == 0:
                first_time = time

            time -= first_time
            x_coords.append(int(x))
            y_coords.append(-int(y))
            time_list.append(round(time, 3))
            if i == 0:
                start_times.append(time)

        stroke_list.append({"x": x_coords, "y": y_coords, "time": time_list})

    return stroke_list, start_times

def create_functions_from_strokes(stroke_dict, parameter="t"):
    if not isinstance(stroke_dict, edict):
        stroke_dict = edict(stroke_dict)
    x_func = interpolate.interp1d(stroke_dict[parameter], stroke_dict.x)
    y_func = interpolate.interp1d(stroke_dict[parameter], stroke_dict.y)
    return x_func, y_func

def get_eos_from_sos(sos):
    eos = np.ones(sos.shape[-1])
    eos[:-1] = sos[1:]
    return eos

def prep_stroke_dict(strokes, time_interval=None, scale_time_distance=True):
    """ Takes in a "raw" stroke list for one image
        Each element of stroke_list is a dict with keys x,y,time
        OR
        Each element is a list of array-like strokes of the form [L x (x,y),...]

        time_interval (float): duration of upstroke events; None=original duration
        Returns:
             x(t), y(t), stroke_up_down(t), start_times
                 Note that the last start time is the end of the last stroke
    """

    x_list = []
    y_list = []
    t_list = []
    start_strokes = []
    start_times = []
    epsilon = EPSILON

    # Epsilon is the amount of time before or after a stroke for the interpolation
    # Time between strokes must be greater than epsilon, or interpolated points between strokes will result
    if time_interval is None or time_interval < epsilon:
        time_interval = epsilon * 3

    distance = 0

    # Loop through each stroke
    for i, stroke_dict in enumerate(strokes):
        if isinstance(stroke_dict, dict):
            x_coords = stroke_dict["x"]
            y_coords = stroke_dict["y"]
            key = "time" if "time" in stroke_dict else "t"
            time = stroke_dict[key]
        else: # numpy input - TIME x (X,Y)
            _arr = np.asarray(stroke_dict)
            if not len(_arr):
                continue
            x_coords = _arr[:,0].tolist()
            y_coords = _arr[:,1].tolist()
            if _arr.shape[-1] > 2:
                time = _arr[:,2].tolist()
            else:
                time = np.zeros(_arr.shape[0]).tolist()

        xs = np.asarray(x_coords)
        ys = np.asarray(y_coords)
        distance += np.sum(distance_metric(xs, ys)) # total stroke distance

        x_list += x_coords
        y_list += y_coords
        start_strokes += [1] + [0] * (len(x_coords)-1)

        # Set duration for "upstroke" events
        if not time_interval is None and i > 0:
            next_start_time = time[0]
            last_end_time = t_list[-1]
            t_offset = time_interval + last_end_time - next_start_time
            t_list_add = [t + t_offset for t in time]
        else:
            t_list_add = time

        t_list += t_list_add
        start_times += [t_list_add[0]]

    # Add the last time to the start times
    start_times += [t_list_add[-1]]
    start_strokes += [0]

    ## Normalize
    y_list = np.asarray(y_list)
    x_list = np.asarray(x_list)

    assert t_list[0] == start_times[0] # first time should be first start time
    t_list = np.asarray(t_list) - t_list[0]
    start_strokes = np.asarray(start_strokes)
    start_times = np.asarray(start_times) - start_times[0] # zero start

    y_list, scale_param = normalize(y_list)
    x_list, scale_param = normalize(x_list, scale_param)
    if scale_param <= 0:
        return None

    distance = distance / scale_param # normalize the distance

    if scale_time_distance:
        if t_list[-1]==0:
            warnings.warn(f"Last time is 0 {t_list}")
            return None
        else:
            time_factor = distance / (t_list[-1])
            t_list = t_list * time_factor
            start_times = start_times * time_factor

    # Have interpolation not move after last point
    x_list = np.append(x_list, x_list[-1])
    y_list = np.append(y_list, y_list[-1])
    t_list = np.append(t_list, t_list[-1] + 10)

    if np.max(y_list) == 0: # perfectly horizontal line
        warnings.warn(f"Max y is 0 {y_list}")
        x_to_y = 1
        return None
    else:
        x_to_y = np.max(x_list) / np.max(y_list)

    # Start strokes (binary list) will now be 1 short!
    d_list = reparameterize_as_func_of_distance(x_list, y_list, start_strokes)
    start_distances = np.r_[d_list[start_strokes==1], d_list[-2]]

    # print(start_distances[-3:])
    # print(start_times[-3:])
    # stop
    output = edict({"x":x_list, "y":y_list, "t":t_list, "d":d_list, "start_times":start_times, "start_distances":start_distances, "x_to_y":x_to_y,
                    "start_strokes":start_strokes, "raw":strokes, "tmin":start_times[0], "tmax":start_times[-2],
                    "trange":start_times[-2]-start_times[0], "drange":d_list[-2]-d_list[0]})

    #print(d_list, x_list, y_list, t_list)
    return output

def resample(gt, distance=True):
    # Calculate distances
    EPSILON = 1e-6
    #not_start_stroke_online = 1 - np.tile(gt[:-1, 2:], 2) + eos*EPSILON
    sos = [np.round(gt[:,2])==1]

    # Add origin for reference
    gt_with_origin = np.vstack([[0,0,0],gt])
    distances = np.sum((gt_with_origin[1:,:2]-gt_with_origin[0:-1, :2])**2, axis=1)
    distances[sos] = EPSILON # distances for start strokes are epsilon
    cum_distance = np.cumsum(distances)

    x_func = interpolate.interp1d(cum_distance, gt[:, 0])
    y_func = interpolate.interp1d(cum_distance, gt[:, 1])

    # Create funcs
    start_of_stroke_distance_values = cum_distance[sos]

    x, y, is_start_stroke = sample(x_func, y_func,
                                   start_of_stroke_distance_values,
                                   number_of_samples=gt.shape[0],
                                   noise=False,
                                   last_time=cum_distance[-1])
    gt = np.vstack([x,y,is_start_stroke]).transpose()

    #draw_from_gt(gt)

    return gt

## DOES THIS WORK? SHOULD BE THE SAME AS BATCH_TORCH, NEED TO TEST
def relativefy_batch(batch, reverse=False):
    """ A tensor: Batch, Width, Vocab

    Args:
        batch:
        reverse:

    Returns:

    """
    import warnings
    warnings.warn("relativefy_batch: Untested")
    for i,b in enumerate(batch):
        #print(batch.size(), batch)
        #print(batch[i,:,0])
        #print(i, b)
        relativefy(b[:, 0], reverse=reverse)
        batch[i] = relativefy(b[:,0], reverse=reverse)
    return batch

class PredConvolver:
    def __init__(self, convolve_type, kernel_length=21):
        convolve_functions = {"cumsum":relativefy_batch_torch, "conv_weight":conv_weight, "conv_window": conv_window}
        self.convolve_func = convolve_functions[convolve_type]
        if convolve_type=="conv_weight":
            kernel = Tensor(range(0, kernel_length)).unsqueeze(1).repeat(1, 1, 1, 1) / (kernel_length - 1)
            self.kwargs = {"kernel": kernel, "inverse_kernel":1-kernel, "kernel_length": kernel_length}
        elif convolve_type=="conv_window":
            kernel_window = torch.ones(kernel_length).unsqueeze(1).repeat(1, 1, 1, 1)
            self.kwargs = {"kernel": kernel_window, "kernel_length": kernel_length}
        elif convolve_type=="cumsum": # NOT BEING USED PRESENTLY
            self.kwargs = {"reverse":True}
        logger.info(("Convolve Options", self.__dict__))

    def convolve(self, pred_rel, indices, gt):
        return self.convolve_func(pred_rel=pred_rel, gt_abs=gt, indices=indices, **self.kwargs)


def relativefy_batch_torch(batch, reverse=False, indices=slice(0,None), **kwargs):
    """ A tensor: Batch, Width, Vocab
        Modifies branch
    """
    if reverse:
        # Only update the x-coords
        batch[:, :, indices] = torch.cumsum(batch[:, :, indices], dim=1)
        return batch
    else:
        # The first item in batch is not changed
        # Subtract the current item from next item to get delta
        batch[:,1:,indices] = batch[:, 1:, indices]-batch[:, :-1, indices] # all items in batch, entire sequence, only X coords
        return batch

# m = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

KERNEL_LENGTH = 9
KERNEL = Tensor(range(0,KERNEL_LENGTH)).unsqueeze(1).repeat(1, 1, 1, 1)/(KERNEL_LENGTH-1)
INVERSE_KERNEL = 1-KERNEL

def conv_weight(gt_abs, pred_rel, gt_rel=None, indices=slice(0,None), kernel=KERNEL, inverse_kernel=INVERSE_KERNEL, kernel_length=KERNEL_LENGTH, **kwargs):
    """ BATCH, WIDTH (GT LENGTH), HEIGHT (VOCAB SIZE)
        INDICES MUST BE SLICE/LIST
    """
    if kernel is None:
        Tensor(range(0,kernel_length)).unsqueeze(1).repeat(1, 1, 1, 1)/(kernel_length-1)

    if inverse_kernel is None:
        inverse_kernel = 1 - kernel

    width = gt_abs.shape[1]
    pred_rel = pred_rel[:,:width] # truncate any extra preds due to white space

    cumsum = torch.zeros(*gt_abs.shape)
    cumsum[:, kernel_length:, indices] = gt_abs[:, :width - kernel_length, indices]  # BATCH, WIDTH, VOCAB

    if gt_rel is None:
        gt_rel = relativefy_batch_torch(gt_abs.detach().clone(), indices=indices)

    # Add channel dimension
    gt_rel_exp = gt_rel.unsqueeze(1)[:,:,:,indices]  # BATCH, CHANNEL, WIDTH, VOCAB
    pred_rel_exp = pred_rel.unsqueeze(1)[:,:,:,indices]

    # Functionary way
    gt_rel[:,:,indices] = functional.conv2d(gt_rel_exp, inverse_kernel, padding=[kernel_length - 1, 0]).squeeze(1)[:,:width]
    pred_rel[:,:,indices] = functional.conv2d(pred_rel_exp, kernel, padding=[kernel_length - 1, 0]).squeeze(1)[:,:width] + gt_rel[:,:,indices] + cumsum[:,:,indices]
    return pred_rel

KERNEL_WINDOW = torch.ones(KERNEL_LENGTH).unsqueeze(1).repeat(1, 1, 1, 1)
def conv_window(gt_abs, pred_rel, indices=slice(0,None), kernel_window=KERNEL_WINDOW, kernel_length=KERNEL_LENGTH, **kwargs):
    """ BATCH, WIDTH (GT LENGTH), HEIGHT (VOCAB SIZE)
        INDICES MUST BE SLICE/LIST

        THIS EXPECTS RELATIVE AND RETURNS abs!!!
    """
    width = gt_abs.shape[1]
    pred_rel = pred_rel[:,:width] # truncate any extra preds due to white space

    cumsum = torch.zeros(*gt_abs.shape)
    cumsum[:, kernel_length:, indices] = gt_abs[:, :width - kernel_length, indices]  # BATCH, WIDTH, VOCAB

    # Add channel dimension
    pred_rel_exp = pred_rel.unsqueeze(1)[:,:,:,indices]

    # Functionary way
    pred_rel[:,:,indices] = functional.conv2d(pred_rel_exp, kernel_window, padding=[kernel_length - 1, 0]).squeeze(1)[:,:width] + cumsum[:,:,indices]
    return pred_rel

def test_conv_weight():
    gt_length = 20
    rel_x = np.array(range(0,gt_length))
    rel_y = np.random.randint(0, 10, 20)
    start = np.random.randint(0, 2, gt_length)

    gt = np.c_[rel_x, rel_y, start][np.newaxis]  # BATCH, WIDTH, HEIGHT/VOCAB
    pred = gt.copy()
    pred[:, 7, 0] = 12

    pred = Tensor(pred)  # relative
    gt = Tensor(gt)
    gt_rel = Tensor(gt)
    gt = torch.cumsum(gt, dim=1)  # abs

    for conv_func in conv_window, conv_weight:
        print(conv_func)
        # Anything with itself should be equivalent to a cumulative sum
        x = conv_func(gt,gt_rel.clone())
        np.testing.assert_almost_equal(x.numpy(), gt, decimal=5)

        # Should get back on track
        x = conv_func(gt,pred.clone())
        np.testing.assert_almost_equal(x[:,-4:].numpy(), gt[:,-4:], decimal=5)
        np.testing.assert_almost_equal(x[:,0:6].numpy(), gt[:,0:6], decimal=5)

        # With batching
        gt2 = Tensor(np.r_[gt,gt])
        gt2_rel = Tensor(np.r_[gt_rel.clone(),gt_rel.clone()]) # pred is relative
        x = conv_func(gt2,gt2_rel)
        np.testing.assert_almost_equal(x.numpy(), gt2, decimal=5)

        # With batching + index
        gt2 = Tensor(np.r_[gt,gt])
        gt2_rel = Tensor(np.r_[gt_rel.clone(),gt_rel.clone()]) # pred is relative
        x = conv_func(gt2,gt2_rel, indices=[1,])
        np.testing.assert_almost_equal(x.numpy()[:,:,1], gt2[:,:,1], decimal=5)
        np.testing.assert_almost_equal(x.numpy()[:,:,0], gt2_rel[:,:,0], decimal=5)


def relativefy(x, reverse=False):
    """
    Args:
        x:
        reverse:

    Returns:

    """
    if isinstance(x, np.ndarray):
        return relativefy_numpy(x, reverse)
    elif isinstance(x, torch.Tensor):
        return relativefy_torch(x, reverse)
    else:
        raise Exception(f"Unexpected type {type(x)}")

def convert_stroke_numbers_to_start_strokes(x):
    """ Count start strokes from "stroke_numbers" as the moment the stroke number exceeds .5 (e.g. 2.51 = 3rd stroke point)

    Args:
        x:

    Returns:

    """
    start_strokes = np.zeros(x.shape)
    # Where the stroke number crosses the threshold
    next_number_indices = np.argwhere(np.round(x[1:]) != np.round(x[:-1])) + 1
    start_strokes[next_number_indices] = 1
    return start_strokes

def relativefy_numpy(x, reverse=False):
    """ Make the x-coordinate relative to the previous one
        First coordinate is relative to 0
    Args:
        x (array-like): Just an array of x's coords!

    Returns:

    """
    if reverse:
        return np.cumsum(x,axis=0)
    else:
        return np.insert(x[1:]-x[:-1], 0, x[0])

def relativefy_torch(x, reverse=False, default_value=0):
    """ Make the x-coordinate relative to the previous one
        First coordinate is relative to 0
    Args:
        x:

    Returns:

    """
    if reverse:
        return torch.cumsum(x,dim=0)
    else:
        if default_value==0:
            r = torch.zeros(x.shape)
        elif default_value==1:
            r = torch.ones(x.shape)
        else:
            r = torch.zeros(x.shape)+default_value

        r[1:] = x[1:]-x[:-1]
        return r


def get_all_substrokes(stroke_dict, desired_num_of_strokes=3):
    """

    Args:
        stroke_dict: ['x', 'y', 't', 'start_times', 'x_to_y', 'start_strokes', 'raw', 'tmin', 'tmax', 'trange']
        desired_num_of_strokes:

    Returns:

    """
    if desired_num_of_strokes is None:
        yield stroke_dict
        return

    start_args = np.where(stroke_dict.start_strokes==1)[0] # returns an "array" of the list, just take first index
    start_args = np.append(start_args, None) # last start arg should be the end of the sequence

    # If fewer strokes, just return the whole thing
    if start_args.shape[0] <= desired_num_of_strokes:
        return stroke_dict

    for stroke_number in range(start_args.shape[0] - desired_num_of_strokes): # remember, last start_stroke is really the end stroke
        start_idx = start_args[stroke_number]
        end_idx = start_args[stroke_number + desired_num_of_strokes]

        t = stroke_dict.t[start_idx:end_idx].copy()
        x = stroke_dict.x[start_idx:end_idx].copy()
        y = stroke_dict.y[start_idx:end_idx].copy()
        raw = stroke_dict.raw[stroke_number:stroke_number + desired_num_of_strokes]
        start_strokes = stroke_dict.start_strokes[start_idx:end_idx]
        start_times = stroke_dict.start_times[stroke_number:stroke_number + desired_num_of_strokes + 1].copy()

        y, scale_param = normalize(y)
        x, scale_param = normalize(x, scale_param)
        if scale_param <= 0: # don't deal with this!
            continue

        x_to_y = np.max(x) / np.max(y)

        start_time = t[0]
        t -= start_time
        start_times -= start_time
        output = edict({"x": x,
                     "y": y,
                     "t": t,
                     "start_times": start_times,
                     "start_strokes": start_strokes,
                     "x_to_y":x_to_y,
                     "raw":raw})
        assert start_times[0]==t[0]
        yield output


def normalize(x_list, scale_param=None):
    x_list -= np.min(x_list)

    if scale_param is None:
        scale_param = np.max(x_list)

    if scale_param > 0:
        x_list = x_list / scale_param
    else:
        warnings.warn(f"Scale parameter is {scale_param}, {x_list}")
    return x_list, scale_param

def sample_OLD(function_x, function_y, starts, number_of_samples=64, noise=None, plot=False):
    """ Given some ipolate functions, return

    Args:
        function_x:
        function_y:
        starts:
        number_of_samples:
        noise:
        plot:

    Returns:
        list of x_points, list of y_points, binary list of whether the corresponding point is a start stroke
    """
    last_time = starts[-1] # get the last start point - this should actually be the first start point of the next stroke!!
    interval = last_time / number_of_samples
    std_dev = interval / 3
    time = np.linspace(0, last_time, number_of_samples)

    if noise:
        momentum = .8
        if noise == "random": # random noise over time/distance on each point
            noises = np.random.normal(0, std_dev, time.shape)
        elif noise == "lagged": # cumulative noise
            noises = []
            offset = 0
            noise = 0
            std_dev_decay = 1 - (.9) ** min(number_of_samples, 100)
            # Decay std_dev
            for i in range(0, number_of_samples):
                remaining = number_of_samples - i
                noise = np.random.normal(-offset / remaining + noise * momentum, std_dev)  # add momentum term
                offset += noise
                noises.append(noise + offset)
                if remaining < 100:
                    std_dev *= std_dev_decay
            noises = np.asarray(noises)

        if plot:
            plt.plot(time, noises)
            plt.show()
        time += noises
        time.sort(kind='mergesort')  # not actually a mergesort, but fast on nearly sorted data
        time = np.maximum(time, 0)
        time = np.minimum(time, last_time)

    ## Get start strokes
    start_stroke_idx = [0]  # first one is a start
    for start in starts[1:]:
        already_checked = start_stroke_idx[-1]
        start_stroke_idx.append(np.argmax(time[already_checked:] >= start) + already_checked)

    # print(function_x, function_y, time, start_stroke_idx)
    # time[start_stroke_idx] - start times
    is_start_stroke = np.zeros(time.shape)
    is_start_stroke[start_stroke_idx[:-1]] = 1 # the last "start stroke time" is not the last element, not a start stroke

    #print(time)
    return function_x(time), function_y(time), is_start_stroke

def sample(function_x, function_y, start_times, number_of_samples=64, noise=None, plot=False, last_time=None):
    """ Given some ipolate functions, return

    Args:
        function_x:
        function_y:
        start_times:
        number_of_samples:
        noise: "random" or "lagged"
        plot:
        last_time: If the original stroke does not have extra time
                    built-in with a dummy last start stroke, specify the max here

    Returns:
        list of x_points, list of y_points, binary list of whether the corresponding point is a start stroke
    """
    adj_number_of_samples = number_of_samples - len(start_times)
    if last_time is None:
        _last_time = start_times[-1] # get the last start point - this should actually be the first start point of the next stroke!!
    else:
        _last_time = last_time
    interval = _last_time / (adj_number_of_samples)
    std_dev = interval / 4 # next point is ~3 std deviations away
    time = np.linspace(interval, _last_time - interval, adj_number_of_samples, dtype=np.float64)

    noise = "random" if noise is True else noise
    if noise:
        if noise == "random": # random noise over time/distance on each point
            #np.random.seed(9) 9 with index 27
            noises = np.random.normal(0, std_dev, time.shape)
        else:
            raise NotImplemented(f"Noise type: {noise} not implemented")
        if plot:
            plt.plot(time, noises)
            plt.show()
        time += noises
        time = np.maximum(time, 3*EPSILON) # 0 start stroke will be added below
        time = np.minimum(time, _last_time - 3 * EPSILON)

    time = np.r_[time, start_times]  # add the start times back in; last start time is the same as the end time

    # Add start stroke IDs
    is_start_stroke = np.zeros(len(time))
    is_start_stroke[-len(start_times):] = 1
    time = np.c_[time, is_start_stroke]

    # Sort based on time, split back up
    time = time[np.argsort(time[:, 0], kind='mergesort')]

    is_start_stroke = time[:, 1]
    time = time[:, 0]

    # Make sure the first/last strokes SOS
    if last_time is None:
        assert is_start_stroke[-1] == 1
    assert is_start_stroke[0] == 1
    assert len(time) == number_of_samples

    # start_times2 = time[is_start_stroke.astype(np.int32) == 1]
    # np.testing.assert_equal(start_times2,start_times)

    # x = np.array(function_x(time).tolist())
    # y = np.array(function_y(time).tolist())
    x = function_x(time)
    y = function_y(time)
    return x, y, is_start_stroke


def calc_stroke_distances(x,y,start_strokes):
    """ Calculate total distance of strokes

    Args:
        x: List of x's
        y: List of y's
        start_strokes: List of start stroke identifiers [1,0,0,1...

    Returns:
        distance travelled for each complete stroke
    """
    if isinstance(x, list):
        x=np.array(x)
    if isinstance(y, list):
        y=np.array(y)

    [start_indices] = np.where(start_strokes)
    end_idx = len(start_strokes)-1
    end_indices = np.append((start_indices-1)[1:], end_idx)
    cum_sum = reparameterize_as_func_of_distance(x,y,start_strokes)
    lengths = cum_sum[end_indices] - cum_sum[start_indices]
    return lengths

def reparameterize_as_func_of_distance(x, y, start_strokes, has_repeated_end=True):
    """ Instead of time, re-parameterize entire sequence as distance travelled

    Args:
        x: List of x's
        y: List of y's
        start_strokes: List of start stroke identifiers [1,0,0,1...
        has_repeated_end: the last point is repeated
    Returns:
        distance travelled for each complete stroke
    """
    x=np.asarray(x)
    y=np.asarray(y)

    distances = distance_metric(x,y) # same sized array with 0 as first distance
    distances[start_strokes==1] = 3*EPSILON # don't count pen up motion
    distances[0] = 0 # 0, not epsilon for first one
    cum_sum = np.cumsum(distances) # distance is 0 at first point; keeps desired_num_of_strokes the same

    if has_repeated_end:
        assert x[-1] == x[-2]
        cum_sum[-1] = cum_sum[-2]+10 # for interpolating later

    return cum_sum


def get_stroke_length_gt(x, y, start_points, use_distance=True):
    """ This is really a distance function
    
    Args:
        x: 
        y: 
        start_points: 
        use_distance: 

    Returns:

    """
    input_shape = start_points.shape

    start_indices = np.where(start_points)[0]
    start_point_ct = len(start_indices)
    last_idx = len(start_points)-1

    # Create list of start and end points (x's for interpolation)
    xs = np.repeat(start_indices,2)
    xs[::2] -= 1
    xs = np.append(xs[1:], last_idx)

    ## Create y's for interpolation (the target values)
    ys = np.zeros(2*start_point_ct)


    # Make each target the actual stroke desired_num_of_strokes
    if use_distance:
        distances = calc_stroke_distances(x,y, start_points)
        ys[1::2] = distances
    else:
        # Make each stroke desired_num_of_strokes "1" unit
        ys[1::2] += 1
    interp_xs = np.array(range(0, last_idx+1))
    out = np.interp(interp_xs, xs, ys)

    assert out.shape == input_shape
    return out

def test_gt_stroke_length_generator():
    x = np.array(range(0,17))#**2
    y = np.array(range(0,17))#**2
    m = np.array([1,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0])
    final = get_stroke_length_gt(x, y, m)
    print(final)
    assert np.allclose(final, to_numpy("""[0.         1.41421356 2.82842712 0.         1.41421356 2.82842712, 4.24264069 5.65685425 0.
                               1.41421356 2.82842712 4.24264069, 5.65685425 7.07106781 0.         1.41421356 2.82842712]"""))

    x = np.array(range(0,4))#**2
    y = np.array(range(0,4))#**2
    m = np.array([1,1,0,0,])
    final = get_stroke_length_gt(x, y, m)
    print(final)
    assert np.allclose(final, np.array([0.,0.,1.41421356,2.82842712]))

def to_numpy(array_string):
    from ast import literal_eval
    import numpy as np
    import re
    array_string = re.sub('[\s,]+', ',', array_string)
    array_string = np.array(literal_eval(array_string))
    return array_string

def post_process_remove_strays(gt, max_dist=.2):
    """ Must have stardard SOS 0/1 form (already unrelatified start of strokes)
        Removes points not near either neighbor point and not start strokes


    Args:
        gt:
        max_dist:

    Returns:

    """

    distances = distance_metric(gt[:, 0], gt[:, 1])

    # Where are the distances big AND not a start point
    idx = np.argwhere(distances > max_dist).reshape(-1)
    not_first_stroke = np.argwhere(gt[:, 2] == 0).flatten()
    bad_points = idx[np.where(np.diff(idx) == 1)]
    bad_points = np.intersect1d(not_first_stroke, bad_points)

    if bad_points:
        # Delete them
        gt = np.delete(gt, bad_points, axis=0)

        # Add new start point
        gt[bad_points, 2] = 1
    return gt

def make_more_starts(gt, max_dist=.15):
    """ Make more start points and delete superflouous ones

    Args:
        gt:
        max_dist:

    Returns:

    """
    if isinstance(gt, Tensor):
        gt = gt.detach().numpy()
    distances = distance_metric(gt[:, 0], gt[:, 1])
    idx = np.argwhere(distances > max_dist).reshape(-1)
    gt[idx, 2] = 1
    return gt

def remove_bad_points(gt, max_dist=.2):
    """ Must have already unrelatified start of strokes

    Args:
        gt:
        max_dist:

    Returns:

    """
    distances = distance_metric(gt[:, 0], gt[:, 1])
    idx = np.argwhere(distances > max_dist).reshape(-1)
    gt[idx, 2] = 1
    return gt

def get_nearest_point(reference, moving_component, reference_is_image=False, **kwargs):
    """ For calculating error, reference should be preds (how far do we need to move the GTs)
        For post-process, reference should be GT image/pts (where should we move this pred to?)

    Args:
        reference:
        moving_component:
        reference_is_image:

    Returns:

    """
    if reference_is_image:
        if len(reference.shape)==3:
            reference = np.squeeze(reference)
        height = reference.shape[0]
        y_coords,x_coords = np.where(reference<150/127.5-1)
        reference = np.c_[x_coords, height-y_coords].astype(np.float64) / height # rescale to be 0-1 based on height!

    if "kd" in kwargs and kwargs["kd"] is not None:
        kd = kwargs["kd"]
    else:
        warnings.warn("Generating KD tree")
        kd = KDTree(reference[:, :2])

    distances, neighbor_indices = kd.query(moving_component[:, :2])  # How far do we have to move the GT's to match the predictions? Based on 0-1 height scale
    nearest_points = reference[neighbor_indices]
    return nearest_points, distances, kd

def move_bad_points_deprecated(reference, moving_component, reference_is_image=False, max_distance=6, **kwargs):
    nearest_points, distances, kd = get_nearest_point(reference, moving_component, reference_is_image, **kwargs)
    if isinstance(moving_component, Tensor):
        moving_component = moving_component.detach().numpy()
    moving_component[:,0:2] = nearest_points
    print(moving_component)
    return moving_component #moving_component[distances<max_distance]


def preserve_deleted_start_points(moving_component, proposed_deletion_indices):
    """ If a starting point is deleted, move it to the next point

    Args:
        moving_component:
        proposed_deletion_indices:

    Returns:

    """

    lost_start_points = np.argwhere(np.round(moving_component[proposed_deletion_indices][:, 2]))
    lost_start_points = proposed_deletion_indices[lost_start_points]

    output_start_point_indices = []
    _temp_output_start_point_indices = lost_start_points + 1
    for i in _temp_output_start_point_indices.flatten().tolist():
        while i in proposed_deletion_indices:  # make sure that index is not going to be deleted
            i += 1
        if i < moving_component.shape[0]:
            output_start_point_indices.append(i)
    # globals().update(locals())
    return list(set(output_start_point_indices))


def move_bad_points(reference, moving_component, reference_is_image=True, max_distance=.04, **kwargs):
    # JUST FOR GRAPHING
    item = {"line_imgs": [reference[:, :1000]], "paths": [""]}
    preds = moving_component.clone().detach().numpy()

    # plt.figure(dpi=300)
    # print("Before fix")
    # graph(item, preds=[preds.copy().transpose([1, 0])], _type="eval", epoch="current", config=config, save_folder=None,
    #       plot_points=False, show=True)

    nearest_points, distances, kd = get_nearest_point(reference, moving_component, reference_is_image, **kwargs)
    if isinstance(moving_component, Tensor):
        moving_component = moving_component.detach().numpy()
    moving_component[:, 0:2] = nearest_points

    # Collapse start points a bit
    proposed_deletion_indices = np.argwhere(distances >= max_distance).flatten()
    new_start_points = preserve_deleted_start_points(moving_component, proposed_deletion_indices)
    moving_component[new_start_points, 2] = 1
    output = np.delete(moving_component, proposed_deletion_indices, axis=0)

    # output = moving_component[distances < max_distance]

    # globals().update(locals())
    # graph(item, preds=[output.copy().transpose([1, 0])], _type="eval", epoch="current", config=config, save_folder=None,
    #       plot_points=False, show=True)

    return output

def get_sos_args(sos, stroke_numbers=True):
    """
    Args:
        sos (array like): A list of stroke numbers (1,1,1,1,1,2,2,2,2...)
    Returns:

    """
    if stroke_numbers or not (sos <= 1).all():
        stroke_starts = np.argwhere(relativefy_numpy(sos)).flatten()
    else:
        stroke_starts = np.argwhere(sos).flatten()
    return stroke_starts

def invert_each_stroke(gt, stroke_numbers=True):
    """ reverse stroke point order

    Args:
        gt:
        stroke_numbers:

    Returns:
        GTs with strokes inverted
        SOS arg numbers (0,5,11...)
    """
    if not (gt[:, 2] <= 1).all() or stroke_numbers:
        stroke_starts = np.argwhere(relativefy_numpy(gt[:, 2])).flatten()
    else:
        stroke_starts = np.argwhere(gt[:, 2]).flatten()
    return np.concatenate([np.vstack([x[::-1] for x in np.split(gt[:,:2], stroke_starts) if x.size]), gt[:,2:]], axis=1), stroke_starts


def get_number_of_stroke_pts_from_gt(gt, stroke_numbers=True):
    """ Get the number of stroke points

    Returns:

    """
    if not (gt[:, 2] <= 1).all() or stroke_numbers:
        stroke_starts = np.argwhere(relativefy_numpy(gt[:, 2])).flatten()
    else:
        stroke_starts = np.argwhere(gt[:, 2]).flatten()
    stroke_starts = np.concatenate([stroke_starts,[len(gt)]])
    return stroke_starts[1:]-stroke_starts[:-1]


origin = np.array([0, 1])
distance = lambda x: np.sum((np.asarray(x) - origin) ** 2)

def reorder_strokes(gt, stroke_numbers=False, sos_index=2):
    """ Reverse strokes so start/end point nearest to top left of image is first
        Re-order strokes so stroke with point nearest top left is first

    Args:
        gt:
        stroke_numbers:
        sos_index:

    Returns:

    """
    sos_args = get_sos_args(gt[:,sos_index], stroke_numbers=stroke_numbers)
    strokes = np.split(gt, sos_args)

    # Reverse strokes as needed - start point is always the top-leftmost point
    del_index = []
    for i, stroke in enumerate(strokes):
        if stroke.size:
            # Swap
            if distance(stroke[0, :2]) > distance(stroke[-1, :2]):
                strokes[i][:, :2] = stroke[::-1, :2]
        else:
            del_index.append(i)

    for i in del_index:
        del strokes[i]

    # Reorder strokes as needed - left most start point goes first
    reorder = sorted(strokes, key=lambda stroke: distance(stroke[0, :2]))
    output = np.concatenate(reorder)

    if stroke_numbers: # Stroke numbers are now out of order; find where the strokes change, then re-add
        sos = relativefy(output[:,2])!=0
        output[:,2] = np.cumsum(sos) # Regenerate stroke numbers

        # Should get the same result here
        #np.testing.assert_equal(stroke_recovery.relativefy(output[:,2]), sos)

    return output

def swap_strokes_left_v1(gt, stroke_numbers=False, sos_index=2, distance_threshold=.1):
    """ Reverse strokes so start/end point nearest to top left of image is first
        Re-order strokes so stroke with point nearest top left is first

    Args:
        gt:
        stroke_numbers:
        sos_index:

    Returns:

    """
    # gt = np.array(range(36)).reshape(9, 4)
    # gt[:, 2] = [1, 0, 0, 1, 0, 1, 0, 0, 0]
    # gt[-1, :] = [3.4, 5, 1, 1]
    # print(gt)

    sos_args = get_sos_args(gt[:, sos_index], stroke_numbers=stroke_numbers)
    strokes = np.split(gt, sos_args)[1:]

    avg_x_position = [np.mean(s[:, 0]) for s in strokes if s.size]
    avg_x_position += np.cumsum(
        [distance_threshold] * len(avg_x_position))  # this acts like a threshold, but increases for larger stroke swaps

    x = list(zip(strokes, avg_x_position))
    x.sort(key=lambda m: m[1])

    output = np.concatenate([i[0] for i in x])

    if stroke_numbers: # Stroke numbers are now out of order; find where the strokes change, then re-add
        sos = relativefy(output[:,2])!=0
        output[:,2] = np.cumsum(sos) # Regenerate stroke numbers

    return output


def swap_strokes_left(gt, stroke_numbers=False, sos_index=2, distance_threshold=.1, height=None):
    """ Swap stroke if the most left part is further right than the previous stroke

    Args:
        gt:
        stroke_numbers:
        sos_index:

    Returns:

    """
    if height:
        distance_threshold *= height

    # gt = np.array(range(36)).reshape(9, 4)
    # gt[:, 2] = [1, 0, 0, 1, 0, 1, 0, 0, 0]
    # gt[-1, :] = [3.4, 5, 1, 1]
    # print(gt)

    sos_args = get_sos_args(gt[:, sos_index], stroke_numbers=stroke_numbers)
    strokes = np.split(gt, sos_args)[1:]

    min_x_position = [np.min(s[:, 0]) for s in strokes if s.size]
    min_x_position += np.cumsum([distance_threshold] * len(min_x_position))  # this acts like a threshold, but increases for larger stroke swaps

    x = list(zip(strokes, min_x_position))
    x.sort(key=lambda m: m[1])

    output = np.concatenate([i[0] for i in x])

    if stroke_numbers: # Stroke numbers are now out of order; find where the strokes change, then re-add
        sos = relativefy(output[:,2])!=0
        output[:,2] = np.cumsum(sos) # Regenerate stroke numbers

    return output


def swap_items(l, pos1, pos2):
    l[pos1], l[pos2] = l[pos2], l[pos1]
    return l

def swap_strokes(gt, start, end, pivot):
    """

    Args:
        gt: WIDTH x VOCAB (3)
        start (int): index of end of stroke
        end (int): index of start of stroke
        pivot (int): index of stroke change

    Returns:

    """
    t = np.copy(gt[start:end])
    gt[start:start + end - pivot] = t[pivot - start:end - start]
    gt[start + end - pivot:end] = t[:pivot - start]
    return gt

def swap_to_minimize_l1(pred, gt, exponent=2, stroke_numbers=True, center_of_mass=False):
    """

    Args:
        pred:
        gt:
        exponent:
        stroke_numbers:
        center_of_mass: take center of mass (true) vs.

    Returns:

    """
    gt_stroke_lens = get_number_of_stroke_pts_from_gt(gt, stroke_numbers=stroke_numbers)
    ordering = list(range(len(gt_stroke_lens)))
    pos = 0
    for i in range(len(gt_stroke_lens) - 1):
        end = pos + gt_stroke_lens[i] + gt_stroke_lens[i + 1]
        normal_slice = slice(pos, end)
        alt_gt = np.concatenate([gt[pos + gt_stroke_lens[i]:end, :2], gt[pos:pos + gt_stroke_lens[i], :2]])
        if center_of_mass:
            normal_l1 = abs(np.sum(gt[normal_slice, :2],axis=0) - np.sum(pred[normal_slice, :2], axis=0))
            alternative_l1 = abs(np.sum(alt_gt, axis=0) - np.sum(pred[normal_slice, :2], axis=0))
        else:
            normal_l1 = abs(gt[normal_slice, :2] - pred[normal_slice, :2]) ** exponent
            alternative_l1 = abs(alt_gt - pred[normal_slice, :2]) ** exponent
        #         if i == 0:
        #             print(alt_gt)
        #             print(alternative_l1)
        #             print(normal_l1)
        if np.sum(alternative_l1) < np.sum(normal_l1):
            #print(f"Swap {i} {i + 1}")
            gt = swap_strokes(gt, pos, end, pivot=pos + gt_stroke_lens[i])
            swap_items(gt_stroke_lens, i, i + 1)
            swap_items(ordering, i, i + 1)
        pos += gt_stroke_lens[i]

    if stroke_numbers: # Stroke numbers are now out of order; find where the strokes change, then re-add
        sos = relativefy(gt[:,2])!=0
        gt[:,2] = np.cumsum(sos) # Regenerate stroke numbers

    return gt

## KD TREE MOVE POINTS? TEST THIS
## DELETE POINTS THAT AREN'T CLOSE TO A STROKE
## ANY SUFFICIENTLY LARGE JUMP -> MAKE A START STROKE
## DTW -> PAIR POINTS TOGETHER, EVALUATE HOW ACCURATE ON STROKE BY STROKE LEVEL



if __name__=="__main__":
    test_conv_weight()
    #test_gt_stroke_length_generator()
    Stop
    os.chdir("../data")
    with open("online_coordinate_data/3_stroke_16_v2/train_online_coords.json") as f:
        output_dict = json.load(f)

    instance = output_dict[11]
    render_points_on_image(instance['gt'], img=instance['image_path'], x_to_y=instance["x_to_y"])
