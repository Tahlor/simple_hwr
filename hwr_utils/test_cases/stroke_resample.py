import sys
sys.path.append("../../")
sys.path.append("../")
import os
import numpy as np
from hwr_utils import *
from hwr_utils.stroke_plotting import *
from hwr_utils.stroke_recovery import *
import json

def test():
    json_path = "../../data/online_coordinate_data/8_stroke_vSmall_16/train_online_coords.json"
    json_path = "/media/data/GitHub/simple_hwr/data/online_coordinate_data/MAX_stroke_vlargeTrnSetFull/train_online_coords.json"
    parameter = "d"
    with open(json_path) as f:
        output_dict = json.load(f)

    for x in output_dict:
        output = prep_stroke_dict(x["raw"], time_interval=0, scale_time_distance=True) # list of dictionaries, 1 per file
        x = output["x"]
        y = output["y"]
        is_start_stroke = output["start_strokes"]
        gt = np.array([x,y,is_start_stroke]).transpose([1,0])
        #img = draw_from_gt(gt, show=True, use_stroke_number=False, plot_points=False, linewidth=1)

        # Resample
        x_func, y_func = stroke_recovery.create_functions_from_strokes(output, parameter=parameter) # can be d if the function should be a function of distance
        starts = output.start_times if parameter=="t" else output.start_distances
        x, y, is_start_stroke = stroke_recovery.sample(x_func, y_func, starts, 200, noise=None)
        gt = np.array([x,y,is_start_stroke]).transpose([1,0])
        #img = draw_from_gt(gt, show=True, use_stroke_number=False, plot_points=False, linewidth=1)

if __name__=='__main__':
    test()