from models import start_points
from torch import nn
import torch
import numpy as np
from torch import tensor, Tensor

def test_start_point_attn():
    spa = start_points.StartPointAttnModel1(device="cpu")
    shape = (3,1,60,100)
    input_item = Tensor(np.random.random(shape))
    spa(input_item)

if __name__=='__main__':
    test_start_point_attn()