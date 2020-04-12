import torch
import torch.nn as nn
from hwr_utils import hwr_logger
import logging
#logger = hwr_logger.logger
logger = logging.getLogger("root."+__name__)

'''
An alternative implementation for PyTorch with auto-infering the x-y dimensions.
'''

class AddCoords(nn.Module):

    def __init__(self, with_r=False, zero_center=True, rectangle_x=False, both_x=False, with_sin=True, y_only=False):
        """ Include a rectangle and non-rectangle x

        Args:
            with_r:
            zero_center:
            rectangle_x:
            both_x:
        """
        super().__init__()
        self.with_r = with_r
        self.rectangle_x = rectangle_x
        self.zero_center = zero_center
        self.both = both_x
        self.with_sin = with_sin
        self.y_only = y_only

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, height, width = input_tensor.size()

        height_channel = torch.arange(height).repeat(1, width, 1)
        width_channel = torch.arange(width).repeat(1, height, 1).transpose(1, 2)

        # Rescale from 0 to 1
        height_channel = height_channel.float() / (height - 1)
        width_channel = width_channel.float() / (width - 1)

        # Rescale from -1 to 1
        if self.zero_center:
            width_channel = width_channel * 2 - 1
            height_channel = height_channel * 2 - 1

        if self.y_only:
            height_channel = height_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

            ret = torch.cat([
                input_tensor,
                height_channel.type_as(input_tensor)], dim=1)

        elif self.both: # Create default CoordConv, and one scaled to be same scale as y CoordConv
            xx_rec_channel = width_channel * width / height

            height_channel = height_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
            width_channel = width_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
            xx_rec_channel = xx_rec_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

            ret = torch.cat([
                input_tensor,
                height_channel.type_as(input_tensor),
                xx_rec_channel.type_as(input_tensor),
                width_channel.type_as(input_tensor)], dim=1)

            if self.with_sin:
                xx_sine_channel = torch.sin(xx_rec_channel * 4)
                ret = torch.cat([ret, xx_sine_channel.type_as(input_tensor)], dim=1)

        else:
            if self.rectangle_x:
                width_channel *= width / height

            height_channel = height_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
            width_channel = width_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

            ret = torch.cat([
                input_tensor,
                height_channel.type_as(input_tensor),
                width_channel.type_as(input_tensor)], dim=1)


        if self.with_r:
            rr = torch.sqrt(torch.pow(height_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(width_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret

class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, with_r=False, verbose=False, zero_center=True, method="y_rel", with_sin=False, **kwargs):
        super().__init__()
        in_size = in_channels
        if method=="y_abs":
            method = "y_rel"


        if method=="y_rel":
            rectangle_x = False
            both_x = False
            y_only = True
            in_size = in_channels+1
        elif "y_rel" in method:
            if "x_abs" in method and "x_rel" in method:
                rectangle_x = True
                both_x = True
                y_only = False
                in_size = in_channels + 3
            elif "x_rel" in method:
                rectangle_x = False
                both_x = False
                y_only = False
                in_size = in_channels+2
            elif "x_abs" in method:
                rectangle_x = True
                both_x = False
                y_only = False
                in_size = in_channels + 2
        logger.info(f"COORD CONV: {method}")
        if with_sin:
            in_size += 1

        self.addcoords = AddCoords(with_r=with_r, zero_center=zero_center, rectangle_x=rectangle_x, both_x=both_x, with_sin=with_sin, y_only=y_only)
        self.verbose = verbose
        if with_r:
            in_size += in_channels+3

        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)
        logger.info(f"Zero center {zero_center}")
        logger.info(f"RECT X Coord: {rectangle_x or both_x}")
        logger.info(f"Normalized X Coord: {not rectangle_x or both_x}")
        logger.info(f"Using ABS+REL X Coord Channels: {both_x}")
        logger.info(f"X: {with_sin}")
        logger.info(f"Y: {with_sin}")

    def forward(self, x):
        ret = self.addcoords(x)
        ## See the coordConvs:
        if self.verbose:
            print(ret[0,-1])
            print(ret[0,-2])
        ret = self.conv(ret)
        return ret

def test_cnn():
    import torch
    from models.basic import BidirectionalRNN, CNN
    import torch.nn as nn

    cnn = CNN(nc=1, first_conv_op=CoordConv, verbose=False)
    # cnn = CCNN(nc=1, conv_op=nn.Conv2d)
    batch = 7
    y = torch.rand(batch, 1, 60, 1024)
    a = cnn(y)
    globals().update(locals())


if __name__=="__main__":
    test_cnn()