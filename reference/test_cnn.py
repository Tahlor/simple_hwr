from torch import nn
from hwr_utils.utils import *
#from torchvision.models import resnet
from models import resnet

class CNN(nn.Module):
    def __init__(self, cnnOutSize=1024, nc=3, leakyRelu=False, type="default"):
        """ Height must be set to be consistent; width is variable, longer images are fed into BLSTM in longer sequences

        The CNN learns some kind of sequential ordering because the maps are fed into the LSTM sequentially.

        Args:
            cnnOutSize: DOES NOT DO ANYTHING! Determined by architecture
            nc:
            leakyRelu:
        """
        super(CNN, self).__init__()
        self.cnnOutSize = cnnOutSize
        # self.average_pool = nn.AdaptiveAvgPool2d((512,2))
        # 512, 7, 451
        self.average_pool4 = nn.AvgPool2d(())

        if type == "default":
            self.cnn = self.default_CNN(nc=nc, leakyRelu=leakyRelu)
        elif type == "resnet":
            # self.cnn = torchvision.models.resnet101(pretrained=False)
            self.cnn = resnet.resnet18(pretrained=False, channels=nc)
        elif type == "resnet34":
            self.cnn = resnet.resnet34(pretrained=False, channels=nc)
        elif type == "resnet101":
            self.cnn = resnet.resnet101(pretrained=False, channels=nc)

    def default_CNN(self, nc=3, leakyRelu=False):

        ks = [3, 3, 3, 3, 3, 3, 2]  # kernel size 3x3
        ps = [1, 1, 1, 1, 1, 1, 0]  # padding
        ss = [1, 1, 1, 1, 1, 1, 1]  # stride
        nm = [64, 128, 256, 256, 512, 512, 512]  # number of channels/maps

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(in_channels=nIn, out_channels=nOut, kernel_size=ks[i], stride=ss[i],
                                     padding=ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
            # cnn.add_module(f"printAfter{i}", PrintLayer(name=f"printAfter{i}"))

        # input: 16, 1, 60, 256; batch, channels, height, width
        convRelu(0)  # 16, 64, 60, 1802
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 16, 64, 30, 901
        convRelu(1)  # 16, 128, 30, 901
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 16, 128, 15, 450
        convRelu(2, True)  # 16, 256, 15, 450
        convRelu(3)  # 16, 256, 15, 450
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 16, 256, 7, 451 # kernel_size, stride, padding
        convRelu(4, True)  # 16, 512, 7, 451
        convRelu(5)  # 16, 512, 7, 451
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 16, 512, 3, 452
        convRelu(6, True)  # 16, 512, 2, 451
        return cnn

    """
    0 0 Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    1 ReLU(inplace)
    2 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    3 1 Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    4 ReLU(inplace)
    5 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    6 2 Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    7 BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    8 ReLU(inplace)
    9 3 Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    10 ReLU(inplace)
    11 MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1), dilation=1, ceil_mode=False)
    12 4 Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    13 BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    14 ReLU(inplace)
    15 5 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    16 ReLU(inplace)
    17 MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1), dilation=1, ceil_mode=False)
    18 6 Conv2d(512, 512, kernel_size=(2, 2), stride=(1, 1))
    19 BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    20 ReLU(inplace)
    """

    def post_process(self, conv):
        b, c, h, w = conv.size()  # something like 16, 512, 2, 406
        # print(conv.size())
        conv = conv.view(b, -1, w)  # batch, Height * Channels, Width

        # Width effectively becomes the "time" seq2seq variable
        output = conv.permute(2, 0, 1)  # [w, b, c], first time: [404, 8, 1024] ; second time: 213, 8, 1024
        return output


    def forward(self, input, intermediate_level=None):
        # INPUT: BATCH, CHANNELS (1 or 3), Height, Width
        if intermediate_level is None:
            x = self.post_process(self.cnn(input))
            return x
        else:
            conv = self.cnn[0:intermediate_level](input)
            conv2 = self.cnn[intermediate_level:](conv)
            return self.post_process(conv2), conv


if __name__ == "__main__":
    cnn = CNN(nc=1)
    pool = nn.MaxPool2d(3, (4, 1), padding=1)
    #pool = nn.MaxPool2d((2, 2), (4, 1), (0, 1))
    batch = 7
    y = torch.rand(batch, 1, 60, 1024)
    a, b = cnn(y, intermediate_level=13)
    new = cnn.post_process(pool(b))

    final = torch.cat([a, new], dim=2)
    print(a.size())
    print(final.size())

    def loop():
        for x in range(1000,1100):
            y = torch.rand(2, 1, 60, x)
            # a = cnn(y)
            # print(a.size())
            # STop
            a,b = cnn(y, intermediate_level=13)

            print(a.size(), b.size())
            new = cnn.post_process(pool(b)).size()
            print(new)
            assert new == a.size()
    loop()