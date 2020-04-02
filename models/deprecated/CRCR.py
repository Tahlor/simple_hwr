from torch import nn
from models.basic import GeneralizedBRNN


class CRCR(nn.Module):
    def __init__(self, cnnOutSize=1024, nc=3, leakyRelu=False, type="default"):
        """ Height must be set to be consistent; width is variable, longer images are fed into BLSTM in longer sequences

        The CNN learns some kind of sequential ordering because the maps are fed into the LSTM sequentially.

        Args:
            cnnOutSize: DOES NOT DO ANYTHING! Determined by architecture
            nc:
            leakyRelu:
        """
        super().__init__()
        self.cnnOutSize = cnnOutSize
        self.cnn = self.default_CRCR(nc=nc, leakyRelu=leakyRelu)
        print("Creating a CNN with Recurrent layer")

    def default_CRCR(self, nc=3, leakyRelu=False):

        ks = [3, 3, 3, 3, 3, 3, 2] # kernel size 3x3
        ps = [1, 1, 1, 1, 1, 1, 0] # padding
        ss = [1, 1, 1, 1, 1, 1, 1] # stride
        nm = [64, 128, 256, 256, 512, 512, 512] # number of channels/maps

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(in_channels=nIn, out_channels=nOut, kernel_size=ks[i], stride=ss[i], padding=ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
            #cnn.add_module(f"printAfter{i}", PrintLayer(name=f"printAfter{i}"))

        # input: 16, 1, 60, 256; batch, channels, height, width
        convRelu(0) # 16, 64, 60, 1802
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 16, 64, 30, 901
        convRelu(1) # 16, 128, 30, 901
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 16, 128, 15, 450
        convRelu(2, True) # 16, 256, 15, 450
        convRelu(3) # 16, 256, 15, 450
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 16, 256, 7, 451 b,w,h,c # kernel_size, stride, padding

        # Throw in an RNN here
        #cnn.add_module('Print', PrintLayer())
        input_size = 256*7
        self.crcr = GeneralizedBRNN(input_size, input_size, input_size, permute=True, num_layers=2, dropout=.5)
        cnn.add_module('rnn{0}'.format(0), self.crcr)

        convRelu(4, True) # 16, 512, 7, 451
        convRelu(5) # 16, 512, 7, 451
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 16, 512, 3, 452
        convRelu(6, True)  # 16, 512, 2, 451
        return cnn

    def post_process(self, conv):
        b, c, h, w = conv.size() # something like 16, 512, 2, 406
        conv = conv.view(b, -1, w)  # batch, Height * Channels, Width

        # Width effectively becomes the "time" seq2seq variable
        output = conv.permute(2, 0, 1)  # [w, b, c], first time: [404, 8, 1024] ; second time: 213, 8, 1024
        return output

    def forward(self, input):
        x = self.post_process(self.cnn(input))
        return x