from torch import nn
import torch
import sys
sys.path.append("..")
import numpy as np

MAX_LENGTH = 64

class Renderer(nn.Module):
    def __init__(self, input_vocab_size=3, device="cuda", cnn_type="default64", **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.cnn = self.uncnn = UNCNN(nc=1, cnn_type=cnn_type)

        hidden_size = 1024
        bidirectional = True
        hidden_size = int(hidden_size / 2) if bidirectional else hidden_size
        self.encoder = nn.LSTM(input_size=1024, hidden_size=hidden_size, bidirectional=bidirectional, dropout=.5, num_layers=2)
        self.linear = nn.Linear(input_vocab_size, 1024)
        self.device = device

    def forward(self, input):
        """

        Args:
            input: B x W x (X,Y,SOS) - sequence of relative movements
            gts:

        Returns:

        """

        if self.training:
            return self._forward(input)
        else:
            with torch.no_grad():
                return self._forward(input)

    def _forward(self, input):
        """

        Args:
            input (tensor): GTs, e.g. B x T x 3 (X,Y,SOS)

        Returns:

        """

        # Embedding
        b, T, h = input.size()
        t_rec = input.view(T * b, h)
        embed = self.linear(t_rec) # [T * b, nOut], T*b is the batch size for a FC
        embed = embed.view(b, T, -1).permute(1,0,2) # T x Batch x 1024
        encoding, _ = self.encoder(embed)
        image = self.uncnn(encoding)
        return image

class UNCNN(nn.Module):
    def __init__(self, cnnInSize=1024, nc=1, ngf=64, cnn_type="default64"):
        """ Channels, CNN_IN_SIZE=1024

        Args:
            cnnInSize:
            nc (int): Number of channels in the training images. For color images this is 3
            ngf (int): Size of feature maps in generator

        """
        super().__init__()
        self.cnnInSize = cnnInSize
        self.cnn_type = cnn_type
        late_stride = 2 if cnn_type=="default" else [2,1]
        self.main = nn.Sequential(
            # input is Z, going into a convolution, 16, 512, 2, 451
            # originally: 64, nz, 1, 1,
            ### in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0,
            nn.ConvTranspose2d( int(self.cnnInSize/2), ngf * 8, 4, 1, 0, bias=False), # increasing stride doubles output
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, late_stride, 2, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, late_stride, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, late_stride, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, late_stride, 1, bias=False),
            nn.Sigmoid() # Tanh, Sigmoid
            # state size. (nc) x 64 x 64
            # OUTPUT: BATCH 61 x 1 x H=60 x W

            # Default64: max pools on first one, upsamples after last one
            # Default: max pools on first 2 convs
        )

    def forward(self, input):
        cnn_input = self.unpost_process(input) # W x B x Ch -> B x 512 x 2 x W
        image = self.main(cnn_input) # B x 512 x 2 x W -> B x
        return image

    def unpost_process(self, encoding):
        w, b, c = encoding.size() # Width, Batch, Channels (1024)
        conv_input = encoding.permute(1, 2, 0) # -> batch, height * channels, width
        conv_input = conv_input.view(b, 512, 2, w) # -> BATCH, CHANNEL_DIM (512), 2, WIDTH
        return conv_input

def test_uncnn():
    U = UNCNN()
    #input = torch.rand(16, 512, 2, 451)
    input = torch.rand(213, 16, 1024)
    x = U(input)
    # 123 (128 after padding etc.),16,1024 => 80 x 512
    # 213 (218) "" ""   => 80 x 872
    print(x.shape)
    pass

def test_renderer():
    R = Renderer()
    input = torch.rand(16, 101, 3)
    out = R(input)
    print(out.shape)

if __name__=='__main__':
    #test_uncnn()
    test_renderer()