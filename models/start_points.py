from torch import nn
import torch
from .basic import CNN, BidirectionalRNN
from .CoordConv import CoordConv
import numpy as np

MAX_LENGTH = 64

class StartPointModel(nn.Module):
    def __init__(self, vocab_size=3, device="cuda", cnn_type="default", first_conv_op=CoordConv, first_conv_opts=None, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        if first_conv_op:
            first_conv_op = CoordConv
        self.cnn = CNN(nc=1, first_conv_op=first_conv_op, cnn_type=cnn_type, first_conv_opts=first_conv_opts)
        self.encoder = nn.LSTM(input_size=1024, hidden_size=1024, bidirectional=True, dropout=.5, num_layers=1)
        # self.linear1 = nn.Linear
        self.decoder = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=2, dropout=.5)
        self.linear = nn.Linear(1024, vocab_size)
        self.device = device

    def forward(self, input, gts=None):
        if self.training:
            return self._forward(input, gts)
        else:
            with torch.no_grad():
                return self._forward(input, gts)

    def _forward(self, input, gts=None):
        cnn_output = self.cnn(input)
        _, hidden = self.encoder(cnn_output)  # width, batch, alphabet
        _, b, _ = hidden[0].shape

        outputs = []
        output = torch.zeros((1, b, 1024)).to(self.device)
        for i in range(MAX_LENGTH):
            output, hidden = self.decoder(output, hidden)
            #output = nn.functional.relu(output)
            outputs.append(self.linear(output))

        # sigmoids are done in the loss
        outputs = torch.cat(outputs, dim=0)
        return outputs

class StartPointModel2(nn.Module):
    def __init__(self, vocab_size=3, device="cuda", cnn_type="default", first_conv_op=CoordConv, first_conv_opts=None, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        if first_conv_op:
            first_conv_op = CoordConv
        self.decoder_size = 256
        self.cnn = CNN(nc=1, first_conv_op=first_conv_op, cnn_type=cnn_type, first_conv_opts=first_conv_opts)
        self.encoder = nn.LSTM(input_size=1024, hidden_size=self.decoder_size, bidirectional=True, dropout=.5, num_layers=1)
        self.decoder = nn.LSTM(input_size=self.decoder_size, hidden_size=self.decoder_size, num_layers=2, dropout=.5)
        self.linear = nn.Linear(self.decoder_size, vocab_size)
        self.device = device

    def forward(self, input, gts=None):
        if self.training:
            return self._forward(input, gts)
        else:
            with torch.no_grad():
                return self._forward(input, gts)

    def _forward(self, input, gts=None):
        cnn_output = self.cnn(input)
        _, hidden = self.encoder(cnn_output)  # width, batch, alphabet
        _, b, _ = hidden[0].shape

        outputs = []
        output = torch.zeros((1, b, self.decoder_size)).to(self.device)
        for i in range(MAX_LENGTH):
            output, hidden = self.decoder(output, hidden)
            #output = nn.functional.relu(output)
            outputs.append(self.linear(output))

        # sigmoids are done in the loss
        outputs = torch.cat(outputs, dim=0)
        return outputs

class StartPointAttnModel(nn.Module):
    def __init__(self, vocab_size=3, device="cuda", cnn_type="default", first_conv_op=CoordConv, first_conv_opts=None, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        if first_conv_op:
            first_conv_op = CoordConv
        self.cnn = CNN(nc=1, first_conv_op=first_conv_op, cnn_type=cnn_type, first_conv_opts=first_conv_opts)
        self.encoder = nn.LSTM(input_size=1024, hidden_size=256)
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=1)
        self.decoder = nn.LSTM(input_size=512, hidden_size=256, num_layers=1)
        self.linear = nn.Linear(256, vocab_size)
        self.device = device

    def forward(self, input, gts=None):
        if self.training:
            return self._forward(input, gts)
        else:
            with torch.no_grad():
                return self._forward(input, gts)

    def _forward(self, input, gts=None):
        cnn_output = self.cnn(input)
        encoding, hidden = self.encoder(cnn_output)  # width, batch, alphabet
        _, b, _ = hidden[0].shape

        outputs = []
        output = torch.zeros((1, b, 256)).to(self.device)
        for i in range(MAX_LENGTH):
            context, _ = self.attn(output, encoding, encoding)
            output, hidden = self.decoder(torch.cat([output, context], dim=-1), hidden)
            output = nn.functional.relu(output)
            outputs.append(self.linear(output))

        # sigmoids are done in the loss
        outputs = torch.cat(outputs, dim=0)
        return outputs

class StartPointAttnModelDeep(nn.Module):
    def __init__(self, vocab_size=3, device="cuda", cnn_type="default", first_conv_op=CoordConv, first_conv_opts=None, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        if first_conv_op:
            first_conv_op = CoordConv

        self.decoder_layers = 2
        self.encoder_layers = 2
        self.context_dim = 128
        self.vocab_size = vocab_size
        self.decoder_dim = 128
        self.cnn = CNN(nc=1, first_conv_op=first_conv_op, cnn_type=cnn_type, first_conv_opts=first_conv_opts)
        self.encoder = nn.LSTM(input_size=1024, hidden_size=128, num_layers=self.encoder_layers)
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4)
        self.decoder = nn.LSTM(input_size=self.context_dim+self.decoder_dim, hidden_size=self.decoder_dim, num_layers=self.decoder_layers)
        self.linear = nn.Linear(self.decoder_dim, vocab_size)
        self.device = device

    def forward(self, input, gts=None):
        if self.training:
            return self._forward(input, gts)
        else:
            with torch.no_grad():
                return self._forward(input, gts)

    def _forward(self, input, gts=None):
        cnn_output = self.cnn(input)
        encoding, hidden = self.encoder(cnn_output)  # width, batch, alphabet
        _, b, _ = hidden[0].shape

        outputs = []
        output = torch.zeros((1, b, self.decoder_dim)).to(self.device)
        hidden = 2 * [torch.zeros((2, b, self.decoder_dim)).to(self.device)]
        for i in range(MAX_LENGTH):
            context, _ = self.attn(output, encoding, encoding)
            output, hidden = self.decoder(torch.cat([output, context], dim=-1), hidden)
            output = nn.functional.relu(output)
            outputs.append(self.linear(output))

        # sigmoids are done in the loss
        outputs = torch.cat(outputs, dim=0)
        return outputs


class StartPointAttnModelFull(nn.Module):
    def __init__(self, vocab_size=4, device="cuda", cnn_type="default", first_conv_op=CoordConv, first_conv_opts=None, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        if first_conv_op:
            first_conv_op = CoordConv

        self.decoder_layers = 2
        self.encoder_layers = 2
        self.context_dim = 128
        self.vocab_size = vocab_size
        self.decoder_dim = 128
        self.cnn = CNN(nc=1, first_conv_op=first_conv_op, cnn_type=cnn_type, first_conv_opts=first_conv_opts)
        self.encoder = nn.LSTM(input_size=1024, hidden_size=128, num_layers=self.encoder_layers)
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4)
        self.decoder = nn.LSTM(input_size=self.context_dim+vocab_size, hidden_size=self.decoder_dim, num_layers=self.decoder_layers)
        self.linear = nn.Linear(self.context_dim+self.decoder_dim, vocab_size)
        self.device = device

    def forward(self, input, gts=None):
        if self.training:
            return self._forward(input, gts)
        else:
            with torch.no_grad():
                return self._forward(input, gts)

    def _forward(self, input, gts=None):
        print(input.shape)
        cnn_output = self.cnn(input)
        encoding, hidden = self.encoder(cnn_output)  # width, batch, alphabet

        #assert encoding[-1,-1,-1] == hidden[0][-1,-1,-1])
        # https://stackoverflow.com/questions/48302810/whats-the-difference-between-hidden-and-output-in-pytorch-lstm
        _, b, _ = encoding.shape
        outputs = []

        hidden = [torch.zeros(2, b, self.decoder_dim).to(self.device)] * self.decoder_layers
        y = torch.zeros((1, b, self.vocab_size)).to(self.device)
        c_t = torch.zeros((1, b, self.context_dim)).to(self.device)
        for i in range(MAX_LENGTH):
            s1, hidden = self.decoder(torch.cat([y, c_t], dim=-1), hidden)
            c_t, _ = self.attn(s1, encoding, encoding)
            y = self.linear(torch.cat([s1, c_t], dim=-1))
            outputs.append(y)
        # sigmoids are done in the loss
        outputs = torch.cat(outputs, dim=0)
        return outputs