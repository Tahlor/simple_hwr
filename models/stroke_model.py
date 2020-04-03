from torch import nn
import torch
from .basic import CNN, BidirectionalRNN
from .CoordConv import CoordConv
from hwr_utils.utils import is_dalai

class StrokeRecoveryModel(nn.Module):
    def __init__(self, vocab_size=5, device="cuda", cnn_type="default64", first_conv_op=CoordConv, first_conv_opts=None, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

        if not "nHidden" in kwargs:
            self.nHidden = 128
        if not "num_layers" in kwargs:
            self.num_layers = 2

        if first_conv_op:
            first_conv_op = CoordConv
        if not is_dalai():
            self.rnn = BidirectionalRNN(nIn=1024, nHidden=self.nHidden, nOut=vocab_size, dropout=.5, num_layers=self.num_layers, rnn_constructor=nn.LSTM)
            self.cnn = CNN(nc=1, first_conv_op=first_conv_op, cnn_type=cnn_type, first_conv_opts=first_conv_opts)
        else:
            self.rnn = BidirectionalRNN(nIn=64, nHidden=1, nOut=vocab_size, dropout=.5, num_layers=1,
                                        rnn_constructor=nn.LSTM)
            self.cnn = fake_cnn
            self.cnn.cnn_type = "FAKE"
            print("DALAi!!!!")

    def forward(self, input):
        if self.training:
            #print("TRAINING MODE")
            return self._forward(input)
        else:
            with torch.no_grad():
                #print("EVAL MODE")
                return self._forward(input)

    def _forward(self, input):
        cnn_output = self.cnn(input) # W, B, 1024
        # w,b,d = cnn_output.shape
        # width_positional = torch.arange(w).repeat(1, w, 1) / 60
        # sine_width_positional = torch.arange(w).repeat(1, w, 1) / 60

        rnn_output = self.rnn(cnn_output) # width, batch, alphabet
        # sigmoids are done in the loss
        return rnn_output


def fake_cnn(img):
    b, c, h, w = img.shape
    #print(w % 2 + w)
    return torch.ones(w % 2 + w+2, b, 64)

fake_cnn.cnn_type = "FAKE"

class StartEndPointReconstructor(nn.Module):
    def __init__(self, vocab_size=2, device="cuda", cnn_type="default64", first_conv_op=CoordConv, first_conv_opts=None, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        if first_conv_op:
            first_conv_op = CoordConv
        vocab_size = 2 # OVERRIDE

        self.decoder_layers = 2
        self.encoder_layers = 2
        self.context_dim = 128
        self.vocab_size = vocab_size
        self.decoder_dim = 128
        self.embed_pos = torch.nn.Sequential( # absolute/relative start and end positions
                torch.nn.Linear(8, 128),
                torch.nn.ReLU())
        self.cnn = CNN(nc=1, first_conv_op=first_conv_op, cnn_type=cnn_type, first_conv_opts=first_conv_opts)
        self.encoder = nn.LSTM(input_size=1024, hidden_size=128, num_layers=self.encoder_layers)
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4)
        self.decoder = nn.LSTM(input_size=self.context_dim+vocab_size, hidden_size=self.decoder_dim, num_layers=self.decoder_layers)
        self.linear = nn.Linear(self.context_dim+self.decoder_dim, vocab_size)
        self.device = device

    def forward(self, **kwargs):
        if self.training:
            return self._forward(**kwargs)
        else:
            with torch.no_grad():
                return self._forward(**kwargs)

    def _forward(self, start_end_points, image):
        """

        Args:
            start_end_points: Batch X Stroke X 2 (SOS and EOS)
            image:

        Returns:

        """
        embedding = self.start_point_embedding(start_end_points)
        cnn_output = self.cnn(image)
        encoding, hidden = self.encoder(cnn_output)  # width, batch, alphabet

        for i in start_end_points.shape(1):
            c_t = self.embed_pos(start_end_points)  # B, SOS/EOS
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

            rnn_output = self.rnn(torch.cat([cnn_output, embedding], dim=-1)) # width, batch, alphabet
        # sigmoids are done in the loss
        return rnn_output

class AttnStrokeSosEos(nn.Module):
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
        self.embed_initial_state = nn.Linear(8, 128)  # absolute/relative start and end positions
        self.cnn = CNN(nc=1, first_conv_op=first_conv_op, cnn_type=cnn_type, first_conv_opts=first_conv_opts)
        self.encoder = nn.LSTM(input_size=1024, hidden_size=128, num_layers=self.encoder_layers)
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4)
        self.decoder = nn.LSTM(input_size=self.context_dim+vocab_size, hidden_size=self.decoder_dim, num_layers=self.decoder_layers)
        self.linear = nn.Linear(self.context_dim+self.decoder_dim, vocab_size)
        self.device = device

    def forward(self, **kwargs):
        if self.training:
            return self._forward(**kwargs)
        else:
            with torch.no_grad():
                return self._forward(**kwargs)

    def _forward(self, start_end_points, image):
        cnn_output = self.cnn(input)
        encoding, hidden = self.encoder(cnn_output)  # width, batch, alphabet
        c_t = self.embed_initial_state(start_end_points) # B, SOS/EOS
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


### Options:
    # Use CNN
    # Crop the feature maps
    # LSTM it, converting ABS start points to cropped image

        # if False:
        #     # CROP IMAGES BASED ON START/ENDPOINTS
        #     # CONVERT START POINT TO AN INDEX - use image height
        #     img_height = 61
        #     start_idx = startpoint * img_height
        #
        #     # DO THE CNN FOR THE WHOLE IMAGE
        #     # TRUNCATE BASED ON ACTIVATIONS


    # Use Attention
        # How do you encode the absolute starting position?
        # just throw it at it...just throw the absolute and the relative position


# Attention:
    # Predict one SOS to EOS - when it signals EOS, start the next one
    # Predict all first strokes
    # Predict all seconds strokes
    # If sequence runs out, decrement batch size
