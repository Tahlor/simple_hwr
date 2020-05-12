from torch import nn
import torch
from .basic import CNN, BidirectionalRNN
from .CoordConv import CoordConv
from hwr_utils.utils import is_dalai, no_gpu_testing
#from synthesis.synth_models import models
from synthesis.synth_models import models as synth_models

class StrokeRecoveryModel(nn.Module):
    def __init__(self, vocab_size=5, device="cuda", cnn_type="default64", first_conv_op=CoordConv, first_conv_opts=None, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.use_gradient_override = False
        if not "nHidden" in kwargs:
            self.nHidden = 128
        if not "num_layers" in kwargs:
            self.num_layers = 2

        if first_conv_op:
            first_conv_op = CoordConv
        if not no_gpu_testing():
            self.rnn = BidirectionalRNN(nIn=1024, nHidden=self.nHidden, nOut=vocab_size, dropout=.5, num_layers=self.num_layers, rnn_constructor=nn.LSTM)
            self.cnn = CNN(nc=1, first_conv_op=first_conv_op, cnn_type=cnn_type, first_conv_opts=first_conv_opts)
        else:
            self.rnn = BidirectionalRNN(nIn=64, nHidden=1, nOut=vocab_size, dropout=.5, num_layers=1,
                                        rnn_constructor=nn.LSTM)
            self.cnn = fake_cnn
            self.cnn.cnn_type = "FAKE"
            print("DALAi!!!!")

    def forward(self, input):
        if self.training or self.use_gradient_override:
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

class AlexGraves():
    def __init__(self, vocab_size=5,
                 device="cuda",
                 cnn_type="default64",
                 first_conv_op=CoordConv,
                 first_conv_opts=None, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

        model = models.HandWritingSynthesisNet(hidden_size=400,
                                        n_layers=3,
                                        output_size=121,
                                        window_size=vocab_size.vocab_size)

class Synthesis_with_CNN(synth_models.HandWritingSynthesisNet):
    def __init__(self, hidden_size=400, n_layers=3, output_size=121, window_size=77):
        super().__init__(hidden_size, n_layers, output_size, window_size)
        self.vocab_size = window_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.text_mask = torch.ones(32, 64).to("cuda")

        K = 10
        self.EOS = False
        self._phi = []

        self.lstm_1 = nn.LSTM(3 + self.vocab_size, hidden_size, batch_first=True)
        self.lstm_2 = nn.LSTM(
            3 + self.vocab_size + hidden_size, hidden_size, batch_first=True
        )
        # self.lstm_3 = nn.LSTM(
        #     3 + hidden_size, hidden_size, batch_first=True
        # )
        self.lstm_3 = nn.LSTM(
            3 + self.vocab_size + hidden_size, hidden_size, batch_first=True
        )

        self.window_layer = nn.Linear(hidden_size, 3 * K)
        self.output_layer = nn.Linear(n_layers * hidden_size, output_size)

        self.cnn = CNN(nc=1, cnn_type="default64")

        # self.init_weight()

    def compute_window_vector(self, mix_params, prev_kappa, feature_maps, mask, is_map=None):
        # Text would have been text LEN x alphabet
        # Image: IMAGE_LEN x 1024ish
        mix_params = torch.exp(mix_params)

        alpha, beta, kappa = mix_params.split(10, dim=1)

        kappa = kappa + prev_kappa
        prev_kappa = kappa

        u = torch.arange(feature_maps.shape[1], dtype=torch.float32, device=feature_maps.device)

        phi = torch.sum(alpha * torch.exp(-beta * (kappa - u).pow(2)), dim=1)
        if phi[0, -1] > torch.max(phi[0, :-1]):
            self.EOS = True

        # optimize this?
        phi = (phi * mask).unsqueeze(2)
        if is_map:
            self._phi.append(phi.squeeze(dim=2).unsqueeze(1))

        window_vec = torch.sum(phi * feature_maps, dim=1, keepdim=True)
        return window_vec, prev_kappa

    def forward(
        self,
        inputs, # the shifted GTs
        img,   #
        text_mask, # ignore
        initial_hidden, # RNN state
        prev_window_vec,
        prev_kappa,
        is_map=False,
    ):

        feature_maps = self.cnn(img)
        hid_1 = []
        window_vec = []

        state_1 = (initial_hidden[0][0:1], initial_hidden[1][0:1])

        for t in range(inputs.shape[1]):
            inp = torch.cat((inputs[:, t : t + 1, :], prev_window_vec), dim=2)

            hid_1_t, state_1 = self.lstm_1(inp, state_1)
            hid_1.append(hid_1_t)

            mix_params = self.window_layer(hid_1_t)
            window, kappa = self.compute_window_vector(
                mix_params.squeeze(dim=1).unsqueeze(2),
                prev_kappa,
                feature_maps,
                text_mask,
                is_map,
            )

            prev_window_vec = window
            prev_kappa = kappa
            window_vec.append(window)

        hid_1 = torch.cat(hid_1, dim=1)
        window_vec = torch.cat(window_vec, dim=1)

        inp = torch.cat((inputs, hid_1, window_vec), dim=2)
        state_2 = (initial_hidden[0][1:2], initial_hidden[1][1:2])

        hid_2, state_2 = self.lstm_2(inp, state_2)
        inp = torch.cat((inputs, hid_2, window_vec), dim=2)
        # inp = torch.cat((inputs, hid_2), dim=2)
        state_3 = (initial_hidden[0][2:], initial_hidden[1][2:])

        hid_3, state_3 = self.lstm_3(inp, state_3)

        inp = torch.cat([hid_1, hid_2, hid_3], dim=2)
        y_hat = self.output_layer(inp)

        return y_hat, [state_1, state_2, state_3], window_vec, prev_kappa

    def generate(
        self,
        inp,
        text,
        text_mask,
        prime_text,
        prime_mask,
        hidden,
        window_vector,
        kappa,
        bias,
        is_map=False,
        prime=False,
    ):
        seq_len = 0
        gen_seq = []
        with torch.no_grad():
            batch_size = inp.shape[0]
            print("batch_size:", batch_size)
            if prime:
                y_hat, state, window_vector, kappa = self.forward(
                    inp, prime_text, prime_mask, hidden, window_vector, kappa, is_map
                )

                _hidden = torch.cat([s[0] for s in state], dim=0)
                _cell = torch.cat([s[1] for s in state], dim=0)
                # last time step hidden state
                hidden = (_hidden, _cell)
                # # last time step window vector
                # window_vector = window_vector[:, -1:, :]
                # # last time step output vector
                # y_hat = y_hat[:, -1, :]
                # # y_hat = y_hat.squeeze()
                # Z = sample_from_out_dist(y_hat, bias)
                # inp = Z
                # gen_seq.append(Z)
                self.EOS = False
                inp = inp.new_zeros(batch_size, 1, 3)
                _, window_vector, kappa = self.init_hidden(batch_size, inp.device)

            while not self.EOS and seq_len < 2000:
                y_hat, state, window_vector, kappa = self.forward(
                    inp, text, text_mask, hidden, window_vector, kappa, is_map
                )

                _hidden = torch.cat([s[0] for s in state], dim=0)
                _cell = torch.cat([s[1] for s in state], dim=0)
                hidden = (_hidden, _cell)
                # for batch sampling
                # y_hat = y_hat.squeeze(dim=1)
                # Z = sample_batch_from_out_dist(y_hat, bias)
                y_hat = y_hat.squeeze()
                Z = synth_models.sample_from_out_dist(y_hat, bias)
                inp = Z
                gen_seq.append(Z)

                seq_len += 1

        gen_seq = torch.cat(gen_seq, dim=1)
        gen_seq = gen_seq.cpu().numpy()

        print("EOS:", self.EOS)
        print("seq_len:", seq_len)

        return gen_seq


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
