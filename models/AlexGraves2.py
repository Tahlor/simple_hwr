import models.model_utils as model_utils
import torch.nn.functional as F
from torch import nn
import torch
from .basic import CNN, BidirectionalRNN
from hwr_utils.utils import is_dalai, no_gpu_testing, tensor_sum
import sys, random
from synthesis.synth_models import models as synth_models

"""
I think the goal was to use TRACE to train an offline synthesizer as a loss???
It didn't work--one trained using distributions might work better
"""

class AlexGraves(synth_models.HandWritingSynthesisNet):
    def __init__(self, hidden_size=400,
                 n_layers=3,
                 output_size=121,
                 feature_map_dim=1024, # dim of feature map
                 cnn_type="default",
                 device="cuda",
                 model_name="default",
                 **kwargs
                 ):
        """

        Args:
            hidden_size:
            n_layers:
            output_size:
            window_size: The dimension of the characters vector OR feature map (feature map: BATCH x Width x 1024)
            cnn_type:
            **kwargs:
        """
        l = locals()
        kwargs.update({k:v for k,v in locals().items() if k not in ["kwargs", "self"] and "__" not in k}) # exclude self, __class__, etc.
        self.__dict__.update(kwargs)
        super().__init__(**kwargs)

        if model_name == "default":
            self.gt_size = 4  # X,Y,SOS,EOS
            self.device = device
            # self.text_mask = torch.ones(32, 64).to("cuda")

            K = 10  # number of Gaussians in window

            self._phi = []
            self.lstm_1 = nn.LSTM(self.gt_size + self.feature_map_dim, hidden_size, batch_first=True, dropout=.5)
            self.lstm_2 = nn.LSTM(
                self.gt_size + self.feature_map_dim + hidden_size, hidden_size, batch_first=True, dropout=.5
            )
            self.lstm_3 = nn.LSTM(
                self.gt_size + self.feature_map_dim + hidden_size, hidden_size, batch_first=True, dropout=.5
            )

            self.window_layer = nn.Linear(hidden_size, 3 * K) # 3: alpha, beta, kappa
            self.output_layer = nn.Linear(n_layers * hidden_size, output_size)

            self.cnn = CNN(nc=1, cnn_type=self.cnn_type) # output dim: Width x Batch x 1024


    def compute_window_vector(self, mix_params, prev_kappa, feature_maps, mask, is_map=None):
        # Text would have been text LEN x alphabet
        # Image: IMAGE_LEN x 1024ish
        mix_params = torch.exp(mix_params)

        alpha, beta, kappa = mix_params.split(10, dim=1)

        kappa = kappa + prev_kappa
        prev_kappa = kappa

        u = torch.arange(feature_maps.shape[1], dtype=torch.float32, device=feature_maps.device)

        phi = torch.sum(alpha * torch.exp(-beta * (kappa - u).pow(2)), dim=1)

        eos = (phi[:, -1] > torch.max(phi[:, :-1])).type(torch.float) # BATCH x 1

        # optimize this?
        phi = (phi * mask).unsqueeze(2) # PHI: BxW Mask: BxGT.size (4)
        if is_map:
            self._phi.append(phi.squeeze(dim=2).unsqueeze(1))

        window_vec = torch.sum(phi * feature_maps, dim=1, keepdim=True)
        return window_vec, prev_kappa, eos

    def get_feature_maps(self, img):
        return self.cnn(img).permute(1, 0, 2)  # B x W x 1024

    def forward(
        self,
        inputs, # the shifted GTs
        img,   #
        img_mask, # ignore
        initial_hidden, # RNN state
        prev_window_vec,
        prev_kappa,
        is_map=False,
        feature_maps=None,
        prev_eos=None,
        **kwargs
    ):
        batch_size = inputs.shape[0]
        if feature_maps is None:
            feature_maps = self.get_feature_maps(img)

        hid_1 = []
        window_vec = []

        state_1 = (initial_hidden[0][0:1], initial_hidden[1][0:1])

        # Shrink batch as we run out of GTs
        # Use fancy attention instead of window
        # Don't use window at all--just upsample the feature maps--remove for loop & pack to make RNN super fast!
        if prev_eos is None:
            prev_eos = torch.zeros(batch_size)
        all_eos = []
        for t in range(inputs.shape[1]): # loop through width and calculate windows; 1st LSTM no window
            inp = torch.cat((inputs[:, t : t + 1, :], prev_window_vec), dim=2) # BATCH x 1 x (GT_SIZE+1024)
            hid_1_t, state_1 = self.lstm_1(inp, state_1) # hid_1_t: BATCH x 1 x HIDDEN
            hid_1.append(hid_1_t)

            mix_params = self.window_layer(hid_1_t)
            window, kappa, eos = self.compute_window_vector(
                mix_params.squeeze(dim=1).unsqueeze(2), # BATCH x 1 x 10*4
                prev_kappa,
                feature_maps, # BATCH x MAX_FM_LEN x (1024)
                img_mask,
                is_map,
            )

            new_eos = prev_eos = torch.max(prev_eos, eos.cpu())
            all_eos.append(new_eos)

            prev_window_vec = window
            prev_kappa = kappa
            window_vec.append(window)

        hid_1 = torch.cat(hid_1, dim=1) # convert list of states to tensor
        window_vec = torch.cat(window_vec, dim=1) # convert list of weighted inputs to tensor
        all_eos = torch.cat(all_eos, dim=0).reshape(len(all_eos),-1).permute(1,0) # eos is 1D, batch size

        inp = torch.cat((inputs, hid_1, window_vec), dim=2) # BATCH x 394? x (1024+LSTM_hidden+gt_size)
        state_2 = (initial_hidden[0][1:2], initial_hidden[1][1:2])

        hid_2, state_2 = self.lstm_2(inp, state_2)
        inp = torch.cat((inputs, hid_2, window_vec), dim=2)
        # inp = torch.cat((inputs, hid_2), dim=2)
        state_3 = (initial_hidden[0][2:], initial_hidden[1][2:])

        hid_3, state_3 = self.lstm_3(inp, state_3)

        inp = torch.cat([hid_1, hid_2, hid_3], dim=2)
        y_hat = self.output_layer(inp)

        return y_hat, [state_1, state_2, state_3], window_vec, prev_kappa, all_eos

    def generate(
        self,
        feature_maps,
        feature_maps_mask,
        hidden,
        window_vector,
        kappa,
        bias=10, # how close to max argument to be
        forced_size=0,
        **kwargs):

        seq_len = 0
        gen_seq = []

        # Keep going condition
        if forced_size:
            condition = lambda seq_len, eos, batch_size: seq_len<forced_size
        else:
            condition = lambda seq_len, eos, batch_size: seq_len < 2000 and tensor_sum(eos) < batch_size/2

        with torch.no_grad():
            batch_size = feature_maps.shape[0]
            #print("batch_size:", batch_size)
            Z = torch.zeros((batch_size, 1, self.gt_size)).to(self.device)
            eos = 0
            while condition(seq_len, eos, batch_size):

                y_hat, state, window_vector, kappa, eos = self.forward(
                    inputs=Z,
                    img=None,
                    feature_maps=feature_maps,
                    img_mask=feature_maps_mask,
                    initial_hidden=hidden,
                    prev_window_vec=window_vector,
                    prev_kappa=kappa,
                    previous_eos=eos
                )

                _hidden = torch.cat([s[0] for s in state], dim=0)
                _cell = torch.cat([s[1] for s in state], dim=0)
                hidden = (_hidden, _cell)

                y_hat = y_hat.squeeze(dim=1)
                Z = model_utils.sample_batch_from_out_dist(y_hat, bias, gt_size=self.gt_size)

                if self.gt_size==4:
                    Z[:, 0:1, 3:4] = eos.unsqueeze(1)
                # if Z.shape[-1] < self.gt_size:
                #     Z = F.pad(input=Z, pad=(0, self.gt_size-Z.shape[-1]), mode='constant', value=0)
                gen_seq.append(Z)
                seq_len += 1

        gen_seq = torch.cat(gen_seq, dim=1)
        gen_seq = gen_seq.cpu().numpy()

        print("seq_len:", seq_len)

        return gen_seq

class AlexGraves2(AlexGraves):
    """ Just do a BRNN instead of a window vector
    """

    def __init__(self, hidden_size=400,
                 n_layers=2,
                 output_size=122,
                 window_size=1024, # dim of feature map
                 cnn_type="default",
                 device="cuda",
                 **kwargs
                 ):
        """

        Args:
            hidden_size:
            n_layers:
            output_size:
            window_size: The dimension of the characters vector OR feature map (feature map: BATCH x Width x 1024)
            cnn_type:
            **kwargs:
        """
        super().__init__(hidden_size=hidden_size, # 400
                         n_layers=n_layers, # 2
                         output_size=output_size, # 20*6+2 (EOS,SOS)
                         window_size=window_size,
                         cnn_type=cnn_type,
                         device=device,
                         model_name="version2",
                         **kwargs)

        # Create model
        hidden_size_factor = 2
        self.brnn1 = BidirectionalRNN(nIn=1024, nHidden=hidden_size, nOut=hidden_size*hidden_size_factor,
                                      dropout=.5, num_layers=n_layers,
                                      rnn_constructor=nn.LSTM,
                                      batch_first=True,
                                      return_states=True)

        self.rnn2 = BidirectionalRNN(nIn=hidden_size*hidden_size_factor+self.gt_size,
                                     nHidden=hidden_size,
                                     nOut=output_size,
                                     dropout=.5,
                                     num_layers=n_layers,
                                     rnn_constructor=nn.LSTM,
                                     bidirectional=False,
                                     batch_first=True,
                                     return_states=True)

    def forward(
        self,
        inputs, # the GTs that start with 0
        img,   #
        img_mask, # ignore
        initial_hidden=None, # RNN state
        prev_window_vec=None,
        prev_kappa=None,
        is_map=False,
        feature_maps=None,
        prev_eos=None,
        lengths=None,
        reset=False,
        **kwargs
    ):
        batch_size = inputs.shape[0]
        if feature_maps is None:
            feature_maps = self.get_feature_maps(img) # B,W,1024

        # Upsample to be the same length as the (lontest) GT-strokepoint-width dimension
        shp = inputs.shape[1]
        feature_maps_upsample = torch.nn.functional.interpolate(feature_maps.permute(0,2,1),
                                                                size=shp,
                                                                mode='linear',
                                                                align_corners=None).permute(0,2,1)

        # Pack it up (pack it in)
        if lengths is not None:
            feature_maps_upsample = torch.nn.utils.rnn.pack_padded_sequence(feature_maps_upsample, lengths, batch_first=True, enforce_sorted=False)
        # print("FM", feature_maps_upsample.shape, feature_maps_upsample.stride())
        # print("IN", inputs.shape)

        if reset:
            brnn_states, rnn_states = None, None
        else:
            brnn_states, rnn_states = initial_hidden

        brnn_output, brnn_states = self.brnn1(feature_maps_upsample, brnn_states) # B, W, hidden
        rnn_input = torch.cat((inputs, brnn_output), dim=2)#.contiguous() # B,W, hidden+4
        rnn_output, rnn_states = self.rnn2(rnn_input, rnn_states) # B, W, hidden

        return rnn_output, [brnn_states, rnn_states], None, None, None
        # RNN states: tuple (hidden state, cell state)
            # # layers, B, Hidden Dim; 2,25,400

        # BRNN states: tuple (size=number of layers)
            # (# layers * 2 bidirectional), B, Hidden Dim; 4,25,400

    def generate(
        self,
        feature_maps,
        feature_maps_mask,
        hidden, # (BRNN_HIDDEN, BRNN_CELL), (RNN_HIDDEN, RNN_CELL)
        window_vector,
        kappa,
        bias=10,  # how close to max argument to be
        gts=None,
        **kwargs):

        seq_len = 0
        gen_seq = []
        hidden = None, None
        with torch.no_grad():
            batch_size = feature_maps.shape[0]
            #print("batch_size:", batch_size)
            Z = torch.zeros((batch_size, 1, self.gt_size)).to(self.device)
            eos = 0
            while seq_len < 400: # and tensor_sum(eos) < batch_size/2:
                y_hat, hidden, window_vector, kappa, _ = self.forward(
                    inputs=Z,
                    img=None,
                    feature_maps=feature_maps,
                    img_mask=feature_maps_mask,
                    initial_hidden=hidden,
                    prev_window_vec=window_vector,
                    prev_kappa=kappa,
                    previous_eos=eos
                )

                y_hat = y_hat.squeeze(dim=1)
                Z = model_utils.sample_batch_from_out_dist2(y_hat, bias, gt_size=self.gt_size)
                eos = Z[:,:,3]
                gen_seq.append(Z)
                seq_len += 1

        gen_seq = torch.cat(gen_seq, dim=1)
        gen_seq = gen_seq.cpu().numpy()

        print("seq_len:", seq_len)

        return gen_seq


    def init_hidden(self, batch_size, device):
        initial_hidden = (
            torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
        )
        window_vector = torch.zeros(batch_size, 1, self.feature_map_dim, device=device)
        kappa = torch.zeros(batch_size, 10, 1, device=device)
        return initial_hidden, window_vector, kappa