import torch.nn.functional as F
from torch import nn
import torch
from .basic import CNN, BidirectionalRNN
from .CoordConv import CoordConv
from hwr_utils.utils import is_dalai, no_gpu_testing, tensor_sum
import sys, random
sys.path.append("./synthesis")
#from synthesis.synth_models import models
from synthesis.synth_models import models as synth_models
import models.model_utils as model_utils
import models.stroke_model import AlexGraves

class AlexGravesCombined(AlexGraves):
    def __init__(self, hidden_size=400,
                 n_layers=3,
                 output_size=121,
                 window_size=1024, # dim of feature map
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
        super().__init__(hidden_size, n_layers, output_size, window_size)
        self.vocab_size = window_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.cnn_type = cnn_type
        self.gt_size = 4 # X,Y,SOS,EOS
        self.device = device
        #self.text_mask = torch.ones(32, 64).to("cuda")

        K = 10 # number of Gaussians in window

        if model_name == "combined":
            self.lstm_1_letters = nn.LSTM(self.gt_size + self.vocab_size, hidden_size, batch_first=True, dropout=.5)
            self.lstm_1 = nn.LSTM(self.gt_size + self.vocab_size, hidden_size, batch_first=True, dropout=.5)
            self.lstm_2 = nn.LSTM(
                self.gt_size + self.vocab_size + hidden_size, hidden_size, batch_first=True, dropout=.5
            )
            # self.lstm_3 = nn.LSTM(
            #     self.gt_size + hidden_size, hidden_size, batch_first=True
            # )
            self.lstm_3 = nn.LSTM(
                self.gt_size + self.vocab_size + hidden_size, hidden_size, batch_first=True, dropout=.5
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

        hid_1 = torch.cat(hid_1, dim=1)
        window_vec = torch.cat(window_vec, dim=1)
        all_eos = torch.cat(all_eos, dim=0).reshape(len(all_eos),-1).permute(1,0)

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
        **kwargs):

        seq_len = 0
        gen_seq = []
        with torch.no_grad():
            batch_size = feature_maps.shape[0]
            #print("batch_size:", batch_size)
            Z = torch.zeros((batch_size, 1, self.gt_size)).to(self.device)
            eos = 0
            while seq_len < 2000 and tensor_sum(eos) < batch_size/2:

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

    def generate(
        self,
        feature_maps,
        feature_maps_mask,
        hidden,
        window_vector,
        kappa,
        bias=10, # how close to max argument to be
        **kwargs):

        seq_len = 0
        gen_seq = []
        with torch.no_grad():
            batch_size = feature_maps.shape[0]
            #print("batch_size:", batch_size)
            Z = torch.zeros((batch_size, 1, self.gt_size)).to(self.device)
            eos = 0
            while seq_len < 2000 and tensor_sum(eos) < batch_size/2:

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
