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
from models.stroke_model import AlexGraves

class AlexGravesCombined(AlexGraves):
    def __init__(self, hidden_size=400,
                 n_layers=3,
                 output_size=121,
                 feature_map_dim=1024, # dim of feature map
                 alphabet_dim=None,
                 cnn_type="default",
                 device="cuda",
                 model_name="combined",
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
        kwargs.update({k:v for k,v in locals().items() if k not in ["kwargs", "self"] and "__" not in k}) # exclude self, __class__, etc.
        super().__init__(**kwargs)

        self.feature_map_dim = feature_map_dim
        self.alphabet_dim = alphabet_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.cnn_type = cnn_type
        self.gt_size = 4 # X,Y,SOS,EOS
        self.device = device
        #self.text_mask = torch.ones(32, 64).to("cuda")

        K = 10 # number of Gaussians in window

        if model_name == "combined":
            letter_embedding_size = self.alphabet_dim + self.gt_size
            self.embedding = nn.Linear(alphabet_dim, feature_map_dim)
            self.lstm_1_letters = nn.LSTM(letter_embedding_size, hidden_size, batch_first=True, dropout=.5)
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

    def init_hidden(self, batch_size, device):
        initial_hidden = [
            (torch.zeros(1, batch_size, self.hidden_size, device=device),
            torch.zeros(1, batch_size, self.hidden_size, device=device))
        ] * self.n_layers # [layer1, layer2 ...] ; layer1 = hidden, cell (1xbatchxdim)
        window_fm = torch.zeros(batch_size, 1, self.feature_map_dim, device=device)
        window_letters = torch.zeros(batch_size, 1, self.alphabet_dim, device=device)
        kappa = torch.zeros(batch_size, 10, 1, device=device)
        return initial_hidden, window_fm, window_letters, kappa


    def compute_window_vector(self, mix_params, prev_kappa, feature_maps, mask, is_map=None):
        """
            kappa: location of window
            alpha: importance of window in mixture
            beta: width of the window
            phi:

        Args:
            mix_params:
            prev_kappa:
            feature_maps:
            mask:
            is_map:

        Returns:

        """
        # Text would have been text LEN x alphabet
        # Image: IMAGE_LEN x 1024ish
        mix_params = torch.exp(mix_params)

        alpha, beta, kappa = mix_params.split(10, dim=1)

        kappa = kappa + prev_kappa
        prev_kappa = kappa

        u = torch.arange(feature_maps.shape[1], dtype=torch.float32, device=feature_maps.device)

        phi = torch.sum(alpha * torch.exp(-beta * (kappa - u).pow(2)), dim=1) # this is the weights defined by a,b,k

        eos = (phi[:, -1] > torch.max(phi[:, :-1])).type(torch.float) # BATCH x 1

        # optimize this?
        phi = (phi * mask).unsqueeze(2) # PHI: BxW     Mask: BxGT.size (4)

        window_vec = torch.sum(phi * feature_maps, dim=1, keepdim=True)
        return window_vec, prev_kappa, eos

    def get_feature_maps(self, img):
        return self.cnn(img).permute(1, 0, 2)  # B x W x 1024

    def forward(
        self,
        inputs, # the shifted GTs
        img,   #
        img_mask, # ignore
        initial_hidden,
        feature_maps=None,
        image_lstm_args=None, # initial_hidden, prev_eos, prev_kappa, prev_window_vec
        letter_lstm_args=None,
        letter_mask=None, #
        letter_gt=None, # one hot
        **kwargs
    ):
        state_1, state_2, state_3 = initial_hidden
        batch_size = inputs.shape[0]
        if feature_maps is None:
            feature_maps = self.get_feature_maps(img)

        # And stuff not NONE!!!
        if (random.random() < .5 and self.mode == "random") or self.mode == "image_only" or self.mode == "combine":
            hid_1, window_vec, all_eos, kappa, state_1, image_lstm_args = self.first_layer(
                initial_hidden=image_lstm_args["initial_hidden"],
                prev_eos=image_lstm_args["prev_eos"],
                prev_kappa=image_lstm_args["prev_kappa"],
                prev_window_vec=image_lstm_args["prev_window_vec"],
                batch_size=batch_size,
                inputs=inputs,
                feature_maps=feature_maps,
                img_mask=img_mask,
                lstm=self.lstm_1,
            )
            inp = torch.cat((inputs, hid_1, window_vec), dim=2) # BATCH x 394? x (1024+LSTM_hidden+gt_size)

        elif self.mode in ["letter_only", "random"]:
            hid_1_L, window_vec_L, all_eos_L, kappa_L, state_1_L, letter_lstm_args = self.first_layer(
                **letter_lstm_args,
                feature_maps=letter_gt,
                batch_size=batch_size,
                inputs=inputs,
                img_mask=letter_mask,
                lstm=self.lstm_1_letters,
                output_embedding=self.embedding
            )
            inp = torch.cat((inputs, hid_1_L, window_vec_L), dim=2)  # BATCH x 394? x (1024+LSTM_hidden+gt_size)
            window_vec = window_vec_L
            hid_1 = hid_1_L
        else:
            raise Exception("Unknown mode")

        # if True:
            #inp = torch.cat((inputs, (hid_1_L+hid_1)/2, window_vec, window_vec_L), dim=2)
            # inp = (inp+inp2)/2

        hid_2, state_2 = self.lstm_2(inp, state_2)
        inp = torch.cat((inputs, hid_2, window_vec), dim=2)
        # inp = torch.cat((inputs, hid_2), dim=2)

        hid_3, state_3 = self.lstm_3(inp, state_3)

        inp = torch.cat([hid_1, hid_2, hid_3], dim=2)
        y_hat = self.output_layer(inp)

        return y_hat, [None, state_2, state_3], image_lstm_args, letter_lstm_args

    def first_layer(self, initial_hidden,
                    prev_eos,
                    prev_window_vec,
                    prev_kappa,
                    batch_size,
                    inputs,
                    feature_maps,
                    img_mask,
                    lstm,
                    output_embedding=None,
                    **kwargs):
        """

        Args:
            initial_hidden: tuple (cell state, hidden state)
            prev_eos:
            prev_window_vec:
            prev_kappa:
            batch_size:
            inputs:
            feature_maps:
            img_mask:
            lstm:
            **kwargs:

        Returns:

        """

        hid_1 = []
        window_vec = []

        state_1 = initial_hidden#(initial_hidden[0][0:1], initial_hidden[1][0:1])

        # Shrink batch as we run out of GTs
        # Use fancy attention instead of window
        # Don't use window at all--just upsample the feature maps--remove for loop & pack to make RNN super fast!
        if prev_eos is None:
            prev_eos = torch.zeros(batch_size)
        all_eos = []
        for t in range(inputs.shape[1]): # loop through width and calculate windows; 1st LSTM no window
            inp = torch.cat((inputs[:, t : t + 1, :], prev_window_vec), dim=2) # BATCH x 1 x (GT_SIZE+1024)
            hid_1_t, state_1 = lstm(inp, state_1) # hid_1_t: BATCH x 1 x HIDDEN
            hid_1.append(hid_1_t)

            mix_params = self.window_layer(hid_1_t)
            window, kappa, eos = self.compute_window_vector(
                mix_params.squeeze(dim=1).unsqueeze(2), # BATCH x 1 x 10*4
                prev_kappa,
                feature_maps, # BATCH x MAX_FM_LEN x (1024)
                img_mask,
            )

            new_eos = prev_eos = torch.max(prev_eos, eos.cpu())
            all_eos.append(new_eos)

            prev_window_vec = window
            prev_kappa = kappa
            if not output_embedding is None:
                embd_window = output_embedding(window)
            else:
                embd_window = window
            window_vec.append(embd_window)

        hid_1 = torch.cat(hid_1, dim=1)
        window_vec = torch.cat(window_vec, dim=1)
        all_eos = torch.cat(all_eos, dim=0).reshape(len(all_eos),-1).permute(1,0)

        # Last steps in everything -- mostly for generating
        state_dict = {"prev_kappa":kappa, "prev_eos":new_eos, "initial_hidden":(state_1), "prev_window_vec":window, "prev_eos": new_eos}
        return hid_1, window_vec, all_eos, kappa, state_1, state_dict


    def generate(
        self,
        feature_maps,
        feature_maps_mask,
        initial_hidden,
        image_lstm_args=None,  # initial_hidden, prev_eos, prev_kappa, prev_window_vec
        letter_lstm_args=None,
        letter_mask=None,  #
        letter_gt=None,  # one hot
        bias=10, # how close to max argument to be
        **kwargs):

        seq_len = 0
        gen_seq = []
        hidden=initial_hidden
        with torch.no_grad():
            batch_size = feature_maps.shape[0]
            #print("batch_size:", batch_size)
            Z = torch.zeros((batch_size, 1, self.gt_size)).to(self.device)
            eos = 0
            while seq_len < 2000 and tensor_sum(eos) < batch_size/2:

                y_hat, hidden, image_lstm_args, letter_lstm_args = self.forward(
                    inputs=Z, # the shifted GTs
                    img=None,   #
                    img_mask=feature_maps_mask, # ignore
                    initial_hidden=hidden,
                    feature_maps=feature_maps,
                    image_lstm_args=image_lstm_args, # initial_hidden, prev_eos, prev_kappa, prev_window_vec
                    letter_lstm_args=letter_lstm_args,
                    letter_mask=letter_mask, #
                    letter_gt=letter_gt, # one hot
                    )

                y_hat = y_hat.squeeze(dim=1)
                Z = model_utils.sample_batch_from_out_dist(y_hat, bias, gt_size=self.gt_size)
                eos, total = 0,0

                # Take average EOS
                if not image_lstm_args["prev_eos"] is None:
                    eos += image_lstm_args["prev_eos"]
                    total+=1
                if not letter_lstm_args["prev_eos"] is None:
                    eos += letter_lstm_args["prev_eos"]
                    total+=1
                if total==2:
                    eos = eos/2
                eos = eos.unsqueeze(1)

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
