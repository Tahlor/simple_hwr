import torch.nn.functional as F
from torch import nn
import torch
from .basic import CNN, BidirectionalRNN
from .CoordConv import CoordConv
from hwr_utils.utils import is_dalai, no_gpu_testing, tensor_sum
import sys, random
sys.path.append("./synthesis")
#from synthesis.synth_models import models


class StrokeRecoveryModel(nn.Module):
    def __init__(self, vocab_size=4, device="cuda", cnn_type="default64", first_conv_op=CoordConv, first_conv_opts=None, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.use_gradient_override = False
        if not "nHidden" in kwargs:
            self.nHidden = 128
        if not "num_layers" in kwargs:
            self.num_layers = 2
        print("Layers", self.num_layers, "nHidden", self.nHidden)
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

    def forward(self, input, lengths, **kwargs):
        if self.training or self.use_gradient_override:
            #print("TRAINING MODE")
            return self._forward(input, lengths, **kwargs)
        else:
            with torch.no_grad():
                #print("EVAL MODE")
                return self._forward(input, lengths, **kwargs)

    def _forward(self, input, lengths=None, **kwargs):
        cnn_output = self.cnn(input) # W, B, 1024
        # w,b,d = cnn_output.shape
        # width_positional = torch.arange(w).repeat(1, w, 1) / 60
        # sine_width_positional = torch.arange(w).repeat(1, w, 1) / 60

        if lengths is not None and False:
            cnn_output = torch.nn.utils.rnn.pack_padded_sequence(cnn_output, lengths, batch_first=False, enforce_sorted=False)

        rnn_output = self.rnn(cnn_output,) # width, batch, alphabet

        # if lengths is not None:
        #     rnn_output, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=False, enforce_sorted=False)

        # sigmoids are done in the loss
        return rnn_output # W x B x GTshape


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

    def _forward(self, start_end_points, image, **kwargs):
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

class TMinus1(nn.Module):
    def __init__(self, vocab_size=4, device="cuda", cnn_type="default64", first_conv_op=CoordConv,
                 first_conv_opts=None,
                 rnn_mode="all_zeros",   # all_zeros - super fast; correct_gts - super fast; "NA" - use hte previous one?
                 mode="random", #"random"ly apply kd; always_fix=always apply kd; never_fix
                 **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.use_gradient_override = False
        if not "nHidden" in kwargs:
            self.nHidden = 256
        if not "num_layers" in kwargs:
            self.num_layers = 2

        self.vocab_size = vocab_size
        self.n_layers = self.num_layers
        self.cnn_type = cnn_type
        self.gt_size = 4 # X,Y,SOS,EOS
        self.device = device
        #self.text_mask = torch.ones(32, 64).to("cuda")

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

        self.cnn = CNN(nc=1, cnn_type=self.cnn_type) # output dim: Width x Batch x 1024

        self.mode = mode
        self.rnn_mode = rnn_mode
        if rnn_mode in ("all_zeros", "correct_gts"):
            self.forward = self.forward_fast
        else:
            self.forward = self.forward_main

        # Create model
        hidden_size_factor = 2
        self.brnn1 = BidirectionalRNN(nIn=1024, nHidden=self.nHidden, nOut=self.nHidden*hidden_size_factor,
                                      dropout=.5, num_layers=self.n_layers,
                                      rnn_constructor=nn.LSTM,
                                      batch_first=True,
                                      return_states=True)

        self.rnn2 = BidirectionalRNN(nIn=self.nHidden*hidden_size_factor+self.gt_size,
                                     nHidden=self.nHidden,
                                     nOut=self.gt_size,
                                     dropout=.5,
                                     num_layers=self.n_layers,
                                     rnn_constructor=nn.LSTM,
                                     bidirectional=False,
                                     batch_first=True,
                                     return_states=True)

    def get_feature_maps(self, img):
        return self.cnn(img).permute(1, 0, 2)  # B x W x 1024

    def forward_fast(
        self,
        img,
        label_lengths=None,
        item=None,
        inputs=None,
        feature_maps=None,
        initial_hidden=None,
        lengths=None,
        reset=False,
        mode="random", # always_fix, never_fix
        **kwargs
    ):
        """ GTS can be injected here, fast

        Args:
            inputs:
            img:
            feature_maps:
            initial_hidden:
            lengths:
            reset:
            **kwargs:

        Returns:

        """
        #inputs = item["gt"].to(self.device)
        if inputs is None:
            if self.rnn_mode=="all_zeros":
                inputs = torch.zeros(item["rel_gt"][:, :-1].shape).to(self.device)
            elif self.rnn_mode=="correct_gts":
                inputs = item["rel_gt"][:, :-1].to(self.device)
            else:
                raise NotImplemented

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

        if reset or initial_hidden is None:
            brnn_states, rnn_states = None, None
        else:
            brnn_states, rnn_states = initial_hidden

        brnn_output, brnn_states = self.brnn1(feature_maps_upsample, brnn_states) # B, W, hidden
        rnn_input = torch.cat((inputs, brnn_output), dim=2)#.contiguous() # B,W, hidden+4
        rnn_output, rnn_states = self.rnn2(rnn_input, rnn_states) # B, W, hidden

        return rnn_output

    def forward_main(
        self,
        img,
        label_lengths=None,
        item=None,
        inputs=None,
        feature_maps=None,
        initial_hidden=None,
        lengths=None,
        reset=False,
        **kwargs
    ):
        """ Manual calculation for each step

        Args:
            item:
            inputs:
            img:
            feature_maps:
            initial_hidden:
            lengths:
            reset:
            mode: always_fix, never_fix
            **kwargs:

        Returns:

        """

        if feature_maps is None:
            feature_maps = self.get_feature_maps(img) # B,W,1024

        inputs = item["gt"]
        # Upsample to be the same length as the (longest) GT-strokepoint-width dimension
        shp = inputs.shape[1]
        feature_maps_upsample = torch.nn.functional.interpolate(feature_maps.permute(0,2,1), # put width last, interpolate, put width back
                                                                size=shp,
                                                                mode='linear',
                                                                align_corners=None).permute(0,2,1)

        # Pack it up (pack it in)
        if lengths is not None:
            feature_maps_upsample = torch.nn.utils.rnn.pack_padded_sequence(feature_maps_upsample, lengths, batch_first=True, enforce_sorted=False)

        if reset or initial_hidden is None:
            brnn_states, rnn_states = None, None
        else:
            brnn_states, rnn_states = initial_hidden

        # Put feature maps through a B-LSTM
        brnn_output, brnn_states = self.brnn1(feature_maps_upsample, brnn_states) # B, W, hidden

        _input = torch.zeros(inputs[:,0,:].shape).to(self.device) # 28x4
        output = []
        curr_locations = _input[:,:2] # B x 2 (x,y)

        # Preds should all be relative!
        for t in range(feature_maps_upsample.shape[1]): # B, W, 1024
            rnn_input = torch.cat((_input, brnn_output[:,t]), dim=1).unsqueeze(1 )#.contiguous() # B,1,hidden+4
            rnn_output, rnn_states = self.rnn2(rnn_input, rnn_states) # B, W, hidden

            ## Y IS ABSOLUTE
            ##
            if (self.mode=="random" and random.random() < .5) or self.mode=="always_fix":
                # RNN output needs to be added to running sum
                # gts need to be in absolute coordinates
                new_points = (rnn_output.squeeze(1)[:,:2]+curr_locations).detach().cpu() # BATCH X 4

                # Query GTs, find best prediction
                # Only do it for 0 through label_length
                # These are nearest indices in GT, -1 => last point in GT
                idx = [kd.query(new_points[i,:2])[1] if t < item["label_lengths"][i] else -1 for i,kd in enumerate(item["kdtree"])]
                i = range(0, len(idx)), idx # batch_idx, GT_match_idx
                best_gts = item["gt"][i][:,:2].to(self.device) # Bx2; these need to be ABS

                # Calculate delta -- vector to push prediction to nearest GT
                # delta = abs_position - (current_abs_position + pred_relative_movement)
                delta = best_gts - (curr_locations + rnn_output[:, -1, :2].detach()) # vector to push
                rnn_output[:, -1, :2] += delta #.to(self.device) # nudges prediction to GT

                # Curr location is just this best position, feed into next iteration
                curr_locations = best_gts
                output.append(rnn_output)
                _input = rnn_output.detach().squeeze(1) # WHAT IS BEING SQUEEZED HERE?
            else:
                curr_locations += rnn_output[:, -1, :2].detach()
                output.append(rnn_output)

        rnn_output = torch.cat(output, axis=1)
        return rnn_output.permute(1, 0, 2) # Convert to W, B, Vocab #, [brnn_states, rnn_states]

        # RNN states: tuple (hidden state, cell state)
            # # layers, B, Hidden Dim; 2,25,400

        # BRNN states: tuple (size=number of layers)
            # (# layers * 2 bidirectional), B, Hidden Dim; 4,25,400






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

# class AlexGraves():
#     def __init__(self, feature_map_dim=5,
#                  device="cuda",
#                  cnn_type="default64",
#                  first_conv_op=CoordConv,
#                  first_conv_opts=None, **kwargs):
#         super().__init__()
#         self.__dict__.update(kwargs)
#
#         model = Synthesis_with_CNN(hidden_size=400,
#                                         n_layers=3,
#                                         output_size=121,
#                                         window_size=feature_map_dim)
