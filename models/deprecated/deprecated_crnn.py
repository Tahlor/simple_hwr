from torch import nn
from hwr_utils.utils import *
from models.basic import BidirectionalRNN, CNN

class CRNN(nn.Module):
    """ Original CRNN

    Modified to add some parameters to put it on even ground with the writer-classifier
    """
    def __init__(self, cnnOutSize, nc, alphabet_size, nh, n_rnn=2, leakyRelu=False, recognizer_dropout=.5, rnn_constructor=nn.LSTM):
        super().__init__()

        self.cnn = CNN(cnnOutSize, nc, leakyRelu=leakyRelu)
        self.rnn = BidirectionalRNN(cnnOutSize, nh, alphabet_size, dropout=recognizer_dropout, rnn_constructor=rnn_constructor)
        self.softmax = nn.LogSoftmax()

    def forward(self, input):
        conv = self.cnn(input)
        output = self.rnn(conv)
        return output,

class CRNN_with_writer_classifier(nn.Module):
    """ CRNN with writer classifier
        nh: LSTM dimension
    """

    def __init__(self, rnn_input_dim, nc, alphabet_size, nh, number_of_writers=512, writer_rnn_output_size=128, leakyRelu=False,
                 embedding_size=64, writer_dropout=.5, writer_rnn_dimension=128, mlp_layers=(64, None, 128), recognizer_dropout=.5,
                 detach_embedding=True, online_augmentation=False, use_writer_classifier=True, rnn_constructor=nn.LSTM):
        super().__init__()
        self.cnn = CNN(cnnOutSize=1024, nc=nc, leakyRelu=leakyRelu)
        self.softmax = nn.LogSoftmax()
        self.use_writer_classifier = use_writer_classifier

        self.rnn = BidirectionalRNN(rnn_input_dim, nh, alphabet_size, dropout=recognizer_dropout, rnn_constructor=rnn_constructor)

        if self.use_writer_classifier:
            self.writer_classifier = BidirectionalRNN(rnn_input_dim, writer_rnn_dimension, writer_rnn_output_size, dropout=writer_dropout)
            self.detach_embedding=detach_embedding

            ## Create a MLP on the end to create an embedding
            if "embedding" in mlp_layers:
                embedding_idx = mlp_layers.index("embedding")
                if embedding_idx != get_last_index(mlp_layers, "embedding"):
                    warnings.warn("Multiple dimensions in MLP specified as 'embedding'")
                mlp_layers = [m if m != "embedding" else embedding_size for m in mlp_layers] # replace None with embedding size
            else:
                embedding_idx = None

            self.mlp = MLP(writer_rnn_output_size, number_of_writers, mlp_layers, dropout=writer_dropout, embedding_idx=embedding_idx) # dropout = 0 means no dropout

    def forward(self, input, online=None, classifier_output=None):
        conv = self.cnn(input)

        # Vanilla classifier
        rnn_input = conv

        # concatenate online flag as needed
        if online is not None:
            rnn_input = torch.cat([rnn_input, online.expand(conv.shape[0], -1, -1)], dim=2)

        # concatenate hwr with classifier
        if self.use_writer_classifier:
            classifier_output1 = torch.mean(self.writer_classifier(rnn_input),0,keepdim=False) # RNN dimensional vector
            classifier_output, embedding = self.mlp(classifier_output1, layer="output+embedding") # i.e. 671 dimensional vector

            # Attach style/classifier embedding
            if self.detach_embedding:
                rnn_input = torch.cat([rnn_input, embedding.expand(conv.shape[0], -1, -1).detach()], dim=2) # detach embedding
            else:
                rnn_input = torch.cat([rnn_input, embedding.expand(conv.shape[0], -1, -1)], dim=2)  # keep embedding attached

        # rnn features
        recognizer_output = self.rnn(rnn_input)

        return recognizer_output, classifier_output

class CRNN_2Stage(nn.Module):
    """ CRNN with writer classifier
        nh: LSTM dimension
        nc: number of channels

    """
    def __init__(self, rnn_input_dim, nc, alphabet_size, rnn_hidden_dim, n_rnn=2, leakyRelu=False, recognizer_dropout=.5, online_augmentation=False,
                 first_rnn_out_dim=128, rnn_constructor=nn.LSTM):
        super().__init__()
        self.softmax = nn.LogSoftmax()
        self.cnn = CNN(1024, nc, leakyRelu=leakyRelu)
        self.first_rnn  = BidirectionalRNN(rnn_input_dim, rnn_hidden_dim, first_rnn_out_dim, dropout=recognizer_dropout, rnn_constructor=rnn_constructor)
        self.second_rnn = BidirectionalRNN(rnn_input_dim + first_rnn_out_dim, rnn_hidden_dim, alphabet_size, dropout=recognizer_dropout, rnn_constructor=rnn_constructor)

    def forward(self, input, online=None, classifier_output=None):
        conv = self.cnn(input)
        rnn_input = conv # [width/time, batch, feature_maps]

        if online is not None:
            rnn_input = torch.cat([rnn_input, online.expand(conv.shape[0], -1, -1)], dim=2)

        # First Stage
        first_stage_output = self.first_rnn(rnn_input)

        # Second stage
        cnn_rnn_concat = torch.cat([rnn_input, first_stage_output], dim=2)
        recognizer_output = self.second_rnn(cnn_rnn_concat)

        #print(first_stage_output.shape)
        #print(conv.shape)
        #print(cnn_rnn_concat.shape)

        return recognizer_output, rnn_input


class Nudger(nn.Module):

    def __init__(self, rnn_input_dim, nc, rnn_hidden_dim, rnn_layers=2, rnn_dropout=.5, leakyRelu=False):
        """
        Args:
            final_rnn (nn.Module): The RNN that this output is added to
            rnn_input_dim: Dimension of RNN input - this should already include e.g. augmentation dimension flag
            nc: number of channels in image
            alphabet_size: Number of letters in output alphabet
            rnn_hidden_dim: Dimension of context/state vectors
            rnn_layers: Number of layers in RNN
            leakyRelu:
        """

        super().__init__()
        self.nudger_rnn = BidirectionalRNN(rnn_input_dim, rnn_hidden_dim, rnn_input_dim, dropout=rnn_dropout, num_layers=rnn_layers, rnn_constructor=rnn_constructor)

    def forward(self, feature_maps, recognizer_rnn, classifier_output=None):
        """

        Args:
            feature_maps: The output of the CNN plus additional flags
            recognizer_rnn: The nn.Module that classifies text
            classifier_output:
        Returns:

        """
        # Nudger
        nudger_output = self.nudger_rnn(feature_maps)

        # Second stage
        nudged_cnn_encoding = feature_maps + nudger_output
        recognizer_output_refined = recognizer_rnn(nudged_cnn_encoding)
        return recognizer_output_refined, nudged_cnn_encoding
