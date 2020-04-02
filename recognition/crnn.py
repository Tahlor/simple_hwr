#from torchvision.models import resnet
from models.deprecated.CRCR import CRCR
from models.deprecated.deprecated_crnn import *
from torch.autograd import Variable
from models.basic import BidirectionalRNN, CNN
from models.CoordConv import CoordConv
from hwr_utils import utils
from hwr_utils.stroke_recovery import relativefy_batch_torch
import logging

logger = logging.getLogger("root."+__name__)

MAX_LENGTH=60

def to_value(loss_tensor):
    return torch.sum(loss_tensor.cpu(), 0, keepdim=False).item()

class basic_CRNN(nn.Module):
    """ CRNN with writer classifier
    """
    def __init__(self, cnnOutSize, nc, alphabet_size, rnn_hidden_dim, rnn_layers=2, leakyRelu=False,
                 recognizer_dropout=.5, rnn_input_dimension=1024, rnn_constructor=nn.LSTM, cnn_type="default", coord_conv=False):
        super().__init__()
        self.softmax = nn.LogSoftmax()
        self.dropout = recognizer_dropout

        first_conv_op = CoordConv if coord_conv else nn.Conv2d

        if cnn_type in ["default", "intermediates", "default64"] or "resnet" in cnn_type:
            self.cnn = CNN(cnnOutSize, nc, leakyRelu=leakyRelu, cnn_type=cnn_type, first_conv_op=first_conv_op)
        elif cnn_type=="crcr":
            self.cnn = CRCR(cnnOutSize, nc, leakyRelu=leakyRelu, type=cnn_type)
        else:
            raise Exception("Invalid CNN specified")
        self.rnn = BidirectionalRNN(rnn_input_dimension, rnn_hidden_dim, alphabet_size, dropout=recognizer_dropout, num_layers=rnn_layers, rnn_constructor=rnn_constructor)

    def my_eval(self):
        self.rnn.rnn.dropout = 0

    def my_train(self):
        self.rnn.rnn.dropout = self.dropout

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.my_eval()

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
        self.my_train()

    def forward(self, input, online=None, classifier_output=None):
        """

        Args:
            input:
            online:
            classifier_output:

        Returns:
            tuple: normal prediction, refined prediction, normal CNN encoding, nudged CNN encoding

        """
        conv = self.cnn(input)
        rnn_input = conv # [width/time, batch, feature_maps]

        if online is not None:
            rnn_input = torch.cat([rnn_input, online.expand(conv.shape[0], -1, -1)], dim=2)
        recognizer_output = self.rnn(rnn_input)
        return recognizer_output, rnn_input

def create_CRNN(config):
    check_inputs(config)
    # For apples-to-apples comparison, CNN outsize is OUT_SIZE + EMBEDDING_SIZE
    crnn = basic_CRNN(cnnOutSize=config['cnn_out_size'], nc=config['num_of_channels'], alphabet_size=config['alphabet_size'], rnn_hidden_dim=config["rnn_dimension"],
                recognizer_dropout=config["recognizer_dropout"], rnn_input_dimension=config["rnn_input_dimension"], rnn_layers=config["rnn_layers"],
                      rnn_constructor=config["rnn_constructor"], cnn_type=config["cnn"])
    return crnn

def check_inputs(config):
    if not config["style_encoder"] or config["style_encoder"] in ["2StageNudger", "2Stage"]:
        use_writer_classifier = False
        config["embedding_size"] = 0
        config["num_of_writers"] = 0
        config['writer_rnn_output_size'] = 0
        config["embedding_size"] = 0
        config["writer_dropout"] = 0
        config["mlp_layers"] = []

    # Setup RNN input dimension
    config["rnn_input_dimension"] = config["cnn_out_size"] + config["embedding_size"]

    if config["online_augmentation"] and config["online_flag"]:
        config["rnn_input_dimension"] += 1

    if config["rnn_type"].lower() == "gru":
        config["rnn_constructor"]=nn.GRU
    elif config["rnn_type"].lower() == "lstm" or True:
        config["rnn_constructor"]=nn.LSTM
    return config

def create_CRNNClassifier(config, use_writer_classifier=True):
    # Don't use writer classifier
    check_inputs(config)
    crnn = CRNN_with_writer_classifier(rnn_input_dim=config["rnn_input_dimension"], nc=config['num_of_channels'], alphabet_size=config['alphabet_size'], nh=config["rnn_dimension"],
                                       number_of_writers=config["num_of_writers"], writer_rnn_output_size=config['writer_rnn_output_size'],
                                       embedding_size=config["embedding_size"],
                                       writer_dropout=config["writer_dropout"], recognizer_dropout=config["recognizer_dropout"],
                                       writer_rnn_dimension=config["writer_rnn_dimension"],
                                       mlp_layers=config["mlp_layers"], detach_embedding=config["detach_embedding"],
                                       online_augmentation=config["online_augmentation"], use_writer_classifier=use_writer_classifier, rnn_constructor=config["rnn_constructor"])
    return crnn

def create_2Stage(config):
    check_inputs(config)
    crnn = CRNN_2Stage(rnn_input_dim=config["rnn_input_dimension"], nc=config['num_of_channels'], alphabet_size=config['alphabet_size'], rnn_hidden_dim=config["rnn_dimension"],
                       n_rnn=2, leakyRelu=False, recognizer_dropout=config["recognizer_dropout"],
                       online_augmentation=config["online_augmentation"], first_rnn_out_dim=128, rnn_constructor=config["rnn_constructor"])
    return crnn

def create_Nudger(config):
    check_inputs(config)
    crnn = Nudger(rnn_input_dim=config["rnn_input_dimension"], nc=config['num_of_channels'], rnn_hidden_dim=config["rnn_dimension"],
                            rnn_layers=config["nudger_rnn_layers"], leakyRelu=False, rnn_dropout=config["recognizer_dropout"], rnn_constructor=config["rnn_constructor"])
    return crnn


def create_stroke_CRNN(config):
    check_inputs(config)
    # For apples-to-apples comparison, CNN outsize is OUT_SIZE + EMBEDDING_SIZE
    crnn = CRNN_with_strokes(cnnOutSize=config['cnn_out_size'], nc=config['num_of_channels'], alphabet_size=config['alphabet_size'], rnn_hidden_dim=config["rnn_dimension"],
                recognizer_dropout=config["recognizer_dropout"], rnn_input_dimension=config["rnn_input_dimension"], rnn_layers=config["rnn_layers"],
                      rnn_constructor=config["rnn_constructor"], cnn_type=config["cnn"])
    return crnn


class CRNN_with_strokes(nn.Module):
    """ CRNN with writer classifier
    """
    def __init__(self, cnnOutSize, nc, alphabet_size, rnn_hidden_dim, rnn_layers=2, leakyRelu=False,
                 recognizer_dropout=.5, rnn_input_dimension=1024, rnn_constructor=nn.LSTM, cnn_type="default", coord_conv=False):
        super().__init__()
        self.softmax = nn.LogSoftmax()
        self.dropout = recognizer_dropout

        first_conv_op = CoordConv if coord_conv else nn.Conv2d

        if cnn_type in ["default", "intermediates", "default64"] or "resnet" in cnn_type:
            self.cnn = CNN(cnnOutSize, nc, leakyRelu=leakyRelu, cnn_type=cnn_type, first_conv_op=first_conv_op)
        else:
            raise Exception("Invalid CNN specified")
        self.rnn = BidirectionalRNN(rnn_input_dimension, rnn_hidden_dim, alphabet_size, dropout=recognizer_dropout, num_layers=rnn_layers, rnn_constructor=rnn_constructor)

    def my_eval(self):
        self.rnn.rnn.dropout = 0

    def my_train(self):
        self.rnn.rnn.dropout = self.dropout

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.my_eval()

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
        self.my_train()

    def forward(self, input, strokes=None, classifier_output=None):
        """

        Args:
            input:
            online:
            classifier_output:

        Returns:
            tuple: normal prediction, refined prediction, normal CNN encoding, nudged CNN encoding

        """
        conv = self.cnn(input)
        rnn_input = conv # [width/time, batch, feature_maps]

        if strokes is not None:
            rnn_input = torch.cat([rnn_input, strokes.permute(1,0,2)], dim=2)
        recognizer_output = self.rnn(rnn_input)
        return recognizer_output, rnn_input
