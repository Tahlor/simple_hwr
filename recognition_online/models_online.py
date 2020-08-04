from torch import nn
from hwr_utils.utils import *
from hwr_utils import utils
from hwr_utils.stroke_recovery import relativefy_batch_torch
import logging
from models.basic import BidirectionalRNN, CNN

class OnlineRecognizer(nn.Module):
    """ CRNN with writer classifier
    """
    def __init__(self, nIn, nHidden, vocab_size, dropout=.5, num_layers=3, ):
        super().__init__()

        self.rnn = BidirectionalRNN(nIn=nIn,
                                    nHidden=nHidden,
                                    nOut=vocab_size,
                                    dropout=dropout,
                                    num_layers=num_layers,
                                    rnn_constructor=nn.LSTM)

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
        return self.rnn(input)

