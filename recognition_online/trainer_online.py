#from torchvision.models import resnet
from models.deprecated.deprecated_crnn import *
from torch.autograd import Variable
from hwr_utils import utils
from hwr_utils.stroke_recovery import relativefy_batch_torch, conv_weight, conv_window, PredConvolver
from hwr_utils.stroke_dataset import img_width_to_pred_mapping
import logging
from loss_module import loss_metrics
from loss_module import losses
import sys
sys.path.append("..")
logger = logging.getLogger("root."+__name__)
from trainers import Trainer

MAX_LENGTH=60

RELU = nn.ReLU()

def to_value(loss_tensor):
    return torch.sum(loss_tensor.cpu(), 0, keepdim=False).item()

class TrainerOnlineRecognition(Trainer):
    def __init__(self, model, optimizer, config, loss_criterion=None):
        super().__init__(model, optimizer, config, loss_criterion)
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.loss_criterion = loss_criterion
        self.decoder = config.decoder
        self.device = self.config.device
        if config is None:
            self.logger = utils.setup_logging()
        else:
            self.logger = config.logger
        self.opts = None

    def default(self, o):
        return None

    def train(self, item, train=True, **kwargs):
        return self._train(item, train=train, **kwargs)

    def _train(self, item, train=True, **kwargs):
        """ Item is the whole thing from the dataloader

        Args:
            loss_fn:
            item:
            train: train/update the model
            **kwargs:

        Returns:

        """
        item = edict(item)
        online_strokes = item["rel_gt"] # not "strokes"
        if online_strokes.shape[2] > 3: # batch, width, [x,y,sos,eos]
            online_strokes = online_strokes[:,:,:3]
        online_strokes = online_strokes.to(self.device)
        if train:
            self.model.train()
            self.config.counter.update(epochs=0,
                                       instances=online_strokes.shape[0],
                                       updates=1)

        pred_strs, loss_recognizer, loss, err, weight = self.eval(online_strokes,
                          self.model,
                          label_lengths=item.label_lengths,
                          device=self.config.device,
                          gt=item["gt_text"],
                          train=train,
                          item=item)  # This evals and permutes result, Width,Batch,Vocab -> Batch, Width, Vocab

        if train:
            self.optimizer.zero_grad()
            loss_recognizer.backward()
            torch.nn.utils.clip_grad_norm_(self.config.model.parameters(), 10)
            self.optimizer.step()

        # Update all other stats
        self.update_stats(loss, err, weight, kind="Training" if train else "Test")

        return loss, None, pred_strs

    def test(self, item, **kwargs):
        self.model.eval()
        return self.train(item, train=False, **kwargs)

    def eval(self, strokes, model, label_lengths=None, device="cuda",
             gt=None, train=False, sigmoid_activations=None, relu_activations=None,
             truncate=0, item=None):
        """

        Args:
            strokes:
            model:
            label_lengths:
            device:
            gt:
            train:
            sigmoid_activations:
            relu_activations:
            truncate:
            item:

        Returns:

        """
        # This comes out BATCH,WIDTH,VOCAB
        pred_logits = output_batch = model(strokes)

        preds_size = Variable(torch.IntTensor([pred_logits.size(1)] * pred_logits.size(0))) # <- what? isn't this square? why are we tiling the size?

        #output_batch = pred_logits.permute(1, 0, 2) # Width,Batch,Vocab -> Batch, Width, Vocab
        pred_strs = list(self.decoder.decode_training(output_batch))

        # pred_logits: max_width x B x alphabet
        # item.gt_text_indices (labels): all indices concatenated
        # preds size: B x longest W
        # label_lengths: B x [# of chars]

        # Loss criterion will take the log probs
        loss_recognizer = self.loss_criterion(pred_logits.permute(1, 0, 2), # CTC requires length, batch, vocab
                                              torch.tensor(item.concatenated_gt_text_indices, requires_grad=False),
                                              preds_size,
                                              torch.tensor(item.label_lengths, requires_grad=False))
        loss = torch.mean(loss_recognizer.cpu(), 0, keepdim=False).item()

        # Error Rate
        err, weight = calculate_cer(pred_strs, gt)

        return pred_strs, loss_recognizer, loss, err, weight

    def update_stats(self, loss, err, weight, kind="Training"):
        self.config["stats"][f"HWR_{kind}_Loss"].accumulate(loss, 1) # Might need to be divided by batch size?
        self.config["stats"][f"{kind}_Error_Rate"].accumulate(err, weight)

if __name__=='__main__':
    pass