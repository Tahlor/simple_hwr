import sys
sys.path.append("..")

import json
from models.deprecated.deprecated_crnn import *
from torch.autograd import Variable
from hwr_utils import utils
import logging
from torch.distributions import bernoulli, uniform

logger = logging.getLogger("root."+__name__)

from hwr_utils import distortions, string_utils

def stable_softmax(X, dim=2):
    max_vec = torch.max(X, dim, keepdim=True)
    exp_X = torch.exp(X - max_vec[0])
    sum_exp_X = torch.sum(exp_X, dim, keepdim=True)
    X_hat = exp_X / sum_exp_X
    return X_hat

class TrainerBaseline(json.JSONEncoder):
    def __init__(self, model, optimizer, config, loss_criterion):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.loss_criterion = loss_criterion
        self.idx_to_char = self.config["idx_to_char"]
        self.train_decoder = string_utils.naive_decode
        self.decoder = config["decoder"]

        if self.config["n_warp_iterations"]:
            print("Using test warp")

    def default(self, o):
        return None

    def train(self, line_imgs, online, labels, label_lengths, gt, retain_graph=False, step=0):
        self.model.train()
        self.config.counter.update(epochs=0, instances=line_imgs.shape[0], updates=1)
        #logger.info(self.config.counter.__dict__)

        pred_tup = self.model(line_imgs, online)
        pred_logits, rnn_input, *_ = pred_tup[0].cpu(), pred_tup[1], pred_tup[2:]

        # Calculate HWR loss
        preds_size = Variable(torch.IntTensor([pred_logits.size(0)] * pred_logits.size(1))) # <- what? isn't this square? why are we tiling the size?

        output_batch = pred_logits.permute(1, 0, 2) # Width,Batch,Vocab -> Batch, Width, Vocab
        pred_strs = list(self.decoder.decode_training(output_batch))

        # Get losses
        logger.debug("Calculating CTC Loss: {}".format(step))

        # pred_logits: max_width x B x alphabet
        # labels: all indices concatenated
        # preds size: B x longest W
        # label_lengths: B x [# of chars]

        loss_recognizer = self.loss_criterion(pred_logits, labels, preds_size, label_lengths)

        # Backprop
        logger.debug("Backpropping: {}".format(step))
        self.optimizer.zero_grad()
        loss_recognizer.backward(retain_graph=retain_graph)
        self.optimizer.step()

        loss = torch.mean(loss_recognizer.cpu(), 0, keepdim=False).item()

        # Error Rate
        self.config["stats"]["HWR_Training_Loss"].accumulate(loss, 1) # Might need to be divided by batch size?
        logger.debug("Calculating Error Rate: {}".format(step))
        err, weight = calculate_cer(pred_strs, gt)

        logger.debug("Accumulating stats")
        self.config["stats"]["Training_Error_Rate"].accumulate(err, weight) # USING MANUAL WEIGHTS HERE FOR TRAINING

        return loss, err, pred_strs

    def test(self, line_imgs, online, gt, force_training=False, nudger=False, validation=True, with_iterations=False):
        if with_iterations:
            self.config.logger.debug("Running test with iterations")
            return self.test_warp(line_imgs, online, gt, force_training, nudger, validation=validation)
        else:
            self.config.logger.debug("Running normal test")
            return self.test_normal(line_imgs, online, gt, force_training, nudger, validation=validation)

    def test_normal(self, line_imgs, online, gt, force_training=False, nudger=False, validation=True):
        """

        Args:
            line_imgs:
            online:
            gt:
            force_training: Run test in .train() as opposed to .eval() mode

        Returns:

        """

        if force_training:
            self.model.train()
        else:
            self.model.eval()

        pred_tup = self.model(line_imgs, online)
        pred_logits, rnn_input, *_ = pred_tup[0].cpu(), pred_tup[1], pred_tup[2:]

        output_batch = pred_logits.permute(1, 0, 2)
        pred_strs = list(self.decoder.decode_test(output_batch))

        # Error Rate
        if nudger:
            return rnn_input
        else:
            err, weight = calculate_cer(pred_strs, gt)
            self.update_test_cer(validation, err, weight)
            loss = -1 # not calculating test loss here
            return loss, err, pred_strs

    def update_test_cer(self, validation, err, weight, prefix=""):
        if validation:
            self.config.logger.debug("Updating validation!")
            stat = self.config["designated_validation_cer"]
            self.config["stats"][f"{prefix}{stat}"].accumulate(err, weight)
        else:
            self.config.logger.debug("Updating test!")
            stat = self.config["designated_test_cer"]
            self.config["stats"][f"{prefix}{stat}"].accumulate(err, weight)
            #print(self.config["designated_test_cer"], self.config["stats"][f"{prefix}{stat}"])

    def test_warp(self, line_imgs, online, gt, force_training=False, nudger=False, validation=True):
        if force_training:
            self.model.train()
        else:
            self.model.eval()

        #use_lm = config['testing_language_model']
        #n_warp_iterations = config['n_warp_iterations']

        compiled_preds = []
        # Loop through identical images
        # batch, repetitions, c/h/w
        for n in range(0, line_imgs.shape[1]):
            imgs = line_imgs[:,n,:,:,:]
            pred_tup = self.model(imgs, online)
            pred_logits, rnn_input, *_ = pred_tup[0].cpu(), pred_tup[1], pred_tup[2:]
            output_batch = pred_logits.permute(1, 0, 2)
            pred_strs = list(self.decoder.decode_test(output_batch))
            compiled_preds.append(pred_strs) # reps, batch

        compiled_preds = np.array(compiled_preds).transpose((1,0)) # batch, reps

        # Loop through batch items
        best_preds = []
        for b in range(0, compiled_preds.shape[0]):
            preds, counts = np.unique(compiled_preds[b], return_counts=True)
            best_pred = preds[np.argmax(counts)]
            best_preds.append(best_pred)

        # Error Rate
        if nudger:
            return rnn_input
        else:
            err, weight = calculate_cer(best_preds, gt)
            self.update_test_cer(validation, err, weight)
            loss = -1 # not calculating test loss here
            return loss, err, pred_strs

class TrainerStrokes(TrainerBaseline):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, line_imgs, strokes, labels, label_lengths, gt, retain_graph=False, step=0):
        self.model.train()
        self.config.counter.update(epochs=0, instances=line_imgs.shape[0], updates=1)

        pred_tup = self.model(line_imgs, strokes)
        pred_logits, rnn_input, *_ = pred_tup[0].cpu(), pred_tup[1], pred_tup[2:]

        # Calculate HWR loss
        preds_size = Variable(torch.IntTensor([pred_logits.size(0)] * pred_logits.size(1))) # <- what? isn't this square? why are we tiling the size?

        output_batch = pred_logits.permute(1, 0, 2) # Width,Batch,Vocab -> Batch, Width, Vocab
        pred_strs = list(self.decoder.decode_training(output_batch))

        # Get losses
        logger.debug("Calculating CTC Loss: {}".format(step))
        loss_recognizer = self.loss_criterion(pred_logits, labels, preds_size, label_lengths)

        # Backprop
        logger.debug("Backpropping: {}".format(step))
        self.optimizer.zero_grad()
        loss_recognizer.backward(retain_graph=retain_graph)
        self.optimizer.step()

        loss = torch.mean(loss_recognizer.cpu(), 0, keepdim=False).item()

        # Error Rate
        self.config["stats"]["HWR_Training_Loss"].accumulate(loss, 1) # Might need to be divided by batch size?
        logger.debug("Calculating Error Rate: {}".format(step))
        err, weight = calculate_cer(pred_strs, gt)

        logger.debug("Accumulating stats")
        self.config["stats"]["Training_Error_Rate"].accumulate(err, weight)

        return loss, err, pred_strs

    def test_normal(self, line_imgs, online, gt, force_training=False, nudger=False, validation=True):
        """

        Args:
            line_imgs:
            online:
            gt:
            force_training: Run test in .train() as opposed to .eval() mode

        Returns:

        """

        if force_training:
            self.model.train()
        else:
            self.model.eval()

        pred_tup = self.model(line_imgs, online)
        pred_logits, rnn_input, *_ = pred_tup[0].cpu(), pred_tup[1], pred_tup[2:]

        output_batch = pred_logits.permute(1, 0, 2)
        pred_strs = list(self.decoder.decode_test(output_batch))

        # Error Rate
        if nudger:
            return rnn_input
        else:
            err, weight = calculate_cer(pred_strs, gt)
            self.update_test_cer(validation, err, weight)
            loss = -1 # not calculating test loss here
            return loss, err, pred_strs

class TrainerStrokesAG(TrainerStrokes):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def sample_batch_from_out_dist(y_hat, bias):
        batch_size = y_hat.shape[0]
        split_sizes = [1] + [20] * 6
        y = torch.split(y_hat, split_sizes, dim=1)

        eos_prob = torch.sigmoid(y[0])
        mixture_weights = stable_softmax(y[1] * (1 + bias), dim=1)
        mu_1 = y[2]
        mu_2 = y[3]
        std_1 = torch.exp(y[4] - bias)
        std_2 = torch.exp(y[5] - bias)
        correlations = torch.tanh(y[6])

        bernoulli_dist = bernoulli.Bernoulli(probs=eos_prob)
        eos_sample = bernoulli_dist.sample()

        K = torch.multinomial(mixture_weights, 1).squeeze()

        mu_k = y_hat.new_zeros((y_hat.shape[0], 2))

        mu_k[:, 0] = mu_1[torch.arange(batch_size), K]
        mu_k[:, 1] = mu_2[torch.arange(batch_size), K]
        cov = y_hat.new_zeros(y_hat.shape[0], 2, 2)
        cov[:, 0, 0] = std_1[torch.arange(batch_size), K].pow(2)
        cov[:, 1, 1] = std_2[torch.arange(batch_size), K].pow(2)
        cov[:, 0, 1], cov[:, 1, 0] = (
            correlations[torch.arange(batch_size), K]
            * std_1[torch.arange(batch_size), K]
            * std_2[torch.arange(batch_size), K],
            correlations[torch.arange(batch_size), K]
            * std_1[torch.arange(batch_size), K]
            * std_2[torch.arange(batch_size), K],
        )

        X = torch.normal(
            mean=torch.zeros(batch_size, 2, 1), std=torch.ones(batch_size, 2, 1)
        ).to(y_hat.device)
        Z = mu_k + torch.matmul(cov, X).squeeze()

        sample = y_hat.new_zeros(batch_size, 1, 3)
        sample[:, 0, 0:1] = eos_sample
        sample[:, 0, 1:] = Z.squeeze()
        return sample

