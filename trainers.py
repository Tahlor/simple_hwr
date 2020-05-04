#from torchvision.models import resnet
from models.deprecated.deprecated_crnn import *
from torch.autograd import Variable
from hwr_utils import utils
from hwr_utils.stroke_recovery import relativefy_batch_torch, conv_weight, conv_window, PredConvolver
import logging
from loss_module import loss_metrics

logger = logging.getLogger("root."+__name__)

MAX_LENGTH=60

RELU = nn.ReLU()

def to_value(loss_tensor):
    return torch.sum(loss_tensor.cpu(), 0, keepdim=False).item()

class Trainer:
    def __init__(self, model, optimizer, config, loss_criterion=None, **kwargs):
        global SIGMOID
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.loss_criterion = loss_criterion
        self.relative_indices = self.get_indices(config.pred_opts, "cumsum")
        self.sigmoid_indices = self.get_indices(config.pred_opts, "sigmoid")
        self.relu_indices = self.get_indices(config.pred_opts, "relu")
        self.convolve_indices = self.get_indices(config.pred_opts, "convolve") # NOT IMPLEMENTED
        SIGMOID = torch.nn.Sigmoid().to(config.device)

        self.activations = [None] * len(config.pred_opts)
        for i in self.sigmoid_indices:
            self.activations[i] = SIGMOID
        for i in self.relu_indices:
            self.activations[i] = RELU

        if config is None:
            self.logger = utils.setup_logging()
        else:
            self.logger = config.logger

    @staticmethod
    def truncate(preds, label_lengths):
        """ Take in rectangular GT tensor, return as list where each element in batch has been truncated

        Args:
            preds:
            label_lengths:

        Returns:

        """

        preds = [preds[i][:label_lengths[i], :] for i in range(0, len(label_lengths))]
        return preds

    def test(self, item, **kwargs):
        self.model.eval()
        return self.train(item, train=False, **kwargs)

    def eval(self, **kwargs):
        raise NotImplemented

    def train(self, **kwargs):
        raise NotImplemented
    
    @staticmethod
    def get_indices(pred_opts, keyword):
        return [i for i,x in enumerate(pred_opts) if x and keyword.lower() in x.lower()]

class TrainerStrokeRecovery(Trainer):
    def __init__(self, model, optimizer, config, loss_criterion=None):
        super().__init__(model, optimizer, config, loss_criterion)
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.loss_criterion = loss_criterion
        if config is None:
            self.logger = utils.setup_logging()
        else:
            self.logger = config.logger
        self.opts = None
        logger.info(("Relative Idices", self.relative_indices))
        if config.convolve_func == "cumsum":
            self.convolve = None # use relativefy
        else:
            self.convolve = PredConvolver(config.convolve_func, kernel_length=config.cumsum_window_size).convolve

    def default(self, o):
        return None

    @staticmethod
    def truncate(preds, label_lengths):
        """ Return a list

        Args:
            preds:
            label_lengths:

        Returns:

        """

        preds = [preds[i][:label_lengths[i], :] for i in range(0, len(label_lengths))]
        #targs = [targs[i][:label_lengths[i], :] for i in range(0, len(label_lengths))]
        return preds

    def train(self,  item, train=True, **kwargs):
        return self._train(item, train=train, **kwargs)
        # try:
        #     return self._train(item, train=train, **kwargs)
        # except Exception as e:
        #     logger.error(e)
        # return None, None, None

    def _train(self, item, train=True, **kwargs):
        """ Item is the whole thing from the dataloader

        Args:
            loss_fn:
            item:
            train: train/update the model
            **kwargs:

        Returns:

        """
        line_imgs = item["line_imgs"].to(self.config.device)
        label_lengths = item["label_lengths"]
        gt = item["gt_list"]
        suffix = "_train" if train else "_test"

        if train:
            self.model.train()
            self.config.counter.update(epochs=0, instances=line_imgs.shape[0], updates=1)
            #print(self.config.stats[])

        preds = self.eval(line_imgs, self.model, label_lengths=label_lengths, relative_indices=self.relative_indices,
                          device=self.config.device, gt=item["gt"], train=train, convolve=self.convolve)  # This evals and permutes result, Width,Batch,Vocab -> Batch, Width, Vocab

        loss_tensor, loss = self.loss_criterion.main_loss(preds, item, suffix)

        # Update all other stats
        self.update_stats(item, preds, train=train)

        if train:
            self.optimizer.zero_grad()
            loss_tensor.backward()
            torch.nn.utils.clip_grad_norm_(self.config.model.parameters(), 10)
            self.optimizer.step()

        ## Take post activations
        # DO A RELU IF NOT DOING sigmoid later!!!
        if self.sigmoid_indices or self.relu_indices:
            # PREDS ARE A LIST
            for i, p in enumerate(preds):
                preds[i][:, self.sigmoid_indices] = SIGMOID(p[:, self.sigmoid_indices])
                if self.relu_indices:
                    preds[i][:, self.relu_indices] = RELU(p[:, self.relu_indices])

        return loss, preds, None

    def test(self, item, **kwargs):
        self.model.eval()
        return self.train(item, train=False, **kwargs)

    @staticmethod
    def eval(line_imgs, model, label_lengths=None, relative_indices=None, device="cuda",
             gt=None, train=False, convolve=None, sigmoid_activations=None, relu_activations=None):
        """ For offline data, that doesn't have ground truths
        """
        line_imgs = line_imgs.to(device)
        pred_logits = model(line_imgs).cpu()

        new_preds = pred_logits
        preds = new_preds.permute(1, 0, 2) # Width,Batch,Vocab -> Batch, Width, Vocab


        if relative_indices:
            if not train or convolve is None:
                preds = relativefy_batch_torch(preds, reverse=True, indices=relative_indices)  # assume they were in relative positions, convert to absolute
            else:
                preds = convolve(pred_rel=preds, indices=relative_indices, gt=gt)

        ## Shorten - label lengths currently = width of image after CNN
        if not label_lengths is None:
            preds = TrainerStrokeRecovery.truncate(preds, label_lengths) # Convert square torch object to a list, removing predictions related to padding


        # THIS IS A "PRE" ACTIVATION, MUST NOT BE DONE DURING TRAINING!
        if (sigmoid_activations or relu_activations) and not train:
            # PREDS ARE A LIST
            for i, p in enumerate(preds):
                preds[i][:, sigmoid_activations] = SIGMOID(p[:, sigmoid_activations])
                if relu_activations:
                    preds[i][:, relu_activations] = RELU(p[:, relu_activations])


        return preds

    def update_stats(self, item, preds, train=True):
        suffix = "_train" if train else "_test"

        ## If not using L1 loss, report the stat anyway
        # if "l1" not in self.loss_criterion.loss_names:
        #     # Just a generic L1 loss for x,y coords
        #     l1_loss = to_value(self.config.L1.lossfun(preds, item["gt_list"], item["label_lengths"])) # don't divide by batch size
        #     self.config.stats["l1"+suffix].accumulate(l1_loss)

        # Don't do the nearest neighbor search by default
        if (self.config.training_nn_loss and train) or (self.config.test_nn_loss and not train) \
                and self.config.counter.epochs % self.config.test_nn_loss_freq==0:
            self.config.stats["nn"+suffix].accumulate(loss_metrics.calculate_nn_distance(item, preds))


class TrainerStartPoints(Trainer):
    def __init__(self, model, optimizer, config, loss_criterion=None):
        super().__init__(model, optimizer, config, loss_criterion)
        self.opts = None
        self.relative = self.get_relative_indices(config.pred_opts)

    def train(self, item, train=True, **kwargs):
        """ Item is the whole thing from the dataloader

        Args:
            loss_fn:
            item:
            train: train/update the model
            **kwargs:

        Returns:

        """
        line_imgs = item["line_imgs"].to(self.config.device)
        label_lengths = item["label_lengths"]
        gt = item["start_points"]
        suffix = "_train" if train else "_test"

        ## Filter GTs to just the start points and the EOS point; the EOS point will be the finish point of the last stroke
        if train:
            self.model.train()
            self.config.counter.update(epochs=0, instances=line_imgs.shape[0], updates=1)

        preds = self.eval(line_imgs, gt, self.model, label_lengths=label_lengths,
                          device=self.config.device, train=train, relative_indices=self.relative_indices,
                          activation=self.sigmoid_indices)  # This evals and permutes result, Width,Batch,Vocab -> Batch, Width, Vocab

	# Shorten pred to be the desired_num_of_strokes of the ground truth
        pred_list = []
        for i, pred in enumerate(preds):
            pred_list.append(pred[:len(gt[i])])

        loss_tensor, loss = self.loss_criterion.main_loss(preds, item, suffix)

        if train:
            self.optimizer.zero_grad()
            loss_tensor.backward()
            self.optimizer.step()
        return loss, pred_list, None

    @staticmethod
    def eval(line_imgs, gt, model, label_lengths=None, device="cuda", train=False, convolve=None,
             relative_indices=None, activation=None):
        """ For offline data, that doesn't have ground truths
        """
        line_imgs = line_imgs.to(device)
        pred_logits = model(line_imgs, gt).cpu()
        preds = pred_logits.permute(1, 0, 2) # Width,Batch,Vocab -> Batch, Width, Vocab

        if relative_indices:
            preds = relativefy_batch_torch(preds, reverse=True, indices=relative_indices)  # assume they were in relative positions, convert to absolute

        if activation:
            preds[:, :, activation] = SIGMOID(preds[:, :, activation])
        return preds

class TrainerStartEndStroke(Trainer):
    def __init__(self, model, optimizer, config, loss_criterion=None):
        super().__init__(model, optimizer, config, loss_criterion)
        self.opts = None
        self.relative = self.get_relative_indices(config.pred_opts)

    def train(self, item, train=True, **kwargs):
        """ Item is the whole thing from the dataloader

        Args:
            loss_fn:
            item:
            train: train/update the model
            **kwargs:

        Returns:

        """
        line_imgs = item["line_imgs"].to(self.config.device)
        label_lengths = item["label_lengths"]
        gt = item["start_points"]
        start_end_points = item["start_points"] # these are the start and end points
        suffix = "_train" if train else "_test"

        if train:
            self.model.train()
            self.config.counter.update(epochs=0, instances=line_imgs.shape[0], updates=1)

        preds = self.eval(start_end_points, line_imgs, self.model, label_lengths=label_lengths,
                          device=self.config.device, train=train, relative_indices=self.relative_indices,
                          activation=self.sigmoid_indices)  # This evals and permutes result, Width,Batch,Vocab -> Batch, Width, Vocab

	# Shorten pred to be the desired_num_of_strokes of the ground truth
        pred_list = []
        for i, pred in enumerate(preds):
            pred_list.append(pred[:len(gt[i])])

        loss_tensor, loss = self.loss_criterion.main_loss(preds, item, suffix)

        if train:
            self.optimizer.zero_grad()
            loss_tensor.backward()
            self.optimizer.step()
        return loss, pred_list, None

    @staticmethod
    def eval(start_end_points, line_imgs, model, label_lengths=None, device="cuda", train=False, convolve=None,
             relative_indices=None, activation=None):
        """ For offline data, that doesn't have ground truths
        """
        line_imgs = line_imgs.to(device)
        pred_logits = model(line_imgs).cpu()
        preds = pred_logits.permute(1, 0, 2) # Width,Batch,Vocab -> Batch, Width, Vocab

        if relative_indices:
            preds = relativefy_batch_torch(preds, reverse=True, indices=relative_indices)  # assume they were in relative positions, convert to absolute
        if activation:
            preds[:, :, activation] = SIGMOID(preds[:, :, activation])
        return preds



def flatten_params(parameters):
    """
    flattens all parameters into a single column vector. Returns the dictionary to recover them
    :param: parameters: a generator or list of all the parameters
    :return: a dictionary: {"params": [#params, 1],
    "indices": [(start index, end index) for each param] **Note end index in uninclusive**

    """
    l = [torch.flatten(p) for p in parameters]
    indices = []
    s = 0
    for p in l:
        size = p.shape[0]
        indices.append((s, s+size))
        s += size
    flat = torch.cat(l).view(-1, 1)
    return {"params": flat, "indices": indices}


class GeneratorTrainer(Trainer):
    def __init__(self, model, optimizer, config, stroke_model, loss_criterion=None, training_dataset=None, **kwargs):
        super().__init__(model, optimizer, config, loss_criterion)
        self.loss_criterion = loss_criterion
        self.generator_model = model
        self.stroke_model = stroke_model
        self.training_dataset = training_dataset
        self.loss_version = kwargs["loss_type"] if "loss_type" in kwargs else "SM2SM"
        self.device = self.config.device

    def get_strokes(self, img):
        #line_imgs = line_imgs.to(device)
        pred_logits = self.stroke_model(img).cpu()
        return pred_logits.permute(1, 0, 2) # Width,Batch,Vocab -> Batch, Width, Vocab

    def test(self, item, **kwargs):
        self.model.eval()
        image = self.eval(item, train=False, **kwargs)
        loss = self.loss_criterion.main_loss(image, item, suffix="_test", targ_key="gt_list")
        return image

    def eval(self, input, **kwargs):
        image = self.generator_model(input) # different widths
        return image

    def stroke_eval(self, input):
        pred_logits = self.stroke_model(input).cpu()
        return pred_logits.permute(1, 0, 2)

    def train(self, item, train=True, **kwargs):
        gt_strokes = item["gt_list"]
        gt_image = item["line_imgs"]
        pred_image = self.eval(item["gt"].to(self.config.device)) # BATCH x 1 x H x W

        predicted_strokes = None

        ## Truncate images to be compared to GT images
        ## everything from Stroke Model will be: X=relative, Y=abs, SOS
        ## GT LIST is always ABS, ABS, Stroke_Number
        ## make sure you're comparing apples to apples

        if self.loss_version.lower()=="mse":
            loss_tensor, loss = self.loss_criterion.main_loss(pred_image, item, suffix="_train", targ_key="line_imgs")

        elif self.loss_version.lower()=="sm2sm":
            # Compare stroke-model strokes predicted by GT image and synthetic image
            predicted_strokes = self.stroke_eval(pred_image[:, :, 1:62])

            # Create predicted strokes as needed
            #### Convert both sets of Y to be relative

            if item["predicted_strokes_gt"][0] is None:
                if True:
                #with torch.no_grad(): # don't need gradients for predicted GT strokes
                    predicted_strokes_gt_batch = self.stroke_eval(gt_image.to(self.config.device)).detach()

                    ## Adjust GT SOS to Stroke Number
                    #### SOS SHOULD BE STRAIGHT UP COMPARED TO SOS ON THE DTW SINCE BOTH ARE PREDICTED ### ???
                    if True:
                        # GT approximation should be in stroke number format (for now)
                        predicted_strokes_gt_batch = relativefy_batch_torch(predicted_strokes_gt_batch, reverse=True, indices=2)

                        # Needs to be rounded to work correctly
                        predicted_strokes_gt_batch[:,:,2] = predicted_strokes_gt_batch[:,:,2].round()

                for batch_idx, data_idx in enumerate(item["gt_idx"]):
                    self.training_dataset[data_idx]["predicted_strokes_gt"] = predicted_strokes_gt_batch[batch_idx]

                item["predicted_strokes_gt"] = predicted_strokes_gt_batch #.to(self.device)

            loss_tensor, loss = self.loss_criterion.main_loss(predicted_strokes, item, suffix="_train", targ_key="predicted_strokes_gt")

        elif self.loss_version.lower()=="sm2gt":
            # Compare predicted strokes and GT strokes
            predicted_strokes = self.stroke_model(pred_image)
            #self.loss_criterion.loss(predicted_strokes, gt_strokes)
            loss_tensor, loss = self.loss_criterion.main_loss(predicted_strokes, item, suffix="_train", targ_key="gt_list")


        if train:
            self.optimizer.zero_grad()
            loss_tensor.backward()
            torch.nn.utils.clip_grad_norm_(self.config.model.parameters(), 10)
            self.optimizer.step()

        return loss, pred_image, predicted_strokes

