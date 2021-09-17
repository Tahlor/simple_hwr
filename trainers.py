#from torchvision.models import resnet
from models.deprecated.deprecated_crnn import *
from torch.autograd import Variable
from hwr_utils import utils
from hwr_utils.stroke_recovery import relativefy_batch_torch, conv_weight, conv_window, PredConvolver
from hwr_utils.stroke_dataset import img_width_to_pred_mapping
import logging
from loss_module import loss_metrics
from loss_module import losses

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
        SIGMOID = torch.nn.Sigmoid().to(config.device)

        if "pred_opts" in config:
            self.relative_indices = self.get_indices(config.pred_opts, "cumsum")
            self.sigmoid_indices = self.get_indices(config.pred_opts, "sigmoid")
            self.relu_indices = self.get_indices(config.pred_opts, "relu")
            self.convolve_indices = self.get_indices(config.pred_opts, "convolve") # NOT IMPLEMENTED

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
    def _truncate(preds, label_lengths, window=0):
        """ Take in rectangular GT tensor, return as list where each element in batch has been truncated

        Args:
            preds:
            label_lengths:

        Returns:

        """

        preds = [preds[i][:label_lengths[i]+window, :] for i in range(0, len(label_lengths))]
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


class TrainerStrokeRecovery(Trainer):
    def __init__(self, model, optimizer, config, loss_criterion=None):
        super().__init__(model, optimizer, config, loss_criterion)
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.truncate = config.truncate
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
                          device=self.config.device, gt=item["gt"], train=train, convolve=self.convolve,
                          truncate=self.truncate, item=item)  # This evals and permutes result, Width,Batch,Vocab -> Batch, Width, Vocab

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

        # if it has EOS
        if preds[0].shape[-1] == 4:
            for i in range(len(preds)):
                eos = np.argmax(preds[i][:,3]>.5)
                if eos >= 300:
                    preds[i] = preds[i][:eos+1]

        return loss, preds, None

    def test(self, item, **kwargs):
        self.model.eval()
        return self.train(item, train=False, **kwargs)

    @staticmethod
    def eval(line_imgs, model, label_lengths=None, relative_indices=None, device="cuda",
             gt=None, train=False, convolve=None, sigmoid_activations=None, relu_activations=None,
             truncate=0, item=None):
        """ For offline data, that doesn't have ground truths
        """
        line_imgs = line_imgs.to(device)
        pred_logits = model(line_imgs, label_lengths, item=item).cpu()

        new_preds = pred_logits
        preds = new_preds.permute(1, 0, 2) # Width,Batch,Vocab -> Batch, Width, Vocab


        if relative_indices:
            if not train or convolve is None:
                preds = relativefy_batch_torch(preds, reverse=True, indices=relative_indices)  # assume they were in relative positions, convert to absolute
            else:
                preds = convolve(pred_rel=preds, indices=relative_indices, gt=gt)

        ## Shorten - label lengths currently = width of image after CNN
        truncate_window = 0 if truncate else 20
        if not label_lengths is None: #and truncate_window >= 0:
            # Convert square torch object to a list, removing predictions related to padding
            # Add a buffer of 20, so that each pred goes 20 past the EOS
            preds = TrainerStrokeRecovery._truncate(preds, label_lengths, window=truncate_window)

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

        self.stroke_model.training = True # turn on gradients
        self.stroke_model.use_gradient_override = True
        self.stroke_model.eval() # eval() do dropout etc. as needed

        self.m = self.stroke_model.cnn.cnn.conv1.weight.T.clone()

        self.training_dataset = training_dataset
        self.loss_version = kwargs["loss_type"] if "loss_type" in kwargs else "SM2SM"
        self.device = self.config.device

        self.white_bias = losses.BiasLoss(loss_indices=None).lossfun

    def get_strokes(self, img):
        #line_imgs = line_imgs.to(device)
        pred_logits = self.stroke_model(img).cpu()
        return pred_logits.permute(1, 0, 2) # Width,Batch,Vocab -> Batch, Width, Vocab

    def test(self, item, **kwargs):
        self.model.eval()
        return self.train(item, train=False, **kwargs)

    def eval(self, input, **kwargs):
        image = self.generator_model(input)
        return image

    def stroke_eval(self, input, **kwargs):
        pred_logits = self.stroke_model(input).cpu().permute(1, 0, 2)
        return pred_logits

    def sm2sm(self, item, pred_image, gt_image):
        # Compare stroke-model strokes predicted by GT image and synthetic image
        self.stroke_model.train()
        #white_loss_tensor = 0
        white_loss_tensor = self.white_bias(pred_image, targs=1, label_lengths=None) * .01 # bias toward whiteness

        label_lengths = item["label_lengths"]
        predicted_strokes = self.stroke_eval(pred_image[:, :, :], item=item)
        predicted_strokes = relativefy_batch_torch(predicted_strokes, reverse=True, indices=0) # sum the x-axis

        # Manual truncation
        predicted_strokes = self._truncate(predicted_strokes, label_lengths, window=0)
        # Create predicted strokes as needed
        #### Convert both sets of Y to be relative

        if item["predicted_strokes_gt"][0] is None:
            self.stroke_model.eval()
            with torch.no_grad(): # don't need gradients for predicted GT strokes
                predicted_strokes_gt_batch = self.stroke_eval(gt_image.to(self.config.device), item=item).detach()
                predicted_strokes_gt_batch = relativefy_batch_torch(predicted_strokes_gt_batch, reverse=True, indices=0)  # sum the x-axis
                predicted_strokes_gt_batch[:,:,self.sigmoid_indices] = SIGMOID(predicted_strokes_gt_batch[:,:,self.sigmoid_indices])
                ## Adjust GT SOS to Stroke Number
                #### SOS SHOULD BE STRAIGHT UP COMPARED TO SOS ON THE DTW SINCE BOTH ARE PREDICTED ### ???
                if True:
                    # Needs to be rounded to work correctly - since new strokes are determined by not equalling previous
                    # This logic can be updated
                    predicted_strokes_gt_batch[:, :, 2] = predicted_strokes_gt_batch[:, :, 2].round()

                    # GT approximation should be in stroke number format (for now)
                    predicted_strokes_gt_batch = relativefy_batch_torch(predicted_strokes_gt_batch, reverse=True,
                                                                        indices=2)


            # Truncate
            predicted_strokes_gt_batch = self._truncate(predicted_strokes_gt_batch, label_lengths, window=0)
            for batch_idx, data_idx in enumerate(item["gt_idx"]):
                self.training_dataset[data_idx]["predicted_strokes_gt"] = predicted_strokes_gt_batch[batch_idx]

            item["predicted_strokes_gt"] = predicted_strokes_gt_batch  # .to(self.device)

        loss_tensor, loss = self.loss_criterion.main_loss(predicted_strokes, item, suffix="_train",
                                                          targ_key="predicted_strokes_gt")

        # Make sure stroke model isn't training
        #assert torch.all(torch.eq(self.m, self.stroke_model.cnn.cnn.conv1.weight.T))
        return loss_tensor+white_loss_tensor.cpu(), loss, predicted_strokes


    def train(self, item, train=True, **kwargs):
        if train:
            self.model.train()
            suffix="_train"
        else:
            self.model.eval()
            suffix="_test"
        gt_strokes = item["gt_list"]
        gt_image = item["line_imgs"]
        # Truncate the pred image to be the size of the original (in square format)
        pred_image = self.eval(item["rel_gt"].to(self.config.device))[:,:,:,:gt_image.shape[-1]] # BATCH x 1 x H x W
        self.config.counter.update(epochs=0, instances=gt_image.shape[0], updates=1)

        predicted_strokes = None

        ## Truncate images to be compared to GT images
        ## everything from Stroke Model will be: X=relative, Y=abs, SOS
        ## GT LIST is always ABS, ABS, Stroke_Number
        ## make sure you're comparing apples to apples

        if self.loss_version.lower()=="mse":
            loss_tensor, loss = self.loss_criterion.main_loss(pred_image.cpu(), item, suffix=suffix, targ_key="line_imgs")

        elif self.loss_version.lower()=="sm2sm":
            loss_tensor, loss, predicted_strokes = self.sm2sm(item, pred_image, gt_image)

        elif self.loss_version.lower()=="sm2gt":
            # Compare predicted strokes and GT strokes
            predicted_strokes = self.stroke_model(pred_image)
            #self.loss_criterion.loss(predicted_strokes, gt_strokes)
            loss_tensor, loss = self.loss_criterion.main_loss(predicted_strokes, item, suffix=suffix, targ_key="gt_list")

        if train:
            self.optimizer.zero_grad()
            loss_tensor.backward()
            torch.nn.utils.clip_grad_norm_(self.config.model.parameters(), 10)
            self.optimizer.step()

        return loss, pred_image, predicted_strokes


class GeneratorTrainer2(GeneratorTrainer):
    """ The generator for AG stuff???

    """
    def __init__(self, model, optimizer, config, stroke_model, loss_criterion=None, training_dataset=None, **kwargs):
        super().__init__(model, optimizer, config, stroke_model,
                         loss_criterion=loss_criterion,
                         training_dataset=training_dataset,
                         **kwargs)

    def stroke_eval(self, input, item, **kwargs):
        # Get the item and generate
        batch_size = item["line_imgs"].shape[0]
        initial_hidden, initial_window_vector, initial_kappa = self.stroke_model.init_hidden(batch_size, self.device)

        feature_maps = self.stroke_model.get_feature_maps(input)
        feature_maps_mask = torch.ones(feature_maps.shape[:2]).to(self.config.device) # B x W
        #feature_maps_mask = item["feature_map_mask"].to(self.config.device)

        preds = self.stroke_model.generate(feature_maps=feature_maps,
                                    feature_maps_mask=feature_maps_mask,
                                    hidden=initial_hidden,
                                    window_vector=initial_window_vector,
                                    kappa=initial_kappa,
                                    reset=True,
                                    forced_size=item["gt"].shape[1])

        preds[:, :, 0:1] = np.cumsum(preds[:, :, 0:1], axis=1) # SHOULD THEY BE SUMMED
        preds = torch.from_numpy(preds) # requires_grad=False
        return preds[:,:,:3] # SHAPE?


class AlexGravesTrainer(Trainer):
    def __init__(self, model, optimizer, config, loss_criterion=None, training_dataset=None, DETERMINISTIC=False, **kwargs):
        super().__init__(model, optimizer, config, loss_criterion)
        self.loss_criterion = loss_criterion
        self.generator_model = model
        self.training_dataset = training_dataset
        self.device = self.config.device
        self.DETERMINISTIC = DETERMINISTIC
        if DETERMINISTIC:
            for p in model.parameters():
                p.data.fill_(.01)

        if model.__class__.__name__=="AlexGravesCombined":
            self.train = self.train_new
        else:
            self.train = self.train_old
        self.cnn_type = self.model.cnn.cnn_type

    def test(self, item, **kwargs):
        self.model.eval()
        return self.train(item, train=False, **kwargs)

    def eval(self, input, **kwargs):
        return self.generator_model(**input)

    def stroke_eval(self, input):
        pred_logits = self.stroke_model(input).cpu().permute(1, 0, 2) # -> B,W,VOCAB
        return pred_logits

    def get_inital_lstm_args(self, initial_hidden, window_fm, window_letters, initial_kappa):
        image_lstm_args = {"initial_hidden":initial_hidden[0],
                             "prev_window_vec":window_fm,
                             "prev_eos": None,
                             "prev_kappa": initial_kappa}

        letter_lstm_args = {"initial_hidden": initial_hidden[0],
                             "prev_window_vec": window_letters,
                             "prev_eos": None,
                             "prev_kappa": initial_kappa}
        return image_lstm_args, letter_lstm_args

    def generate(self, item):
        imgs = item["line_imgs"].to(self.config.device)
        feature_maps = self.model.get_feature_maps(imgs) # B, W, 1024
        if "feature_map_mask" in item.keys():
            feature_maps_mask = item["feature_map_mask"].to(self.config.device) # Batch X Width
        else:
            lens = [img_width_to_pred_mapping(b, cnn_type=self.cnn_type) for b in item["img_widths"]]
            max_len = feature_maps.shape[1]
            feature_maps_mask = (torch.arange(max_len).expand(len(lens), max_len) < torch.tensor(lens).unsqueeze(1)).to(self.config.device)

        # letter_mask = item["gt_text_mask"].to(self.device)
        # letter_gt = item["gt_text_one_hot"].to(self.device)
        batch_size = item["line_imgs"].shape[0]
        initial_hidden, window_fm, window_letters, initial_kappa = self.model.init_hidden(batch_size, self.device)
        image_lstm_args, letter_lstm_args = self.get_inital_lstm_args(initial_hidden, window_fm, window_letters, initial_kappa)
        preds = self.model.generate(feature_maps=feature_maps,
                                    feature_maps_mask=feature_maps_mask,
                                    initial_hidden=initial_hidden,
                                    image_lstm_args=image_lstm_args,
                                    # letter_lstm_args=letter_lstm_args,
                                    # letter_gt=letter_gt,
                                    # letter_mask=letter_mask,
                                    reset=True)
        return preds

    def train_new(self, item, train=True, **kwargs):
        """ Alternate, letters only, image only

        Args:
            item:
            train:
            **kwargs:

        Returns:

        """
        if self.DETERMINISTIC:
            train = False

        if train:
            self.model.train()
            suffix="_train"
        else:
            self.model.eval()
            suffix="_test"

        batch_size = item["line_imgs"].shape[0]
        initial_hidden, window_fm, window_letters, initial_kappa = self.model.init_hidden(batch_size, self.device)
        image_lstm_args, letter_lstm_args = self.get_inital_lstm_args(initial_hidden, window_fm, window_letters, initial_kappa)

        imgs = item["line_imgs"].to(self.config.device)
        feature_maps = self.model.get_feature_maps(imgs)
        feature_maps_mask = item["feature_map_mask"].to(self.config.device)
        inputs = item["rel_gt"][:,:-1].to(self.config.device)
        letter_mask = item["gt_text_mask"].to(self.device)
        letter_gt = item["gt_text_one_hot"].to(self.device)
        model_input = {"inputs": inputs, # the shifted GTs
                        "img": imgs,
                        "img_mask": feature_maps_mask, # ignore
                        "initial_hidden": initial_hidden,
                        "image_lstm_args": image_lstm_args,
                        "feature_maps": feature_maps,
                        #"lengths": item["label_lengths"],
                        "reset": True,
                        "letter_lstm_args":letter_lstm_args,
                        "letter_mask": letter_mask,
                        "letter_gt": letter_gt,
                       } # reset hidden/cell states

        y_hat, states, image_lstm_args, letter_lstm_args = self.eval(model_input, ) # BATCH x 1 x H x W
        m = y_hat.detach().cpu().numpy()
        self.config.counter.update(epochs=0, instances=np.sum(item["label_lengths"]), updates=1)
        loss_tensor, loss = self.loss_criterion.main_loss(y_hat.cpu(), item, suffix=suffix, targ_key="rel_gt")

        if train:
            self.optimizer.zero_grad()
            loss_tensor.backward()
            torch.nn.utils.clip_grad_norm_(self.config.model.parameters(), 10)
            if "rnn_parameters" in self.model.__dict__.keys():
                nn.utils.clip_grad_value_(self.model.rnn_parameters, 1)
            self.optimizer.step()

        preds = None
        if chk_flg("return_preds",kwargs):
            image_lstm_args, letter_lstm_args = self.get_inital_lstm_args(initial_hidden, window_fm, window_letters,
                                                                          initial_kappa)

            # Kind of inane, generating based on feature maps and chars
            preds = self.model.generate(feature_maps=feature_maps,
                                        feature_maps_mask=feature_maps_mask,
                                        initial_hidden=initial_hidden,
                                        image_lstm_args=image_lstm_args,
                                        letter_lstm_args=letter_lstm_args,
                                        letter_gt=letter_gt,
                                        letter_mask=letter_mask,
                                        reset=True)
            # Convert to absolute coords
            preds[:,:,0:1] = np.cumsum(preds[:,:,0:1], axis=1)
            preds = torch.from_numpy(preds)
        return loss, preds, y_hat

    def train_old(self, item, train=True, **kwargs):
        """ My original Alex Graves method with IMAGES (not letters)
        """

        if self.DETERMINISTIC:
            train = False

        if train:
            self.model.train()
            suffix = "_train"
        else:
            self.model.eval()
            suffix = "_test"

        batch_size = item["line_imgs"].shape[0]
        initial_hidden, initial_window_vector, initial_kappa = self.model.init_hidden(batch_size, self.device)

        imgs = item["line_imgs"].to(self.config.device)
        #i = imgs.cpu().detach().numpy()
        feature_maps = self.model.get_feature_maps(imgs)
        feature_maps_mask = item["feature_map_mask"].to(self.config.device)
        gt_maps_makks = item["mask"]
        inputs = item["rel_gt"][:, :-1].to(self.config.device)
        # inputs = torch.zeros(item["rel_gt"][:,:-1].shape).to(self.config.device)

        model_input = {"inputs": inputs,  # the shifted GTs
                       "img": imgs,
                       "img_mask": feature_maps_mask,  # ignore
                       "initial_hidden": initial_hidden,  # RNN state
                       "prev_window_vec": initial_window_vector,
                       "prev_kappa": initial_kappa,
                       "feature_maps": feature_maps,
                       # "lengths": item["label_lengths"],
                       "is_map": False,
                       "reset": True}  # reset hidden/cell states

        y_hat, states, window_vec, prev_kappa, eos = self.eval(model_input, )  # BATCH x 1 x H x W
        m = y_hat.detach().cpu().numpy()
        self.config.counter.update(epochs=0, instances=np.sum(item["label_lengths"]), updates=1)
        loss_tensor, loss = self.loss_criterion.main_loss(y_hat.cpu(), item, suffix=suffix, targ_key="rel_gt")

        if train:
            self.optimizer.zero_grad()
            loss_tensor.backward()
            torch.nn.utils.clip_grad_norm_(self.config.model.parameters(), 10)
            if "rnn_parameters" in self.model.__dict__.keys():
                nn.utils.clip_grad_value_(self.model.rnn_parameters.parameters(), 1)
            self.optimizer.step()

        preds = None
        if chk_flg("return_preds", kwargs):
            preds = self.model.generate(feature_maps=feature_maps,
                                        feature_maps_mask=feature_maps_mask,
                                        hidden=initial_hidden,
                                        window_vector=initial_window_vector,
                                        kappa=initial_kappa,
                                        reset=True)
            # Convert to absolute coords
            preds[:, :, 0:1] = np.cumsum(preds[:, :, 0:1], axis=1)
            preds = torch.from_numpy(preds)
        return loss, preds, y_hat


class TrainerStrokeRecoverySampler(TrainerStrokeRecovery):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from hwr_utils import math

    def sample_loop(self):
        pass

    @staticmethod
    def eval(line_imgs, model, label_lengths=None, relative_indices=None, device="cuda",
             gt=None, train=False, convolve=None, sigmoid_activations=None, relu_activations=None,
             truncate=0, item=None):
        """ For offline data, that doesn't have ground truths
        """
        line_imgs = line_imgs.to(device)
        pred_logits = model(line_imgs, label_lengths, item=item).cpu()

        # Sample


        new_preds = pred_logits
        preds = new_preds.permute(1, 0, 2) # Width,Batch,Vocab -> Batch, Width, Vocab

        if relative_indices:
            if not train or convolve is None:
                preds = relativefy_batch_torch(preds, reverse=True, indices=relative_indices)  # assume they were in relative positions, convert to absolute
            else:
                preds = convolve(pred_rel=preds, indices=relative_indices, gt=gt)

        ## Shorten - label lengths currently = width of image after CNN
        truncate_window = 0 if truncate else 20
        if not label_lengths is None: #and truncate_window >= 0:
            # Convert square torch object to a list, removing predictions related to padding
            # Add a buffer of 20, so that each pred goes 20 past the EOS
            preds = TrainerStrokeRecovery._truncate(preds, label_lengths, window=truncate_window)

        # THIS IS A "PRE" ACTIVATION, MUST NOT BE DONE DURING TRAINING!
        if (sigmoid_activations or relu_activations) and not train:
            # PREDS ARE A LIST
            for i, p in enumerate(preds):
                preds[i][:, sigmoid_activations] = SIGMOID(p[:, sigmoid_activations])
                if relu_activations:
                    preds[i][:, relu_activations] = RELU(p[:, relu_activations])


        return preds

"""
# Output gradient clipping
y_hat.register_hook(lambda grad: torch.clamp(grad, -100, 100))

loss.backward()

# LSTM params gradient clipping
if model_type == "prediction":
    nn.utils.clip_grad_value_(model.parameters(), 10)
else:
    nn.utils.clip_grad_value_(model.lstm_1.parameters(), 10)
    nn.utils.clip_grad_value_(model.lstm_2.parameters(), 10)
    nn.utils.clip_grad_value_(model.lstm_3.parameters(), 10)
    nn.utils.clip_grad_value_(model.window_layer.parameters(), 10)
"""



if __name__ == '__main__':
    pass