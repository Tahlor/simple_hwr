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