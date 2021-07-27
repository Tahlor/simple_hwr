def train_old(self, item, train=True, **kwargs):
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
    # m = y_hat.detach().cpu().numpy()
    self.config.counter.update(epochs=0, instances=np.sum(item["label_lengths"]), updates=1)
    loss_tensor, loss = self.loss_criterion.main_loss(y_hat.cpu(), item, suffix=suffix, targ_key="rel_gt")

    if train:
        self.optimizer.zero_grad()
        loss_tensor.backward()
        torch.nn.utils.clip_grad_norm_(self.config.model.parameters(), 10)
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