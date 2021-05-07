from hwr_utils import visualize
from torch.utils.data import DataLoader
from torch import nn
from loss_module.stroke_recovery_loss import StrokeLoss
from trainers import *
from hwr_utils.stroke_dataset import StrokeRecoveryDataset, collate_stroke
from hwr_utils.stroke_recovery import *
from hwr_utils import utils
from torch.optim import lr_scheduler
from timeit import default_timer as timer
import argparse
from hwr_utils.hwr_logger import logger
from loss_module import losses
from models import start_points, stroke_model
from models.AlexGraves import AlexGravesCombined
from hwr_utils.stroke_plotting import *
from hwr_utils.utils import update_LR, reset_LR
from hwr_utils.stroke_plotting import draw_from_gt

## Change CWD to the folder containing this script
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

## Variations:
# Relative position
# CoordConv - 0 center, X-as-rectanlge
# L1 loss, DTW
# Dataset size

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./configs/stroke_config/baseline.yaml", help='Path to the config file.')
    parser.add_argument('--testing', action="store_true", default=False, help='Run testing version')
    #parser.add_argument('--name', type=str, default="", help='Optional - special name for this run')
    opts = parser.parse_args()
    return opts


def run_epoch(dataloader, epoch, report_freq=500, plot_graphs=True):
    # for i in range(0, 16):
    #     line_imgs = torch.rand(batch, 1, 60, 60)
    #     targs = torch.rand(batch, 16, 5)
    instances = 0
    start_time = timer()
    logger.info(("Epoch: ", epoch))

    for i, item in enumerate(dataloader):
        #print(item["label_lengths"])
        current_batch_size = item["line_imgs"].shape[0]
        instances += current_batch_size
        #print(item["gt"].shape, item["label_lengths"])
        last_one = (i+2==len(dataloader) or len(dataloader) <= 2)
        loss, preds, y_hat, *_ = trainer.train(item, train=True, return_preds=last_one) #
        #y = y_hat.cpu().detach().numpy()

        if loss is None:
            continue

        config.stats["Actual_Loss_Function_train"].accumulate(loss)

        if config.counter.updates % report_freq == 0 and i > 0:
            utils.reset_all_stats(config, keyword="_train")
            training_loss = config.stats["Actual_Loss_Function_train"].get_last()
            logger.info(("update: ", config.counter.updates, "combined loss: ", training_loss))

        # Make Epoch 0 preds to make sure it's working
        if epoch==1 and i==0 and not preds is None:
            logger.info(("Preds", preds[0]))
            logger.info(("GTs", item["gt_list"][0]))

        update_LR(config)

        if instances > 75000 or (last_one and not preds is None and plot_graphs):
            graph_procedure(preds, item, epoch=epoch, other=y_hat)
            break

    end_time = timer()
    logger.info(("Epoch duration:", end_time-start_time))

    #config.scheduler.step()
    training_loss = config.stats["Actual_Loss_Function_train"].get_last_epoch()
    return training_loss

def graph_procedure(preds, item, epoch=None, _type="train", other=None):
    # GRAPH
    if epoch is None:
        epoch = config.counter.epochs
    preds_to_graph = [p.permute([1, 0]) for p in preds]
    save_folder = graph(item, config=config, preds=preds_to_graph, _type=_type, epoch=epoch)
    if other is None:
        other = [None]
    utils.write_out(save_folder, "example_data", f"GT {str(item['gt_list'][0])}"
                                                 f"\nPREDS\n{str(preds_to_graph[0].transpose(1,0))}"
                                                 f"\nStartPoints\n{str(item['start_points'][0])}"
                                                 f"\nYHATs\n{str(other[0])}"
                                                 )
    utils.pickle_it({"item":item, "preds":[p.detach().numpy() for p in preds_to_graph]}, Path(save_folder) / "example_data.pickle")

def graph_gts(item):
    """ Make sure relative graphing is correct

    Args:
        item:

    Returns:

    """
    # Graph GTs
    gts = item["rel_gt"].clone().detach()  # B, W, 4
    print(gts.shape)
    gts[:, :, 0:1] = torch.cumsum(gts[:, :, 0:1], axis=1)
    gts = [p.permute([1, 0]) for p in gts]

    # np.save("screw_this.npy", [gts, item["rel_gt"], item["gt"]])
    save_folder = graph(item, config=config, preds=gts, _type="test2", epoch=epoch)


def test(dataloader):
    preds_to_graph = None
    for i, item in enumerate(dataloader):
        loss, preds, y_hat, *_ = trainer.test(item, return_preds= i == 0) #
        if loss is None:
            continue
        if i==0 and not preds is None:
            graph_procedure(preds, item, epoch=None, _type="test",other=y_hat)

        config.stats["Actual_Loss_Function_test"].accumulate(loss)

    utils.reset_all_stats(config, keyword="_test")

    for loss in config.stats:
        try:
            # Print recent snapshot
            plt.plot(config.stats[f"{loss}"].x[-100:], config.stats[f"{loss}"].y[-100:])
            plt.savefig(config.image_dir / f"{loss}")
            plt.clf()
            plt.close('all')

            # Print entire graph
            max_length = min(len(config.stats[f"{loss}"].x), len(config.stats[f"{loss}"].y))
            plt.plot(config.stats[f"{loss}"].x[-max_length:], config.stats[f"{loss}"].y[-max_length:])
            plt.savefig(config.image_dir / f"{loss}_complete")
            plt.clf()
            plt.close('all')
        except Exception as e:
            logger.info(f"Problem graphing: {e}")
            pass

    return config.stats["Actual_Loss_Function_test"].get_last()


def graph(batch,
          config=None,
          preds=None,
          _type="test",
          save_folder="auto",
          epoch="current",
          show=False,
          plot_points=True,
          max_plots=10):
    if save_folder == "auto":
        _epoch = str(epoch)
        save_folder = (config.image_dir / _epoch / _type)
        save_folder.mkdir(parents=True, exist_ok=True)
    elif save_folder is not None:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
    else:
        show = True

    print("saving", save_folder)
    def subgraph(coords, gt_img, name, is_gt=True):
        ## PREDS
        if not is_gt:
            # Prep for other plot
            if coords is None:
                return
            coords = utils.to_numpy(coords[i])
            #print("before round", coords[2])

            # Remove lonely points - only works with stroke numbers
            # coords = post_process_remove_strays(coords)
            if "stroke_number" in config.gt_format:
                idx = config.gt_format.index("stroke_number")
                if config.pred_opts[idx]=="cumsum": # are the PREDS also CUMSUM?? or just the GTs
                    # coords[idx] = convert_stroke_numbers_to_start_strokes(coords[idx])
                    coords[idx] = relativefy_numpy(coords[idx], reverse=False)

            # Round the SOS, EOS etc. items
            coords[2:, :] = np.round(coords[2:, :]) # VOCAB SIZE, LENGTH
            #print("after round", coords[2])
            suffix=""
        else:
            suffix="_gt"
            coords = utils.to_numpy(coords).transpose() # LENGTH, VOCAB => VOCAB SIZE, LENGTH

            if "stroke_number" in config.gt_format:
                idx = config.gt_format.index("stroke_number")
                coords[idx] = relativefy_numpy(coords[idx], reverse=False)

        # this isn't right, not really
        if "x_rel" in config.gt_format:
            idx = config.gt_format.index("x_rel")
            coords[idx] = relativefy_numpy(coords[idx], reverse=True) # take CUMSUM

        # Flip everything for PIL
        # gt_img = torch.flip(gt_img, (0,))

        # Red images
        bg = overlay_images(background_img=gt_img.numpy(), foreground_gt=coords.transpose())
        if save_folder:
            bg.save(save_folder / f"overlay{suffix}_{i}_{name}.png")

        if show:
            plt.figure(dpi=300)
            plt.imshow(bg)
            plt.show()

        ## Undo relative positions for X for graphing
        ## In normal mode, the cumulative sum has already been taken
        if plot_points:
            save_path = save_folder / f"{i}_{name}{suffix}.png" if save_folder else None
            if config.dataset.image_prep.lower().startswith('pil'):
                render_points_on_image(gts=coords, img=gt_img.numpy() , save_path=save_path, origin='lower',
                                       invert_y_image=True, show=show, freq=3)
            else:
                render_points_on_image_matplotlib(gts=coords, img_path=img_path, save_path=save_path,
                                       origin='lower', show=show, freq=3
                                                  )

    # Loop through each item in batch
    for i, el in enumerate(batch["paths"]):
        img_path = el
        # Flip back to upper origin format for PIL
        gt_img = np.squeeze(batch["line_imgs"][i]) # BATCH, CHANNEL, H, W, FLIP IT
        name=Path(batch["paths"][i]).stem
        if _type != "eval":
            if config is None or config.model_name == "normal" or config.model_name=="AlexGraves":
                subgraph(batch["gt_list"][i], gt_img, name, is_gt=True)
            elif config.model_name=="start_points":
                subgraph(batch["start_points"][i], gt_img, name, is_gt=True)
        subgraph(preds, gt_img, name, is_gt=False)
        if i > max_plots or i+2 > len(preds):
            break
    return save_folder

def build_data_loaders(folder, cnn_type, train_size, test_size, **kwargs):
    ## LOAD DATASET
    NUM_WORKERS = 5
    if config.TESTING:
        NUM_WORKERS=1

    if NUM_WORKERS==1:
        warnings.warn("ONLY 1 WORKER!!!")
        if not config.TESTING:
            warnings.warn("AUTOMATIC OVERRIDE, USING 5 WORKERS!!!")
            NUM_WORKERS = 5

    if not config.test_only:
        train_dataset=StrokeRecoveryDataset([folder / "train_online_coords.json", *kwargs["extra_dataset"]],
                                root=config.data_root,
                                max_images_to_load = train_size,
                                cnn_type=cnn_type,
                                training=True,
                                **kwargs,
                                )

        cs = lambda x: train_dataset.collate(x, alphabet_size=train_dataset.alphabet_size)

        # Backup the alphabet
        utils.backup_alphabet(train_dataset, config)

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=NUM_WORKERS,
                                      collate_fn=cs,
                                      pin_memory=False)

        config.n_train_instances = len(train_dataloader.dataset)
    else:
        config.n_train_instances = 1
        train_dataset = train_dataloader = None
        cs = lambda x: train_dataset.collate(x, alphabet_size=config.alphabet_size)

    test_dataset=StrokeRecoveryDataset([folder / "test_online_coords.json"],
                            root=config.data_root,
                            max_images_to_load=test_size,
                            cnn_type=cnn_type,
                            test_dataset = True,
                            **kwargs
                            )
    if set(test_dataset.char_to_idx.keys()) != set(config.char_to_idx.keys()):
        utils.backup_alphabet(source_dict=config, destination_dict=test_dataset)
        warnings.warn("Test and training alphabets are different!")

    test_dataloader = DataLoader(test_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS,
                                  collate_fn=cs,
                                  pin_memory=False)

    # Update alphabets


    n_test_points = 0
    for i in test_dataloader:
        n_test_points += sum(i["label_lengths"])
    config.n_test_instances = len(test_dataloader.dataset)
    config.n_test_points = int(n_test_points)
    config.training_dataset = train_dataset
    config.test_dataset = test_dataset
    return train_dataloader, test_dataloader

def main(config_path, testing=False, eval_only=False, eval_dataset=None, load_path_override=None, eval_once=False):
    global epoch, device, trainer, batch_size, output, loss_obj, config, LOGGER
    torch.cuda.empty_cache()
    os.chdir(ROOT_DIR)

    config = utils.load_config(config_path, hwr=False, testing=testing)
    test_size = config.test_size
    train_size = config.train_size
    batch_size = config.batch_size
    vocab_size = config.feature_map_dim
    device = config.device if not utils.no_gpu_testing() else 'cpu'
    config.device = device # these need to be the same

    # Free GPU memory if necessary
    if device == "cuda":
        utils.kill_gpu_hogs()

    #output = utils.increment_path(name="Run", base_path=Path("./results/stroke_recovery"))
    output = Path(config.results_dir)
    output.mkdir(parents=True, exist_ok=True)
    # folder = Path("online_coordinate_data/3_stroke_32_v2")
    # folder = Path("online_coordinate_data/3_stroke_vSmall")
    # folder = Path("online_coordinate_data/3_stroke_vFull")
    # folder = Path("online_coordinate_data/8_stroke_vFull")
    # folder = Path("online_coordinate_data/8_stroke_vSmall_16")
    folder = Path(config.dataset_folder)

    if config.model_name != "normal":

        # SOS will still be the 2 index, just ignore it!
        # config.input_vocab_size = 3
        #input_vocab_size = 3
        pass

    ### LOAD DATA
    # Alphabet size needed to build model
    # ALthough, the CNN used to be needed to build the dataset, ugh

    logger.info(("Current dataset: ", folder))

    if not eval_only:
        train_dataloader, test_dataloader = build_data_loaders(folder, config.cnn_type, train_size, test_size, **config.dataset,
                                                config=config)

    # This is not exactly correct, but will work if alphabet of training or test is a subset of the other
    alphabet_size = config.alphabet_size if "alphabet_size" in config.keys() else max(config.training_dataset.alphabet_size, config.test_dataset.alphabet_size)
    model_kwargs = {"feature_map_dim": 1024,
                    "device":device,
                    "cnn_type":config.cnn_type,
                    "first_conv_op":config.coordconv,
                    "first_conv_opts":config.coordconv_opts,
                    "alphabet_dim": alphabet_size,
                    **config.model_definition}

    model_dict = {"start_point_lstm": start_points.StartPointModel,
              "start_point_lstm2": start_points.StartPointModel2,
              "start_point_attn": start_points.StartPointAttnModel,
              "start_point_attn_deep": start_points.StartPointAttnModelDeep,
              "start_point_attn_full": start_points.StartPointAttnModelFull,
              "normal": stroke_model.StrokeRecoveryModel,
              "AlexGraves": stroke_model.AlexGraves,
              "TMinus1": stroke_model.TMinus1,
              "AlexGraves2": stroke_model.AlexGraves2,
              "AlexGravesCombined": AlexGravesCombined
              }

    model_class = model_dict[config.model_name]
    model = model_class(**model_kwargs).to(device)

    cnn = model.cnn # if set to a cnn object, then it will resize the GTs to be the same size as the CNN output

    # example = next(iter(test_dataloader)) # BATCH, WIDTH, VOCAB
    # input_vocab_size = example["gt"].shape[-1]

    ## Stats
    # Generic L1 loss
    config.L1 = losses.L1(loss_indices=slice(0, 2))

    if config.use_visdom:
        utils.start_visdom(port=config.visdom_port)
        visualize.initialize_visdom(config["full_specs"], config)
    utils.stat_prep_strokes(config)

    # Create loss object
    if "training_dataset" not in config or config.training_dataset is None:
        trainset_data = None
    else:
        trainset_data = config.training_dataset.data

    config.loss_obj = StrokeLoss(loss_stats=config.stats, counter=config.counter, device=device, training_dataset=trainset_data)

    LR = config.learning_rate * batch_size/24
    logger.info(f"Specified LR: {config.learning_rate}, Effective: {LR}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    config.scheduler = utils.new_scheduler(optimizer, batch_size, gamma=config.scheduler_gamma)  # halves every ~10 "super" epochs
    # config.scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=80, verbose=False,
    #                                             threshold=0.00005, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    if config.model_name.lower == "startpoints":
        trainer = TrainerStartPoints(model, optimizer, config=config, loss_criterion=config.loss_obj)
    elif "AlexGraves" in config.model_name:
        trainer = AlexGravesTrainer(model, optimizer, config=config, loss_criterion=config.loss_obj)
    else:
        trainer = TrainerStrokeRecovery(model, optimizer, config=config, loss_criterion=config.loss_obj)

    config.optimizer=optimizer
    config.trainer=trainer
    config.model = model
    logger.info(f"LR before loading model: {next(iter(config.optimizer.param_groups))['lr']}")
    if load_path_override:
        config.load_path = load_path_override
    if config.load_path: #and not utils.no_gpu_testing(): # don't load model if not using GPU
        utils.load_model_strokes(config, config.load_optimizer)  # should be load_model_strokes??????
        print(config.counter.epochs)

    if config.reset_LR:
        logger.info("Resetting LR")
        reset_LR(config, LR)

    logger.info(f"Starting LR is {next(iter(config.optimizer.param_groups))['lr']}")

    def main_loop():
        check_epoch_build_loss(config, loss_exists=False)
        current_epoch = config.counter.epochs
        for i in range(current_epoch,config.epochs_to_run):
            epoch = i+1
            #config.counter.epochs = epoch
            config.counter.update(epochs=1)
            plot_graphs = True if epoch % config.test_freq == 0 else False

            if train_dataloader:
                loss = run_epoch(train_dataloader, epoch=epoch, report_freq=config.update_freq, plot_graphs=plot_graphs)
                logger.info(f"Epoch: {epoch}, Training Loss: {loss}")

            # Test and save models
            if epoch % config.test_freq == 0:
                test_loss = test(test_dataloader)
                logger.info(f"Epoch: {epoch}, Test Loss: {test_loss}")
                check_epoch_build_loss(config)
                all_test_losses = [x for x in config.stats["Actual_Loss_Function_test"].y if x]
                if len(all_test_losses) and test_loss <= np.nanmin(all_test_losses):
                    utils.save_model_stroke(config, bsf=True)
                    continue # already saved model and strokes

            if epoch % config.save_freq == 0: # how often to save
                utils.save_model_stroke(config, bsf=False) # also saves stats
            else:
                utils.save_stats_stroke(config, bsf=False)
            if config.test_only:
                break

    def eval_once():
        check_epoch_build_loss(config, loss_exists=False)
        test(test_dataloader)

    if not eval_only:
        main_loop()
    else:
        if eval_once:
            eval_once()
        else:
            return config

    ## Bezier curve
    # Have network predict whether it has reached the end of a stroke or not
    # If it has not reached the end of a stroke, the starting point = previous end point

def check_epoch_build_loss(config, loss_exists=True):
    epoch = config.counter.epochs

    # If we should be on loss_fn2
    if (config.first_loss_epochs and epoch == config.first_loss_epochs) or (not loss_exists and epoch >= config.first_loss_epochs):
        if "loss_fns2" in config and config.loss_fns2:
            logger.info("Building loss 2")
            config.loss_obj.build_losses(config.loss_fns2)
            return

    if not loss_exists:
        logger.info("Building loss 1")
        config.loss_obj.build_losses(config.loss_fns)

if __name__=="__main__":
    opts = parse_args()
    main(config_path=opts.config, testing=opts.testing)
    
    # TO DO:
        # logging
        # Get running on super computer - copy the data!