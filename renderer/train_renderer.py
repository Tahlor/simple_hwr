import sys
sys.path.append("..")
from hwr_utils import visualize
from torch.utils.data import DataLoader
from torch import nn
from loss_module.stroke_recovery_loss import StrokeLoss
from trainers import TrainerStrokeRecovery, GeneratorTrainer
from models.stroke_model import StrokeRecoveryModel
from hwr_utils.stroke_dataset import StrokeRecoveryDataset
from hwr_utils.stroke_recovery import *
from hwr_utils import utils
from torch.optim import lr_scheduler
from timeit import default_timer as timer
import argparse
from hwr_utils.hwr_logger import logger
from loss_module import losses
from models import start_points, stroke_model
from hwr_utils.stroke_plotting import *
from hwr_utils.utils import update_LR, reset_LR, plot_loss
from hwr_utils.stroke_plotting import draw_from_gt
import model_renderer

numpify = lambda x : x.detach().cpu().numpy()

## Change CWD to the folder containing this script
ROOT_DIR = Path(os.path.dirname(os.path.realpath(__file__))).parent
os.chdir(ROOT_DIR)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./configs/stroke_config/baseline.yaml",
                        help='Path to the config file.')
    parser.add_argument('--testing', action="store_true", default=False, help='Run testing version')
    # parser.add_argument('--name', type=str, default="", help='Optional - special name for this run')
    opts = parser.parse_args()
    return opts

def run_epoch(dataloader, report_freq=500, plot_graphs=True):
    instances = 0
    start_time = timer()
    logger.info(("Epoch: ", epoch))

    for i, item in enumerate(dataloader):
        current_batch_size = item["line_imgs"].shape[0]
        instances += current_batch_size
        loss, pred_image, predicted_strokes, *_ = trainer.train(item, train=True)

        if loss is None:
            continue

        config.stats["Actual_Loss_Function_train"].accumulate(loss)

        if config.counter.updates % report_freq == 0 and i > 0:
            utils.reset_all_stats(config, keyword="_train")
            training_loss = config.stats["Actual_Loss_Function_train"].get_last()
            logger.info(("update: ", config.counter.updates, "combined loss: ", training_loss))

        if epoch == 1 and i == 0:
            # logger.info(("Preds", preds[0]))
            logger.info(("GTs", item["gt_list"][0]))

        update_LR(config)

    end_time = timer()
    logger.info(("Epoch duration:", end_time - start_time))

    ## Draw the pred image, draw the pred stroke_gt, draw the
    path = (config.image_dir / str(config.counter.epochs) / "train")
    path.mkdir(parents=True, exist_ok=True)
    save_out(item, predicted_strokes, pred_image, path)
    # config.scheduler.step()
    training_loss = config.stats["Actual_Loss_Function_train"].get_last_epoch()
    return training_loss

def save_out(item, predicted_strokes, pred_image, path):
    print("Saving graphs...")
    print(predicted_strokes)
    if predicted_strokes is not None:
        save_stroke_images(pred_image,
                           predicted_strokes, path, is_gt=False)
    else:
        save_images(pred_image, path, is_gt=False)

    # Save GTs
    if "predicted_strokes_gt" in item and item["predicted_strokes_gt"][0] is not None:
        save_stroke_images(item["line_imgs"],
                           item["predicted_strokes_gt"], path, is_gt=True)
    else:
        save_images(item["line_imgs"], path, is_gt=True)


def save_images(list_of_images, path, is_gt, normalized=True):
    rescale = lambda x: (x + 1) * 127.5 if normalized else lambda x: x
    for i, image in enumerate(list_of_images):
        if isinstance(image, Tensor):
            image = numpify(image)
        new_img = Image.fromarray(np.uint8(rescale(np.squeeze(image))), 'L')
        file_name = f"{i}_{'gt' if is_gt else 'pred'}.tif"
        new_img.save(path / file_name)

### THIS SHOULD BE AN OVERLAY UGHH
def save_stroke_images(list_of_images, list_of_coords, path, is_gt):
    for i, image in enumerate(list_of_images):
        if isinstance(image, Tensor):
            image = numpify(image)
        if isinstance(list_of_coords[i], Tensor):
            coords = numpify(list_of_coords[i])

        file_name = f"i_{'gt' if is_gt else 'pred'}.tif"
        coords_i = utils.prep_coords_to_graph(config, coords, is_gt=True)
        img = overlay_images(background_img=image, foreground_gt=coords_i.transpose(), save_path = path / file_name)

def test(dataloader):
    preds_to_graph = None
    for i, item in enumerate(dataloader):
        loss, pred_image, predicted_strokes, *_ = trainer.test(item)
        if loss is None:
            continue
        config.stats["Actual_Loss_Function_test"].accumulate(loss)

    # Save images
    path = (config.image_dir / str(config.counter.epochs) / "test")
    path.mkdir(parents=True, exist_ok=True)
    save_out(item, predicted_strokes, pred_image, path)

    utils.reset_all_stats(config, keyword="_test")

    for loss in config.stats:
        plot_loss(config, loss)

    return config.stats["Actual_Loss_Function_test"].get_last()

def build_data_loaders(folder, cnn, train_size, test_size, **kwargs):
    ## LOAD DATASET
    NUM_WORKERS = 5
    if config.TESTING:
        NUM_WORKERS = 1

    if NUM_WORKERS == 1:
        warnings.warn("ONLY 1 WORKER!!!")
        if not config.TESTING:
            warnings.warn("AUTOMATIC OVERRIDE, USING 5 WORKERS!!!")
            NUM_WORKERS = 5

    train_dataset = StrokeRecoveryDataset([folder / "train_online_coords.json", *kwargs["extra_dataset"]],
                                          root=config.data_root,
                                          max_images_to_load=train_size,
                                          cnn=cnn,
                                          training=True,
                                          **kwargs,
                                          )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS,
                                  collate_fn=train_dataset.collate,
                                  pin_memory=False)

    config.n_train_instances = len(train_dataloader.dataset)

    test_dataset = StrokeRecoveryDataset([folder / "test_online_coords.json"],
                                         root=config.data_root,
                                         max_images_to_load=test_size,
                                         cnn=cnn,
                                         test_dataset=True,
                                         **kwargs
                                         )

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=NUM_WORKERS,
                                 collate_fn=train_dataset.collate,
                                 pin_memory=False)

    n_test_points = 0
    for i in test_dataloader:
        n_test_points += sum(i["label_lengths"])
    config.n_test_instances = len(test_dataloader.dataset)
    config.n_test_points = int(n_test_points)
    config.training_dataset = train_dataset
    config.test_dataset = test_dataset
    return train_dataloader, test_dataloader

def load_stroke_model(config_path, model_path=None):
    config = utils.load_config(config_path, hwr=False, create_logger=False)

    # Supercede the YAML model if one specified
    if not model_path is None:
        config.load_path = model_path

    config.use_visdom = False
    # Free GPU memory if necessary
    if config.device == "cuda":
        utils.kill_gpu_hogs()

    batch_size = config.batch_size
    vocab_size = config.vocab_size
    device = torch.device(config.device)

    model = StrokeRecoveryModel(vocab_size=vocab_size, device=device, cnn_type=config.cnn_type,
                                first_conv_op=config.coordconv, first_conv_opts=config.coordconv_opts).to(device)
    config.model = model
    utils.load_model_strokes(config)  # should be load_model_strokes??????
    model = model.to(device)
    model.eval()
    return model


def main(config_path, testing=False):
    global epoch, device, trainer, batch_size, output, loss_obj, config, LOGGER
    torch.cuda.empty_cache()
    os.chdir(ROOT_DIR)

    config = utils.load_config(config_path, hwr=False, testing=testing, subpath="renderer")

    if config.trainer_args.loss_type.lower() in ["sm2sm"]:
        config.stroke_model = load_stroke_model(config.stroke_model_config, config.stroke_model_pt_override)
    else:
        config.stroke_model = None

    test_size = config.test_size
    train_size = config.train_size
    batch_size = config.batch_size
    vocab_size = config.vocab_size
    device = config.device if not utils.no_gpu_testing() else 'cpu'
    config.device = device  # these need to be the same

    # Free GPU memory if necessary
    if device == "cuda":
        utils.kill_gpu_hogs()

    # output = utils.increment_path(name="Run", base_path=Path("./results/stroke_recovery"))
    output = Path(config.results_dir)
    output.mkdir(parents=True, exist_ok=True)
    folder = Path(config.dataset_folder)

    model_kwargs = {**config.model_definition}

    model_dict = {"start_point_lstm": start_points.StartPointModel,
                  "start_point_lstm2": start_points.StartPointModel2,
                  "start_point_attn": start_points.StartPointAttnModel,
                  "start_point_attn_deep": start_points.StartPointAttnModelDeep,
                  "start_point_attn_full": start_points.StartPointAttnModelFull,
                  "normal": stroke_model.StrokeRecoveryModel,
                  "Renderer": model_renderer.Renderer}

    model_class = model_dict[config.model_definition.model_name]
    model = model_class(**model_kwargs).to(device)

    cnn = model.cnn  # if set to a cnn object, then it will resize the GTs to be the same size as the CNN output
    logger.info(("Current dataset: ", folder))

    train_dataloader, test_dataloader = build_data_loaders(folder, cnn, train_size, test_size, **config.dataset,
                                                           config=config)

    ## Stats
    # Generic L1 loss
    config.L1 = losses.L1(loss_indices=slice(0, 2))

    if config.use_visdom:
        utils.start_visdom(port=config.visdom_port)
        visualize.initialize_visdom(config["full_specs"], config)
    utils.stat_prep_strokes(config)

    # Create loss object
    config.loss_obj = StrokeLoss(loss_stats=config.stats, counter=config.counter, device=device,
                                 training_dataset=config.training_dataset.data)

    LR = config.learning_rate * batch_size / 24
    logger.info(f"Specified LR: {config.learning_rate}, Effective: {LR}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    config.scheduler = utils.new_scheduler(optimizer, batch_size,
                                           gamma=config.scheduler_gamma)  # halves every ~10 "super" epochs
    # config.scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=80, verbose=False,
    #                                             threshold=0.00005, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    trainer = GeneratorTrainer(model, optimizer,
                               stroke_model=config.stroke_model,
                               config=config,
                               loss_criterion=config.loss_obj,
                               training_dataset=config.training_dataset.data,
                               **config.trainer_args)

    config.optimizer = optimizer
    config.trainer = trainer
    config.model = model
    logger.info(f"LR before loading model: {next(iter(config.optimizer.param_groups))['lr']}")

    # Loading not supported yet
    # if config.load_path and not utils.no_gpu_testing():  # don't load model if not using GPU
    #     utils.load_model_strokes(config, config.load_optimizer)
    #     print(config.counter.epochs)

    if config.reset_LR:
        logger.info("Resetting LR")
        reset_LR(config, LR)

    logger.info(f"Starting LR is {next(iter(config.optimizer.param_groups))['lr']}")

    check_epoch_build_loss(config, loss_exists=False)
    current_epoch = config.counter.epochs
    for i in range(current_epoch, config.epochs_to_run):
        epoch = i + 1
        # config.counter.epochs = epoch
        config.counter.update(epochs=1)
        plot_graphs = True if epoch % config.test_freq == 0 else False
        loss = run_epoch(train_dataloader, report_freq=config.update_freq, plot_graphs=plot_graphs)
        logger.info(f"Epoch: {epoch}, Training Loss: {loss}")

        # Test and save models
        if epoch % config.test_freq == 0:
            test_loss = test(test_dataloader)
            logger.info(f"Epoch: {epoch}, Test Loss: {test_loss}")
            check_epoch_build_loss(config)
            all_test_losses = [x for x in config.stats["Actual_Loss_Function_test"].y if x and x > 0]
            if all_test_losses and test_loss <= np.min(all_test_losses):
                utils.save_model_stroke(config, bsf=True)
        if epoch % config.save_freq == 0:  # how often to save
            utils.save_model_stroke(config, bsf=False)  # also saves stats
        else:
            utils.save_stats_stroke(config, bsf=False)

    ## Bezier curve
    # Have network predict whether it has reached the end of a stroke or not
    # If it has not reached the end of a stroke, the starting point = previous end point


def check_epoch_build_loss(config, loss_exists=True):
    epoch = config.counter.epochs

    # If we should be on loss_fn2
    if (config.first_loss_epochs and epoch == config.first_loss_epochs) or (
            not loss_exists and epoch >= config.first_loss_epochs):
        if "loss_fns2" in config and config.loss_fns2:
            logger.info("Building loss 2")
            config.loss_obj.build_losses(config.loss_fns2)
            return

    if not loss_exists:
        logger.info("Building loss 1")
        config.loss_obj.build_losses(config.loss_fns)


if __name__ == "__main__":
    opts = parse_args()
    main(config_path=opts.config, testing=opts.testing)

    # TO DO:
    # logging
    # Get running on super computer - copy the data!