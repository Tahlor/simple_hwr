### BASED ON OFFLINE RECOGNIZER TRAIN SCRIPT

from __future__ import print_function

import sys
sys.path.extend(["..", "."])
import torch, os
from hwr_utils.utils import unpickle_it, npy_loader, dict_to_list
from hwr_utils.hwr_logger import logger
import traceback
from builtins import range
import faulthandler
from hwr_utils import hw_dataset, character_set
from online_dataset import OnlineDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import tensor
import types
from hwr_utils import utils
from hwr_utils.utils import plot_recognition_images
import numpy as np
from hwr_utils.utils import *
from recognition import crnn
import models_online

def validate(model, dataloader, idx_to_char, device, config):
    """ Validate a model -- save if best so far"""

    validation_cer = test(model, dataloader, idx_to_char, device, config, with_analysis=False, plot_all=False, validation=True, with_iterations=False)
    LOGGER.info(f"Validation CER: {validation_cer}")

    if config['lowest_loss'] > validation_cer:
        if config["validation_jsons"]:
            test_cer = test(config["model"], config.test_dataloader, config["idx_to_char"], config["device"], config,
                            validation=False, with_iterations=False)
            LOGGER.info(f"Saving Best Loss! Test CER: {test_cer}")
        else:
            test_cer = validation_cer

        config['lowest_loss'] = validation_cer
        save_model(config, bsf=True)
    return validation_cer

def test(model, dataloader, idx_to_char, device, config, with_analysis=False, plot_all=False, validation=True, with_iterations=False):
    """ Test/validate a model. Validation bool just specifies which stats to update

    Args:
        model:
        dataloader:
        idx_to_char:
        device:
        config:
        with_analysis:
        plot_all:
        validation:
        with_iterations:

    Returns:

    """

    model.eval()
    i = -1
    stat = "validation" if validation else "test"

    for i,x in enumerate(dataloader):
        line_imgs = x['line_imgs'].to(device)
        gt = x['gt']  # actual string ground truth

        if "strokes" in x and x["strokes"] is not None:
            online = x["strokes"].to(device)
        else:
            online = Variable(x['online'].to(device), requires_grad=False).view(1, -1, 1) if config[
                "online_augmentation"] and config["online_flag"] else None


        loss, initial_err, pred_str = config["trainer"].test(line_imgs, online, gt, validation=validation, with_iterations=with_iterations)

        if plot_all:
            imgs = x["line_imgs"][:, 0, :, :, :] if config["n_warp_iterations"] else x['line_imgs']
            plot_recognition_images(imgs, f"{config['current_epoch']}_{i}_testing", pred_str, config["image_test_dir"], plot_count=4)

        # Only do one test
        if config["TESTING"]:
            break

    if i >= 0: # if there was any test data, calculate the CER
        utils.reset_all_stats(config, keyword=stat)
        cer = config["stats"][config[f"designated_{stat}_cer"]].y[-1]  # most recent test CER

        if not plot_all:
            imgs = x["line_imgs"][:, 0, :, :, :] if with_iterations else x['line_imgs']
            plot_recognition_images(imgs, f"{config['current_epoch']}_testing", pred_str, config["image_test_dir"], plot_count=4)

        LOGGER.debug(config["stats"])
        return cer
    else:
        log_print(f"No {stat} data!")
        return np.inf

def run_epoch(model, dataloader, ctc_criterion, optimizer, dtype, config):
    LOGGER.debug(f"Switching model to train")
    model.train()
    config["stats"]["epochs"] += [config["current_epoch"]]
    plot_freq = config["plot_freq"]
    local_instance_counter = 0
    test_freq = 6000 # in terms of instances
    next_update = test_freq

    for i, x in enumerate(dataloader):
        LOGGER.debug(f"Training Iteration: {i}")
        config.counter.update(epochs=1)
        try:
            line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False)
            labels = Variable(x['labels'], requires_grad=False)  # numeric loss_indices version of ground truth
            label_lengths = Variable(x['label_lengths'], requires_grad=False)
            gt = x['gt']  # actual string ground truth
            config["global_step"] += 1
            config["global_instances_counter"] += line_imgs.shape[0]
            local_instance_counter += line_imgs.shape[0]
        except:
            logger.info("Problem with epoch")
            traceback.print_exc()

        loss, initial_err, first_pred_str = config["trainer"].train(line_imgs, online, labels, label_lengths, gt, step=config["global_step"])

        LOGGER.debug("Finished with batch")

        if i == 0:
            plot_recognition_images(x['line_imgs'], f"{config['current_epoch']}_{local_instance_counter}_training", first_pred_str,
                        dir=config["image_train_dir"], plot_count=4)


        # Run a validation set if training set is HUGE and no end in sight
        # if local_instance_counter>=next_update and config['n_train_instances']-local_instance_counter > test_freq:
        #     logger.info("Validating - mid epoch!")
        #     validate(config["model"], config.validation_dataloader, config["idx_to_char"], config["device"], config)
        #     next_update += test_freq
        # 
        #     # Save out example images on the first go
        #     plot_recognition_images(x['line_imgs'], f"{config['current_epoch']}_{local_instance_counter}_training", first_pred_str,
        #                 dir=config["image_train_dir"], plot_count=4)


        # Update stats every 50 instances
        if (config["global_step"] % plot_freq == 0 and config["global_step"] > 0) or config["TESTING"] or config["SMALL_TRAINING"]:
            config["stats"]["updates"] += [config["global_step"]]
            config["stats"]["epoch_decimal"] += [
                config["current_epoch"] + i * config["batch_size"] * 1.0 / config['n_train_instances']]
            LOGGER.info(f"updates: {config['global_step']}")
            utils.reset_all_stats(config, keyword="Training")

        if config["TESTING"] or config["SMALL_TRAINING"]:
            break

    try:
        training_cer_list = config["stats"][config["designated_training_cer"]].y

        # if not training_cer_list:
        #     reset_all_stats(config, keyword="training")
        training_cer = training_cer_list[-1]  # most recent training CER

        # Save out example images on the first go
        plot_recognition_images(x['line_imgs'], f"{config['current_epoch']}_training", first_pred_str,
                    dir=config["image_train_dir"], plot_count=4)

        return training_cer
    except:
        logger.info("Problem with calculating error")
        return np.inf

def make_dataloaders(config, device="cpu"):
    threads = 5
    default_collate = lambda x: hw_dataset.collate(x, device=device)
    train_dataset = OnlineDataset(config.training_jsons,
                              char_to_idx=config["char_to_idx"],
                              root=config["training_root"],
                              warp=config["training_warp"],
                              blur=config["training_blur"],
                              blur_level=config.get("training_blur_level", 1.5),
                              random_distortions=config["training_random_distortions"],
                              distortion_sigma=config["training_distortion_sigma"],
                              writer_id_paths=config["writer_id_pickles"],
                              max_images_to_load=config["images_to_load"],
                              occlusion_size=config["occlusion_size"],
                              occlusion_freq=config["occlusion_freq"],
                              occlusion_level=config["max_intensity"],
                              logger=config["logger"],
                              config=config,
                              **config.dataset)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config["batch_size"],
                                  shuffle=config["training_shuffle"],
                                  num_workers=threads,
                                  collate_fn=default_collate,
                                  pin_memory=device=="cpu")

    # test_dataset = HwDataset(config["testing_jsons"],
    #                          config["char_to_idx"],
    #                          img_height=config["input_height"],
    #                          num_of_channels=config["num_of_channels"],
    #                          root=config["testing_root"],
    #                          warp=config["testing_warp"],
    #                          blur=config.get("testing_blur", 1.5),
    #                          blur_level=config["testing_blur_level"],
    #                          random_distortions=config["testing_random_distortions"],
    #                          distortion_sigma=config["testing_distortion_sigma"],
    #                          images_to_load=config["images_to_load"],
    #                          logger=config["logger"])

    test_dataset = OnlineDataset(config["testing_jsons"],
                             config["char_to_idx"],
                             root=config["testing_root"],
                             warp=False,
                             blur=0,
                             blur_level=0,
                             random_distortions=False,
                             distortion_sigma=0,
                             max_images_to_load=config["images_to_load"],
                             logger=config["logger"],
                             config=config,
                             **config.dataset)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config["batch_size"],
                                 shuffle=False,
                                 num_workers=threads,
                                 collate_fn=default_collate)

    if "validation_jsons" in config and config.validation_jsons: # must be present and non-empty
        validation_dataset = OnlineDataset(config["validation_jsons"],
                                       config["char_to_idx"],
                                       root=config["testing_root"],
                                       warp=False,
                                       blur=config.get("testing_blur", 1.5),
                                       blur_level=config["testing_blur_level"],
                                       random_distortions=config["testing_random_distortions"],
                                       distortion_sigma=config["testing_distortion_sigma"],
                                       max_images_to_load=config["images_to_load"],
                                       logger=config["logger"],
                                       config=config,
                                       **config.dataset)

        validation_dataloader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=config["testing_shuffle"],
                                           num_workers=threads, collate_fn=default_collate)
    else:
        validation_dataset, validation_dataloader = test_dataset, test_dataloader
        config["validation_jsons"]=None

    n_test_points = 0
    for i in test_dataloader:
        n_test_points += sum(i["label_lengths"])
    config.n_train_instances = len(train_dataloader.dataset)
    config.n_test_instances = len(test_dataloader.dataset)
    config.n_test_points = int(n_test_points)
    config.n_validation_instances = len(validation_dataloader.dataset)

    return train_dataloader, test_dataloader, train_dataset, test_dataset, validation_dataset, validation_dataloader


def load_data(config):
    # Load characters and prep datasets
    root = Path(os.path.dirname(os.path.realpath(__file__))).parent / config.training_root

    out_char_to_idx2, out_idx_to_char2, char_freq = character_set.make_char_set(
        config.training_jsons, root=root)
    # Convert to a list to work with easydict
    idx_to_char = dict_to_list(out_idx_to_char2)

    config.char_to_idx, config.idx_to_char, config.char_freq = out_char_to_idx2, idx_to_char, char_freq

    train_dataloader, test_dataloader, train_dataset, test_dataset, validation_dataset, validation_dataloader = make_dataloaders(config=config, device="cpu")

    config['alphabet_size'] = len(config["idx_to_char"])   # alphabet size to be recognized
        #config['num_of_writers'] = train_dataset.classes_count + 1

    logger.info("Number of training instances:", config['n_train_instances'])

    if config["validation_jsons"]:
        logger.info("Number of validation instances:", config.n_validation_instances)

    assert config['n_train_instances'] > 0

    logger.info("Number of test instances:", config.n_test_instances, '\n')
    return train_dataloader, test_dataloader, train_dataset, test_dataset, validation_dataset, validation_dataloader

def check_gpu(config):
    # GPU stuff
    use_gpu = torch.cuda.is_available() and config["gpu_if_available"]
    device = torch.device("cuda" if use_gpu else "cpu")
    dtype = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
    if use_gpu:
        logger.info("Using GPU")
    elif not torch.cuda.is_available():
        logger.info("No GPU found")
    elif not config["gpu_if_available"]:
        logger.info("GPU available, but not using per config")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    return device, dtype



def build_model(config_path):
    global config, LOGGER
    # Set GPU
    utils.choose_optimal_gpu()
    config = utils.load_config(config_path)
    LOGGER = config["logger"]
    config["global_step"] = 0
    config["global_instances_counter"] = 0
    device, dtype = check_gpu(config)

    # Use small batch size when using CPU/testing
    if config["TESTING"]:
        config["batch_size"] = 1

    # Prep data loaders
    LOGGER.info("Loading data...")
    train_dataloader, test_dataloader, train_dataset, test_dataset, validation_dataset, validation_dataloader = load_data(config)

    # for x in train_dataloader:
    #     print(x["labels"])
    #     print(x["label_lengths"])
    #     print(x['gt'])
    #     print(x['paths'])
    #     Stop

    # Decoder
    config["calc_cer_training"] = calculate_cer
    use_beam = config["decoder_type"] == "beam"
    config["decoder"] = Decoder(idx_to_char=config["idx_to_char"], beam=use_beam)

    # Prep optimizer
    if True:
        ctc = torch.nn.CTCLoss()
        log_softmax = torch.nn.LogSoftmax(dim=2).to(device)
        criterion = lambda x, y, z, t: ctc(log_softmax(x), y, z, t)
    else:
        from warpctc_pytorch import CTCLoss
        criterion = CTCLoss()

    LOGGER.info("Building model...")
    # Create classifier
    if config["style_encoder"] == "basic_encoder":
        hw = crnn.create_CRNNClassifier(config)
    elif config["style_encoder"] == "fake_encoder":
        hw = crnn.create_CRNNClassifier(config)
    elif config["style_encoder"] == "2Stage":
        hw = crnn.create_2Stage(config)
        config["embedding_size"] = 0
    elif config["style_encoder"] == "2StageNudger":
        hw = crnn.create_CRNN(config)
        config["nudger"] = crnn.create_Nudger(config).to(device)
        config["embedding_size"] = 0
        config["nudger_optimizer"] = torch.optim.Adam(config["nudger"].parameters(), lr=config['learning_rate'])
    elif config["style_encoder"] == "stroke":
        hw = crnn.create_stroke_CRNN(config)

    else:  # basic HWR
        config["embedding_size"] = 0
        hw = models_online.OnlineRecognizer(nIn=3, nHidden=64, vocab_size=config['alphabet_size'], ) #crnn.create_CRNN(config)

    LOGGER.info(f"Sending model to {device}...")
    hw.to(device)

    # Setup defaults
    defaults = {"starting_epoch":1,
                "model": hw,
                'lowest_loss':float('inf'),
                "criterion":criterion,
                "device":device,
                "dtype":dtype,
                }
    for k in defaults.keys():
        if k not in config.keys():
            config[k] = defaults[k]

    config["current_epoch"] = config["starting_epoch"]

    # Launch visdom
    if config["use_visdom"]:
        visualize.initialize_visdom(config["full_specs"], config)

    # Stat prep - must be after visdom
    stat_prep(config)

    # Create optimizer
    if config["optimizer_type"].lower() == "adam":
        optimizer = torch.optim.Adam(hw.parameters(), lr=config['learning_rate'])
    elif config["optimizer_type"].lower() == "sgd":
        optimizer = torch.optim.SGD(hw.parameters(), lr=config['learning_rate'], nesterov=True, momentum=.9)
    elif config["optimizer_type"].lower() == "adabound":
        from models import adabound
        optimizer = adabound.AdaBound(hw.parameters(), lr=config['learning_rate'])
    else:
        raise Exception("Unknown optimizer type")

    config["optimizer"] = optimizer

    scheduler = lr_scheduler.StepLR(optimizer, step_size=config["scheduler_step"], gamma=config["scheduler_gamma"])
    config["scheduler"] = scheduler

    ## LOAD FROM OLD MODEL
    if config["load_path"]:
        LOGGER.info("Loading old model...")
        load_model(config)
        hw = config["model"].to(device)
        # DOES NOT LOAD OPTIMIZER, SCHEDULER, ETC?


    LOGGER.info("Creating trainer...")
    # Create trainer
    if config["style_encoder"] == "2StageNudger":
        train_baseline = False if config["load_path"] else True
        config["trainer"] = crnn.TrainerNudger(hw, config["nudger_optimizer"], config, criterion,
                                               train_baseline=train_baseline)
    elif config["style_encoder"] == "stroke":
        config["trainer"] = trainer.TrainerStrokes(hw, optimizer, config, criterion)
    else:
        config["trainer"] = trainer.TrainerBaseline(hw, optimizer, config, criterion)

    # Alternative Models
    if config["style_encoder"] == "basic_encoder":
        config["secondary_criterion"] = CrossEntropyLoss()
    else:  # config["style_encoder"] = False
        config["secondary_criterion"] = None
    return config, train_dataloader, test_dataloader, train_dataset, test_dataset, validation_dataset, validation_dataloader

def main(opts):
    global config, LOGGER
    config, train_dataloader, test_dataloader, train_dataset, test_dataset, validation_dataset, validation_dataloader = build_model(opts.config)

    config.train_dataloader = train_dataloader
    config.validation_dataloader = validation_dataloader
    config.test_dataloader = test_dataloader

    if config["test_only"]:
        final_test(config, test_dataloader)
    # Actually train
    else:
        for epoch in range(config["starting_epoch"], config["starting_epoch"] + config["epochs_to_run"]):

            LOGGER.info("Epoch: {}".format(epoch))
            config["current_epoch"] = epoch

            training_cer = run_epoch(config["model"], train_dataloader, config["criterion"], config["optimizer"], config["dtype"], config)

            config["scheduler"].step()

            LOGGER.info("Training CER: {}".format(training_cer))
            #config["train_cer"].append(training_cer)

            # CER plot
            if config["current_epoch"] % config["TEST_FREQ"]== 0:
                validation_cer = validate(config["model"], validation_dataloader, config["idx_to_char"], config["device"], config)
                #config["validation_cer"].append(validation_cer)

            # Save periodically / save BSF
            if not config["results_dir"] is None and not config["SMALL_TRAINING"]:
                if epoch % config["save_freq"] == 0:
                    logger.info("Saving most recent model")
                    save_model(config, bsf=False)

                plt_loss(config)

        # Final test after everything (test with extra warps/transforms/beam search etc.)
        final_test(config, test_dataloader)

def final_test(config, test_dataloader):
    its = max(config['n_warp_iterations'], 11)
    collate_fn2 = lambda x: hw_dataset.collate(x, device=config.device,
                                               n_warp_iterations=its,
                                               warp=config["testing_warp"],
                                               occlusion_freq=0,
                                               occlusion_size=0,
                                               occlusion_level=0)
    test_dataloader.collate_fn = collate_fn2


    ## Do a final test WITH warping and plot all test images
    config["testing_warp"] = True
    test(config["model"], test_dataloader, config["idx_to_char"], config["device"], config, plot_all=True, validation=False, with_iterations=True)
    config["stats"][config["designated_test_cer"]].y[-1] *= -1 # shorthand

def recreate():
    """ Simple function to load model and re-save it with some updates (e.g. model definition etc.)

    Returns:

    """
    path = "./results/BEST/20190807_104745-smallv2/RESUME.yaml"
    path = "./results/BEST/LARGE/LARGE.yaml"
    # import shlex
    # args = shlex.split(f"--config {path}")
    # sys.argv[1:] = args
    # print(sys.argv)
    config, *_ = build_model(path)
    globals().update(locals())

    #save_model(config)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./configs/baseline.yaml", help='Path to the config file.')
    parser.add_argument('--testing', action="store_true", default=False, help='Run testing version')
    #parser.add_argument('--name', type=str, default="", help='Optional - special name for this run')
    opts = parser.parse_args()
    return opts

if __name__ == "__main__":
    #recreate()
    opts = parse_args()
    main(opts)
    Stop
    try:
        main()
    except Exception as e:
        logger.info(e)
        traceback.print_exc()
    finally:
        torch.cuda.empty_cache()



# https://github.com/theevann/visdom-save/blob/master/vis.py
