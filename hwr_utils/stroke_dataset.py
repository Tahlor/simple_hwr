import re
import json
import warnings
from matplotlib import pyplot as plt
import multiprocessing
import torch
from torch.utils.data import Dataset
from scipy.spatial import KDTree
import os
import cv2
import numpy as np
from tqdm import tqdm

from hwr_utils import stroke_recovery
from hwr_utils.utils import unpickle_it, npy_loader
import pickle
from pathlib import Path
import logging
from hwr_utils.utils import EnhancedJSONEncoder
from hwr_utils import distortions
from hwr_utils.stroke_plotting import draw_from_gt, random_pad

logger = logging.getLogger("root."+__name__)

PADDING_CONSTANT = 1 # 1=WHITE, 0=BLACK
MAX_LEN = 64
PARAMETER = "d" # t or d for time/distance resampling

script_path = Path(os.path.realpath(__file__))
project_root = script_path.parent.parent

def read_img(image_path, num_of_channels=1, target_height=61, resize=True, add_distortion=False, vertical_pad=False):
    """

    Args:
        image_path:
        num_of_channels:
        target_height:
        resize:
        add_distortion:
        pad: How many pixels of padding to apply

    Returns:

    """

    if isinstance(image_path, str):
        image_path = Path(image_path)

    if num_of_channels == 3:
        img = cv2.imread(image_path.as_posix())
    elif num_of_channels == 1:  # read grayscale
        img = cv2.imread(image_path.as_posix(), 0)
    else:
        raise Exception("Unexpected number of channels")
    if img is None:
        if not Path(image_path).exists():
            raise Exception(f"{image_path} does not exist")
        logging.warning(f"Warning: image is None: {image_path}")
        return None

    #vertical_pad = False # vertical pad makes the last predictions totally haywire!!
    if vertical_pad:
        target_height -= 4

    percent = float(target_height) / img.shape[0]
    if percent != 1 and resize:
        img = cv2.resize(img, (0, 0), fx=percent, fy=percent, interpolation=cv2.INTER_CUBIC)

    if vertical_pad:
        img = cv2.copyMakeBorder(img, 2, 2, 0, 0, cv2.BORDER_CONSTANT,value=[255,255,255])

    img = img.astype(np.float32)

    if add_distortion:
        img = add_unormalized_distortion(img)

    img = img / 127.5 - 1.0

    # Add channel dimension, since resize and warp only keep non-trivial channel axis
    if num_of_channels == 1:
        img = img[:, :, np.newaxis]

    return img

def add_unormalized_distortion(img):
    """ Converting image INT to image FLOAT. Image will be correct, but FLOATS need to be normalized or converted back to
     np.uint8/16 for PIL to interpret correctly!!!

    Args:
        img:

    Returns:

    """

    x = distortions.change_contrast(
            distortions.gaussian_noise(
            distortions.blur(
                distortions.random_distortions(img.astype(np.float32), noise_max=1.1), # this one can really mess it up, def no bigger than 2
                max_intensity=1.0),
            max_intensity=.30)
            )

    # Apply targeted gaussian on stroke only
    stroke_mask = x<100
    # plt.imshow(x, cmap="gray")
    # plt.show()

    x[stroke_mask] = distortions.gaussian_noise(x[stroke_mask], max_intensity=.8)
    # plt.imshow(x, cmap="gray")
    # plt.show()
    # from time import sleep
    # sleep(1)
    return x
    #return img.astype(np.float64) # this one can really mess it up, def no bigger than 2

def blur(img):
    x = distortions.blur(img.astype(np.float32), max_intensity=1.0)
    return x


def fake_gt():
    gt2 = np.tile(np.array([1, 2, 3, 8, 7, 6, -3, -2, -1]), (4, 1)).transpose()
    gt2[-1, 1] = 10
    gt2[6, 1] = 10

    gt2[:, 2] = [1, 0, 0, 1, 1, 0, 1, 0, 0]
    gt2[:, 3] = [0, 0, 0, 0, 0, 0, 0, 0, 1]
    return gt2


class BasicDataset(Dataset):
    """ The kind of dataset used for e.g. offline data. Just looks at images, and calculates the output size etc.

    """
    def __init__(self, root, extension=".png", cnn=None, pickle_file=None, adapted_gt_path=None, **kwargs):
        # Create dictionary with all the paths and some index
        root = Path(root)
        self.root = root
        self.data = []
        self.num_of_channels = 1
        self.collate = collate_stroke_eval

        self.cnn = cnn

        if not cnn is None and cnn.cnn_type:
            self.cnn_type = cnn.cnn_type
        else:
            self.cnn_type = "default64"

        if "contrast" in kwargs:
            self.contrast = kwargs["contrast"]

        if adapted_gt_path:
            print(f"LOADING FROM {adapted_gt_path}")
            self.data = np.load(adapted_gt_path, allow_pickle=True)

        else:
            if pickle_file is None and cnn:
                self.cnn_type = self.cnn.cnn_type
                output = Path(root / "stroke_cached")
                output.mkdir(parents=True, exist_ok=True)
                pickle_file = output / (self.cnn.cnn_type + ".pickle")
            if Path(pickle_file).exists():
                self.data = unpickle_it(pickle_file)
            else:
                print("Pickle not found, rebuilding")
                # Rebuild the dataset - find all PNG files
                for i in root.rglob("*" + extension):
                    self.data.append({"image_path":i.as_posix()})
                logger.info(("Length of data", len(self.data)))

                # Add label lengths - save to pickle
                if self.cnn:
                    add_output_size_to_data(self.data, self.cnn, key="label_length", root=self.root)
                    logger.info(f"DUMPING cached version to: {pickle_file}")
                    pickle.dump(self.data, pickle_file.open(mode="wb"))

    @staticmethod
    def get_item_from_path(image_path, output_path):
        """ This is used in the server right now?

        Args:
            image_path:
            output_path:

        Returns:

        """

        img = read_img(image_path, num_of_channels=1, vertical_pad=True)
        image_path = output_path
        # plt.imshow(img[:,:,0], cmap="gray")
        # plt.show()

        img = (distortions.change_contrast((img+1)*127.5, contrast=2)/ 127.5 - 1.0)[:,:,np.newaxis]
        # plt.imshow(img, cmap="gray")
        # plt.show()
        # STPO
        label_length = None
        return {
            "line_img": img,
            "gt": [],
            "path": image_path,
            "x_func": None,
            "y_func": None,
            "start_times": None,
            "x_relative": None,
            "label_length": label_length,
        }


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = self.root / item['image_path']
        img = read_img(image_path, num_of_channels=self.num_of_channels, vertical_pad=True)

        # plt.imshow(img[:,:,0], cmap="gray")
        # plt.show()

        img = (distortions.change_contrast((img+1)*127.5, contrast=2)/ 127.5 - 1.0)[:,:,np.newaxis]
        # plt.imshow(img, cmap="gray")
        # plt.show()
        # STPO
        label_length = item["label_length"] if self.cnn else None
        return {
            "line_img": img,
            "gt": [],
            "path": image_path,
            "x_func": None,
            "y_func": None,
            "start_times": None,
            "x_relative": None,
            "label_length": label_length,
        }


class StrokeRecoveryDataset(Dataset):
    def __init__(self,
                 data_paths,
                 img_height=61,
                 num_of_channels=3,
                 root= project_root / "data",
                 max_images_to_load=None,
                 gt_format=None,
                 cnn=None,
                 config=None,
                 image_prep="pil_with_distortions",
                 **kwargs):

        super().__init__()
        self.max_width = 2000

        if not cnn is None and cnn.cnn_type:
            self.cnn_type = cnn.cnn_type
        else:
            self.cnn_type = "default64"


        # Make it an iterable
        if isinstance(data_paths, str) or isinstance(data_paths, Path):
            data_paths = [data_paths]

        self.gt_format = ["x","y","stroke_number","eos"] if gt_format is None else gt_format
        self.collate = collate_stroke
        self.root = Path(root)
        self.num_of_channels = num_of_channels
        self.interval = .05
        self.noise = False
        self.cnn = cnn
        self.config = config
        self.img_height = img_height
        self.image_prep = image_prep
        self.test_dataset = False
        # This will override defaults above
        self.__dict__.update(kwargs)

        ### LOAD THE DATA LAST!!
        if "adapted_gt_path" in kwargs and kwargs["adapted_gt_path"] and "training" in kwargs:
            adapted_gt_path = kwargs["adapted_gt_path"]
            print(f"LOADING FROM {adapted_gt_path}")
            self.data = np.load(adapted_gt_path, allow_pickle=True)
        else:
            logger.info(f"Loading data traditional way {data_paths}")
            self.data = self.load_data(root, max_images_to_load, data_paths)
        logger.info(("Dataloader size", len(self.data)))

    def resample_one(self, item, parameter=PARAMETER):
        """ Resample will be based on time, unless the number of samples has been calculated;
                this is only calculated if you supply a pickle file or a CNN! In this case the number
                of stroke points corresponds to the image width in pixels. Otherwise:
                    * The number of stroke points corresponds to how long it took to write
                    OR
                    * If "scale_time_distance" was selected. the number of stroke points corresponds to how long
                    the strokes are
        Args:
            item: Dictionary with a "raw" dictionary item
        Returns:
            Adds/modifies the "gt" key

        """
        output = stroke_recovery.prep_stroke_dict(item["raw"])  # returns dict with {x,y,t,start_times,x_to_y (aspect ratio), start_strokes (binary list), raw strokes}
        x_func, y_func = stroke_recovery.create_functions_from_strokes(output, parameter=parameter) # can be d if the function should be a function of distance
        if "number_of_samples" not in item:
            item["number_of_samples"] = int(output[parameter+"range"] / self.interval)
            warnings.warn("UNK NUMBER OF SAMPLES!!!")

        if parameter == "t":
            start_times = output.start_times
        elif parameter == "d":
            start_times = output.start_distances
            item["start_distances"] = output.start_distances
        else:
            raise NotImplemented(f"Unknown {parameter}")

        gt = create_gts_from_fn(x_func, y_func, start_times=start_times,
                                number_of_samples=item["number_of_samples"],
                                noise=self.noise,
                                gt_format=self.gt_format)

        item["gt"] = gt  # {"x":x, "y":y, "is_start_time":is_start_stroke, "full": gt}
        item["x_func"] = x_func
        item["y_func"] = y_func
        return item

    def prepare_data(self, data_list, parallel=True, fn=None):
        fn = self.resample_one if fn is None else fn
        if parallel and False:
            poolcount = max(1, multiprocessing.cpu_count()-3)
            pool = multiprocessing.Pool(processes=poolcount)
            all_results = list(pool.imap_unordered(fn, tqdm(data_list)))  # iterates through everything all at once
            pool.close()
        else:
            all_results = []
            for item in data_list:
                all_results.append(fn(item))
        return all_results

    def load_data(self, root, images_to_load, data_paths):
        data = []
        for data_path in data_paths: # loop through JSONs
            data_path = str(data_path)
            print(os.path.join(root, data_path))
            new_data = npy_loader(os.path.join(root, data_path), save=True) # load .npy file if it exists

            if isinstance(new_data, dict):
                new_data = [item for key, item in new_data.items()]

            # Clean up to reduce memory footprint
            for i,d in enumerate(new_data):
                #"gt":new_data[i]["gt"],
                # "number_of_samples":item["number_of_samples"], "start_distances":item["start_distances"],
                item = new_data[i]
                new_data[i] = {"raw":item["raw"],
                               "shape":item["shape"],
                               "image_path":new_data[i]["image_path"]}

            data.extend(new_data)

        # Calculate how many points are needed
        if self.cnn:
            self.cnn_type = self.cnn.cnn_type
            add_output_size_to_data(data, self.cnn, root=self.root, max_width=self.max_width)
            self.cnn=True # remove CUDA-object from class for multiprocessing to work!!

        #print(data[0].keys())

        if images_to_load:
            logger.info(("Original dataloader size", len(data)))
            data = data[:images_to_load]
        logger.info(("Dataloader size", len(data)))

        if "gt" not in data[0].keys() and self.config.dataset.resample:
            data = self.prepare_data(data, parallel=True)
        else:
            data = self.prepare_data(data, parallel=True, fn=self.no_resample)
        logger.debug(("Done resampling", len(data)))
        return data

    def no_resample(self, item):
        stroke_dict = stroke_recovery.prep_stroke_dict(item["raw"])  # returns dict with {x,y,t,start_times,x_to_y (aspect ratio), start_strokes (binary list), raw strokes}
        gt = create_gts(stroke_dict.x, stroke_dict.y, is_start_stroke=stroke_dict.start_strokes,
                        gt_format=self.gt_format)
        item["gt"] = gt  # {"x":x, "y":y, "is_start_time":is_start_stroke, "full": gt}
        item["x_func"] = None
        item["y_func"] = None
        return item

    @staticmethod
    def shrink_gt(gt, height=61, width=None):
        if width:
            max_x = np.ceil(np.max(gt[:, 0]) * height)
            if max_x > width:
                gt[:, 0:2] = gt[:, 0:2] * width/max_x # multiply by ratio to shrink gts
        return gt

    @staticmethod
    def enlarge_gt(gt, height=61, width=None):
        if width:
            max_x = np.floor(np.max(gt[:, 0]) * height)
            if max_x < width:
                gt[:, 0:2] = gt[:, 0:2] * width/max_x # multiply by ratio to shrink gts
        return gt

    @staticmethod
    def prep_image(gt, img_height=61, add_distortion=True, add_blur=False, use_stroke_number=None, **kwargs):
        """ Important that this modifies the actual GT so that plotting afterward still works

        Randomly SQUEEZE OR STRETCH? Would have to change GT length...???
        Args:
            gt:
            img_height:

        Returns:

        """
        # Based on how the GTs were resampled, how big was the original image etc?
        image_width = gts_to_image_size(len(gt))
        # Returns image in upper origin format
        padded_gt_img = random_pad(gt,vpad=3, hpad=5) # pad top, left, bottom
        padded_gt_img = StrokeRecoveryDataset.shrink_gt(padded_gt_img, width=image_width) # shrink to fit
        # padded_gt = StrokeRecoveryDataset.enlarge_gt(padded_gt, width=image_width)  # enlarge to fit - needs to be at least as big as GTs

        img = draw_from_gt(padded_gt_img, show=False, save_path=None, min_width=None, height=img_height,
                           right_padding="random", max_width=8, use_stroke_number=use_stroke_number,
                           **kwargs)

        # img = img[::-1] # convert to lower origin format
        if add_distortion:
            img = add_unormalized_distortion(img)
        elif add_blur:
            img = blur(img)
        #from PIL import Image, ImageDraw
        #Image.fromarray(img.astype(np.uint8), 'L').show()

        # Normalize
        img = img / 127.5 - 1.0 # range: -1, 1

        # Add trivial channel dimension
        img = img[:, :, np.newaxis]

        return img, padded_gt_img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """ data[idx].keys() = 'full_img_path',
                                'xml_path',
                                'image_path',
                                'dataset',
                                'x', 'y', 't',
                                'start_times',
                                'start_strokes',
                                'x_to_y', 'raw',
                                'shape',
                                'number_of_samples',
                                'gt', 'x_func', 'y_func'])

        Args:
            idx:

        Returns:

        """
        # for i in range(200):
        #     if "a01-001z-04_2" in self.data[i]["image_path"]:
        #         print(i)
        #         break

        # Stroke order
        #idx = 27; print("IDX 27")
        item = self.data[idx]
        #print(item["gt"].shape)
        #assert item["gt"].shape[0]==52

        # if not "current_stroke_order" in item:
        #     item["current_stroke_order"] = list(range(np.sum(item["gt"][:,2])))
        #     item["new_stroke_order"] = item["current_stroke_order"]
        # if not np.allclose(item["new_stroke_order"], item["current_stroke_order"]):
        #     # Swap to get current_stroke_order to look like new_stroke_order

        image_path = self.root / item['image_path']

        ## DEFAULT GT ARRAY
        # X, Y, FLAG_BEGIN_STROKE, FLAG_END_STROKE, FLAG_EOS - VOCAB x desired_num_of_strokes
        if self.image_prep.startswith("pil") and not ("no_warp" in self.image_prep):
            if True:
                gt = item["gt"].copy() # LENGTH, VOCAB
            if not self.test_dataset: # don't warp the test data
                gt = distortions.warp_points(gt * self.img_height) / self.img_height  # convert to pixel space
                gt = np.c_[gt,item["gt"][:,2:]]

        else:
            gt = item["gt"].copy()

        assert gt.shape[0] == item["gt"].shape[0]

        # Render image
        add_distortion = "distortion" in self.image_prep.lower() and not self.test_dataset # don't distort the test data
        add_blur = "blur" in self.image_prep.lower()
        if self.image_prep.lower().startswith("pil"):
            img, gt = self.prep_image(gt, img_height=self.img_height,
                                      add_distortion=add_distortion,
                                      add_blur=add_blur,
                                      use_stroke_number=("stroke_number" in self.gt_format),
                                      linewidth=None if self.config.dataset.linewidth is None else self.config.dataset.linewidth,
                                      )
        else:
            # Check if the image is already loaded
            if "line_img" in item and not add_distortion:
                img = item["line_img"]
                assert not img is None
            else:
                # Maybe delete this option
                # The GTs will be the wrong size if the image isn't resized the same way as earlier
                # Assuming e.g. we pass everything through the CNN every time etc.
                img = read_img(image_path, add_distortion=add_distortion)
                assert not img is None
        #gt_reverse_strokes, sos_args = stroke_recovery.invert_each_stroke(gt)
        gt_reverse_strokes = None
        sos_args = stroke_recovery.get_sos_args(gt[:, 2], stroke_numbers=True)

        # Need to convert to relative AFTER warping etc.
        for x in ("x_rel", "y_rel"):
            if x in self.gt_format: # can't do this
                idx = self.gt_format.index(x)
                gt[:,idx] = stroke_recovery.relativefy(gt[:,idx])

        # Assumes dimension 2 is start points, 3 is EOS
        # START POINT MODEL
        if False and gt.shape[-1] > 3:
            #start_points = gt[np.logical_or(gt[:, 2] > 0, gt[:, 3] > 0)][:MAX_LEN] # JUST LEAVE THE SOS's in

            # Logic to get parallel SOS and EOS
            sos = gt[:, 2]
            eos = stroke_recovery.get_eos_from_sos(sos)
            start_points = gt[np.logical_or(sos > 0, eos > 0)][:MAX_LEN]

            # Find things that are both start and end points
            s = np.argwhere(sos + eos > 1).reshape(-1)
            if s.size: # duplicate here so later loss function works correctly
                sos = start_points[:, 2]
                eos = stroke_recovery.get_eos_from_sos(sos)
                s = np.argwhere(sos + eos > 1).reshape(-1)
                if True: # make it so no points are both start and stop
                    start_points[s, 2] = 0
                    replacement = start_points[s]
                    replacement[:, 2] = 1
                else:
                    replacement = [start_points[s]]

                start_points = np.insert(start_points, s, replacement, 0)[:MAX_LEN]
            # try:
            #     width = start_points.shape[0]
            #     assert width % 2 == 0 # for every start point, there is an end point
            #     assert np.sum(start_points[:,2]) == width / 2
            # except:
            #     print("FAILED", start_points[:,2])

        else:
            start_points = np.array([])

        kdtree = KDTree(gt[:, 0:2]) if self.config and "nnloss" in [loss["name"] for loss in self.config.loss_fns] else None

        np.testing.assert_allclose(item["gt"].shape, gt.shape)

        return {
            "line_img": img, # H,W,C
            "gt": gt, # B, W, 3/4
            "gt_reverse_strokes": gt_reverse_strokes,
            "sos_args": sos_args,
            "path": image_path,
            "x_func": item["x_func"] if "x_func" in item else None,
            "y_func": item["y_func"]  if "y_func" in item else None,
            "gt_format": self.gt_format,
            "start_points": start_points,
            "kdtree": kdtree, # Will force preds to get nearer to nearest GTs; really want GTs forced to nearest pred; this will finish strokes better
            "gt_idx": idx,
            "predicted_strokes_gt": None,
            "feature_map_width": gt.shape[1] #img_width_to_pred_mapping(img.shape[1], self.cnn_type)
        }

def create_gts_from_raw_dict(item, interval, noise, gt_format=None):
    """
    Args:
        item: Dictionary with a "raw" item
    Returns:

    """
    output = stroke_recovery.prep_stroke_dict(item) # returns dict with {x,y,t,start_times,x_to_y (aspect ratio), start_strokes (binary list), raw strokes}
    x_func, y_func = stroke_recovery.create_functions_from_strokes(output)
    number_of_samples = int(output.trange/interval)
    return create_gts_from_fn(x_func, y_func, output.start_times,
                              number_of_samples=number_of_samples,
                              noise=noise,
                              gt_format=gt_format)

def create_gts(x,y,is_start_stroke,gt_format):
    # Make coordinates relative to previous one
    # x = stroke_recovery.relativefy(x)

    # Put it together
    gt = []
    padding_constant = []
    assert not np.isnan(x).any()
    assert not np.isnan(y).any()

    for i,el in enumerate(gt_format):
        if el.startswith("x"):
            gt.append(x)
        elif el.startswith("y"):
            gt.append(y)
        elif el == "sos":
            gt.append(is_start_stroke)
        elif el == "eos":
            # Create GT matrix
            end_of_sequence_flag = np.zeros(x.shape[0])
            end_of_sequence_flag[-1] = 1

            gt.append(end_of_sequence_flag)
        elif "sos_interp" in el:
            # Instead of using 1 to denote start of stroke, use 0, increment for each additional stroke based on distance of stroke
            is_start_stroke = stroke_recovery.get_stroke_length_gt(x, y, is_start_stroke, use_distance=(el=="sos_interp_dist"))
            gt.append(is_start_stroke)
        elif el == "stroke_number": # i.e. 1,1,1,1,1,2,2,2,2,2...
            stroke_number = np.cumsum(is_start_stroke)
            gt.append(stroke_number)
        else:
            raise Exception(f"Unkown GT format: {el}")

    gt = np.array(gt).transpose([1,0]) # swap axes -> WIDTH, VOCAB

    # Rearrange strokes and reverse strokes as needed - ASSUMES X,Y,SOS/Strokenumbers
    # gt = reorder_strokes(gt, stroke_numbers=("stroke_number" in gt_format), sos_index=2)

    # print(gt)
    # draw_from_gt(gt, show=True)
    # input()
    # stop
    assert not np.isnan(gt).any()

    return gt


def create_gts_from_fn(x_func, y_func, start_times, number_of_samples, gt_format, noise=None):
    """ Return LENGTH X VOCAB

    Args:
        x_func:
        y_func:
        start_times: [.1,1.2,3.6,...]
        number_of_samples: Number of GT points
        noise: Add some noise to the sampling
        start_of_stroke_method: "normal" - use 1's for starts
                              "interpolated" - use 0's for starts, 1's for ends, and interpolate
                              "interpolated_distance" - use 0's for starts, total distance for ends, and interpolate

    Returns:
        gt array: SEQ_LEN x [X, Y, IS_STROKE_START, IS_END_OF_SEQUENCE]
    """
    # [{'el': 'x', 'opts': None}, {'el': 'y', 'opts': None}, {'el': 'stroke_number', 'opts': None}, {'el': 'eos', 'opts': None}]

    # Sample from x/y functions
    if False:
        x, y, is_start_stroke = stroke_recovery.sample(x_func, y_func, start_times,
                                                       number_of_samples=number_of_samples,
                                                       noise=noise)
    else:
        x, y, is_start_stroke = stroke_recovery.sample(x_func, y_func, start_times[:-1],
                                                       number_of_samples=number_of_samples,
                                                       noise=noise,
                                                       last_time=start_times[-1])

    return create_gts(x, y, is_start_stroke, gt_format)


# def sos_filtered(x,y,is_start_stroke, end_of_sequence_flag):
#     gt.append(x)
#     gt.append(y)
#     gt.append(is_start_stroke)
#     gt.append(end_of_sequence_flag)
#
#         #     stroke_number = np.cumsum(is_start_stroke)
#         #     gt.append(stroke_number)
#         start_points = np.r_[gt[gt[:,2]>0],gt[gt[:,3]>0]][:MAX_LEN] # could end with the EOS point 2x if also a start stroke, NBD


def put_at(start, stop, axis=1):
    if axis < 0:
        return (Ellipsis, ) + (slice(start, stop),) + (slice(None),) * abs(-1-axis)
    else:
        return (slice(None),) * (axis) + (slice(start,stop),)


def calculate_output_size(data, cnn):
    """ For each possible width, calculate the CNN output width
    Args:
        data:

    Returns:

    """
    all_possible_widths = set()
    for i in data:
        all_possible_widths.add(i)

    width_to_output_mapping={}
    for i in all_possible_widths:
        t = torch.zeros(1, 1, 32, i)
        shape = cnn(t).shape
        width_to_output_mapping[i] = shape[-1]
    return width_to_output_mapping


default64_base = lambda width: -(width % 2) + width + 4
def img_width_to_pred_mapping(width, cnn_type="default64"):
    # If using default
    if cnn_type=="default":
        return int(width / 4) + 1
    elif cnn_type in ("default64", "FAKE"):
        return default64_base(width)
    elif cnn_type == "default128":
        return default64_base(width)*2
    elif cnn_type == "default96":
        return int(default64_base(width)*1.5)
    elif not cnn_type: # No CNN type specified
        return None
    else:
        raise Exception(f"Unknown CNN type {cnn_type}")


def add_output_size_to_data(data, cnn, key="number_of_samples", root=None, img_height=61, max_width=2000, force_redo=False):
    """ IMAGE SIZE TO NUMBER OF GTs
    """
    bad_indicies = []
    for i, instance in enumerate(data):
        image_path = root / instance['image_path']

        if not "shape" in instance or force_redo: # HEIGHT, WIDTH, CHANNEL? 61x4037x3
            try:
                img = read_img(image_path)
                instance["shape"] = img.shape
            except:
                print("Failed", image_path)
                bad_indicies.append(i)
                instance["shape"] = [0, 0]
        width = instance["shape"][1]
        if width > max_width:
            print("Too wide", width, image_path)
            bad_indicies.append(i)
        instance[key] = img_width_to_pred_mapping(width, cnn.cnn_type)

    for index in sorted(bad_indicies, reverse=True):
        print("Deleting bad file: ", index)
        del data[index]

    #return width_to_output_mapping

def gts_to_image_size(gt_length):
    """ GTs to image size
    """
    return gt_length - 3 # should be -3

## Hard coded -- ~20% faster
# def pad3(batch, variable_width_dim=1):
#     dims = list(batch[0].shape)
#     max_length = max([b.shape[variable_width_dim] for b in batch])
#     dims[variable_width_dim] = max_length
#
#     input_batch = np.full((len(batch), *dims), PADDING_CONSTANT).astype(np.float32)
#
#     for i in range(len(batch)):
#         b_img = batch[i]
#         img_length = b_img.shape[variable_width_dim]
#         input_batch[i][ :, :img_length] = b_img
#     return input_batch

## Variable, foot loop based
def pad(batch, variable_width_dim=1):
    """ Outer dimension asumed to be batch, variable width dimension excluding batch

    Put at could kind of be moved outside of the loop
    Args:
        batch:
        variable_width_dim:

    Returns:

    """
    dims = list(batch[0].shape)
    max_length = max([b.shape[variable_width_dim] for b in batch])
    dims[variable_width_dim] = max_length
    input_batch = np.full((len(batch), *dims), PADDING_CONSTANT).astype(np.float32)

    for i in range(len(batch)):
        b_img = batch[i]
        img_length = b_img.shape[variable_width_dim]
        input_batch[i][put_at(0, img_length, axis=variable_width_dim)] = b_img
    return input_batch

def test_padding(pad_list, func):
    start = timer()
    for m in pad_list:
        x = func(m)
    # print(x.shape)
    # print(x[-1,-1])
    end = timer()
    logger.info(end - start)  # Time in seconds, e.g. 5.38091952400282
    return x #[0,0]

TYPE = np.float32 #np.float16
def collate_stroke(batch, device="cpu", gt_opts=None, post_length_buffer=20):
    """ Pad ground truths with 0's
        Report lengths to get accurate average loss

        stroke_points_gt : padded with repeated last point
        stroke_points_rel : x_rel, abs_y, SOS, 0's

    Args:
        batch:
        device:

    Returns:

    """
    vocab_size = batch[0]['gt'].shape[-1]
    batch = [b for b in batch if b is not None]
    #These all should be the same size or error
    if len(set([b['line_img'].shape[0] for b in batch])) > 1: # All items should be the same height!
        logger.warning("Problem with collating!!! See hw_dataset.py")
        logger.info(batch)
    assert len(set([b['line_img'].shape[0] for b in batch])) == 1
    assert len(set([b['line_img'].shape[2] for b in batch])) == 1

    batch_size = len(batch)
    dim0 = batch[0]['line_img'].shape[0] # height
    dim1 = max([b['line_img'].shape[1] for b in batch]) # width
    dim2 = batch[0]['line_img'].shape[2] # channel

    max_feature_map_size = max([b['feature_map_width'] for b in batch])
    all_labels_numpy = []
    label_lengths = []
    start_points = []

    # Make input square BATCH, H, W, CHANNELS
    imgs_gt = np.full((batch_size, dim0, dim1, dim2), PADDING_CONSTANT).astype(TYPE)
    max_label = max([b['gt'].shape[0] for b in batch]) # width

    stroke_points_gt = np.full((batch_size, max_label, vocab_size), PADDING_CONSTANT).astype(TYPE)
    stroke_points_rel = np.full((batch_size, max_label+1, vocab_size), 0).astype(TYPE)
    mask = np.full((batch_size, max_label, 1), 0).astype(TYPE)
    feature_map_mask = np.full((batch_size, max_feature_map_size), 0).astype(TYPE)

    # Loop through instances in batch
    for i in range(len(batch)):
        b_img = batch[i]['line_img']
        imgs_gt[i,:,: b_img.shape[1],:] = b_img

        l = batch[i]['gt']
        #all_labels.append(l)
        label_lengths.append(len(l))
        ## ALL LABELS - list of desired_num_of_strokes batch size; arrays LENGTH, VOCAB SIZE
        stroke_points_gt[i, :len(l), :] = l
        stroke_points_gt[i, len(l):, :] = l[-1] # just repeat the last element; this works when using ABS coords for GTs (default) and EOS

        # Relative version - this is 1 indx longer - first one is 0's
        rel_x = stroke_recovery.relativefy_numpy(l[:,0:1])
        stroke_points_rel[i, 1:1+len(l), 0] = rel_x # use relative coords for X, then 0's
        stroke_points_rel[i, 1:1+len(l), 1:2] = stroke_points_gt[i, :len(l), 1:2] # Copy the absolute ones for Y, then 0's
        stroke_points_rel[i, batch[i]['sos_args']+1, 2] = 1 # all 0's => 1's where SOS are
        # No EOS specified for x_rel

        mask[i, :len(l), 0] = 1
        feature_map_mask[i, :batch[i]['feature_map_width']+post_length_buffer] = 1 # keep predicting after

        all_labels_numpy.append(l)
        start_points.append(torch.from_numpy(batch[i]['start_points'].astype(TYPE)).to(device))

    label_lengths = np.asarray(label_lengths)

    line_imgs = imgs_gt.transpose([0,3,1,2]) # batch, channel, h, w
    # print(np.min(line_imgs))
    # plt.hist(line_imgs.flatten())
    # plt.show()

    line_imgs = torch.from_numpy(line_imgs).to(device)
    stroke_points_gt = torch.from_numpy(stroke_points_gt.astype(TYPE)).to(device)
    #label_lengths = torch.from_numpy(label_lengths.astype(np.int32)).to(device)
    stroke_points_gt_rel = torch.from_numpy(stroke_points_rel.astype(TYPE)).to(device)

    mask = torch.from_numpy(mask.astype(TYPE)).to(device)
    feature_map_mask = torch.from_numpy(feature_map_mask.astype(TYPE)).to(device)
    return_d = {
        "feature_map_mask": feature_map_mask,
        "mask": mask,
        "line_imgs": line_imgs,
        "gt": stroke_points_gt, # Numpy Array, with padding
        "rel_gt": stroke_points_gt_rel,
        "gt_list": [torch.from_numpy(l.astype(TYPE)).to(device) for l in all_labels_numpy], # List of numpy arrays
        #"gt_reverse_strokes": [torch.from_numpy(b["gt_reverse_strokes"].astype(TYPE)).to(device) for b in batch],
        "gt_numpy": all_labels_numpy,
        "start_points": start_points,  # List of numpy arrays
        "gt_format": [batch[0]["gt_format"]]*batch_size,
        "label_lengths": label_lengths,
        "paths": [b["path"] for b in batch],
        "x_func": [b["x_func"] for b in batch],
        "y_func": [b["y_func"] for b in batch],
        "kdtree": [b["kdtree"] for b in batch],
        "gt_idx": [b["gt_idx"] for b in batch]
    }

    # Pass everything else through too
    for i in batch[0].keys():
        if i not in return_d.keys():
            return_d[i] = [b[i] for b in batch]

    return return_d


def collate_stroke_eval(batch, device="cpu"):
    """ Pad ground truths with 0's
        Report lengths to get accurate average loss

    Args:
        batch:
        device:

    Returns:

    """

    batch = [b for b in batch if b is not None]
    #These all should be the same size or error
    if len(set([b['line_img'].shape[0] for b in batch])) > 1: # All items should be the same height!
        logger.warning("Problem with collating!!! See hw_dataset.py")
        logger.warning(batch)
    assert len(set([b['line_img'].shape[0] for b in batch])) == 1
    assert len(set([b['line_img'].shape[2] for b in batch])) == 1

    batch_size = len(batch)
    dim0 = batch[0]['line_img'].shape[0] # height
    dim1 = max([b['line_img'].shape[1] for b in batch]) # width
    dim2 = batch[0]['line_img'].shape[2] # channel

    # Make input square (variable vidwth
    input_batch = np.full((batch_size, dim0, dim1, dim2), PADDING_CONSTANT).astype(np.float32)

    for i in range(len(batch)):
        b_img = batch[i]['line_img']
        input_batch[i,:,: b_img.shape[1],:] = b_img

    line_imgs = input_batch.transpose([0,3,1,2]) # batch, channel, h, w
    line_imgs = torch.from_numpy(line_imgs).to(device)

    return {
        "line_imgs": line_imgs,
        "gt": None,
        "gt_list": None, # List of numpy arrays
        "x_relative": None,
        "label_lengths": [b["label_length"] for b in batch],
        "paths": [b["path"] for b in batch],
        "x_func": None,
        "y_func": None,
        "start_times": None,
    }

def some_kind_of_test():
    # x = [np.array([[1,2,3,4],[4,5,3,5]]),np.array([[1,2,3],[4,5,3]]),np.array([[1,2],[4,5]])]
    from timeit import default_timer as timer

    vocab = 4
    iterations = 100
    batch = 32
    min_length = 32
    max_length = 64

    the_list = []

    for i in range(0,iterations): # iterations
        sub_list = []
        for m in range(0,batch): # batch size
            length = np.random.randint(min_length, max_length)
            sub_list.append(np.random.rand(vocab, length))
        the_list.append(sub_list)

    #test_padding(the_list, pad)
    x = test_padding(the_list, pad)
    # y = test_padding(the_list, pad2)
    # assert np.allclose(x,y)


if __name__=="__main__":
    kwargs = {'img_height': 121, 'include_synthetic': False, 'num_of_channels': 1, 'image_prep': 'no_warp_distortion', 'gt_format': ['x', 'y', 'stroke_number'], 'batch_size': 28, 'extra_dataset': []}
    dataset = StrokeRecoveryDataset(data_paths=['online_coordinate_data/ICDAR/train_online_coords.json'],
                                    root="../data",
                                    max_images_to_load = 10,
                                    cnn=None,
                                    **kwargs)
    for i in dataset:
        continue