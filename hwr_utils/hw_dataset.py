import sys
sys.path.append("..")

import re
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from easydict import EasyDict as edict
import os
import cv2
import numpy as np

from hwr_utils import distortions, string_utils
from hwr_utils.utils import unpickle_it

## FOR STROKE BUSINESS
LOADSTROKES = True
if LOADSTROKES:
    from hwr_utils.stroke_recovery import *
    from hwr_utils.stroke_dataset import *

PADDING_CONSTANT = 0
ONLINE_JSON_PATH = ''

def collate(batch, device="cpu", n_warp_iterations=None, warp=True, occlusion_freq=None, occlusion_size=None, occlusion_level=1):
    if n_warp_iterations:
        #print("USING COLLATE WITH REPETITION")
        return collate_repetition(batch, device, n_warp_iterations, warp, occlusion_freq, occlusion_size, occlusion_level=occlusion_level)
    else:
        return collate_basic(batch, device)

def collate_basic(batch, device="cpu"):
    batch = [b for b in batch if b is not None]
    #These all should be the same size or error
    if len(set([b['line_img'].shape[0] for b in batch])) > 1:
        print("Problem with collating!!! See hw_dataset.py")
        print(batch)
    assert len(set([b['line_img'].shape[0] for b in batch])) == 1
    assert len(set([b['line_img'].shape[2] for b in batch])) == 1

    dim0 = batch[0]['line_img'].shape[0] # HEIGHT
    dim1 = max([b['line_img'].shape[1] for b in batch]) # WIDTH
    dim2 = batch[0]['line_img'].shape[2] # CHANNEL

    all_labels = []
    label_lengths = []

    input_batch = np.full((len(batch), dim0, dim1, dim2), PADDING_CONSTANT).astype(np.float32)
    for i in range(len(batch)):
        b_img = batch[i]['line_img']
        input_batch[i,:,:b_img.shape[1],:] = b_img

        l = batch[i]['gt_label']
        all_labels.append(l)
        label_lengths.append(len(l))


    all_labels = np.concatenate(all_labels)
    label_lengths = np.array(label_lengths)

    line_imgs = input_batch.transpose([0,3,1,2]) # batch, channel, h, w
    line_imgs = torch.from_numpy(line_imgs).to(device)
    labels = torch.from_numpy(all_labels.astype(np.int32)).to(device)
    label_lengths = torch.from_numpy(label_lengths.astype(np.int32)).to(device)
    online = torch.from_numpy(np.array([1 if b['online'] else 0 for b in batch])).float().to(device)

    out = {
        "line_imgs": line_imgs,
        "labels": labels,
        "label_lengths": label_lengths,
        "gt": [b['gt'] for b in batch],
        "writer_id": torch.FloatTensor([b['writer_id'] for b in batch]),
        "actual_writer_id": torch.FloatTensor([b['actual_writer_id'] for b in batch]),
        "paths": [b["path"] for b in batch],
        "online": online
    }

    # STROKE
    if "strokes" in batch[0].keys() and not batch[0]['strokes'] is None:
        stroke_batch = np.full((len(batch), img_width_to_pred_mapping(dim1, cnn_type="default64"), 3),
                               PADDING_CONSTANT).astype(np.float32)
        for i in range(len(batch)):
            # STROKE
            b_strokes = batch[i]['strokes']  # W x 3
            stroke_batch[i, :b_strokes.shape[0], :] = b_strokes

        out["strokes"] = torch.from_numpy(stroke_batch).to(device)
    return out

def collate_repetition(batch, device="cpu", n_warp_iterations=21, warp=True, occlusion_freq=None, occlusion_size=None, occlusion_level=1):
    """ Returns multiple versions of the same item for n_warping

    Args:
        batch:
        device:
        n_warp_iterations:
        warp:
        occlusion_freq:
        occlusion_size:
        occlusion_level:

    Returns:

    """
    batch = [b for b in batch if b is not None]
    batch_size = len(batch)
    occlude = occlusion_size and occlusion_freq

    # These all should be the same size or error
    if len(set([b['line_img'].shape[0] for b in batch])) > 1:
        print("Problem with collating!!! See collate_repetition in hw_dataset.py")
        print(batch)
    assert len(set([b['line_img'].shape[0] for b in batch])) == 1
    assert len(set([b['line_img'].shape[2] for b in batch])) == 1

    dim0 = batch[0]['line_img'].shape[0] # height
    dim1 = max([b['line_img'].shape[1] for b in batch]) # width
    dim2 = batch[0]['line_img'].shape[2] # channel

    all_labels = []
    label_lengths = []
    final = np.full((batch_size, n_warp_iterations, dim0, dim1, dim2), PADDING_CONSTANT).astype(np.float32)

    # Duplicate items in batch
    for b_i,x in enumerate(batch):
        # H, W, C
        img = (np.float32(x['line_img']) + 1) * 128.0

        width = img.shape[1]
        for r_i in range(n_warp_iterations):
            new_img = img.copy()
            if warp:
                new_img = distortions.warp_image(new_img)
            if occlude:
                new_img = distortions.occlude(new_img, occlusion_freq=occlusion_freq, occlusion_size=occlusion_size, occlusion_level=occlusion_level)


            # Add channel dimension, since resize and warp only keep non-trivial channel axis
            if len(new_img.shape) == 2:
                new_img = new_img[:, :, np.newaxis]

            new_img = (new_img.astype(np.float32) / 128.0 - 1.0) # H, W, C
            final[b_i, r_i, :, :width, :] = new_img

        l = batch[b_i]['gt_label']
        all_labels.append(l)
        label_lengths.append(len(l))

    all_labels = np.concatenate(all_labels)
    label_lengths = np.array(label_lengths)

    line_imgs = final.transpose([0,1,4,2,3]) # batch, repetitions, channel, h, w
    line_imgs = torch.from_numpy(line_imgs).to(device)
    labels = torch.from_numpy(all_labels.astype(np.int32)).to(device)
    label_lengths = torch.from_numpy(label_lengths.astype(np.int32)).to(device)
    online = torch.from_numpy(np.array([1 if b['online'] else 0 for b in batch])).float().to(device)

    return {
        "line_imgs": line_imgs,
        "labels": labels,
        "label_lengths": label_lengths,
        "gt": [b['gt'] for b in batch],
        "writer_id": torch.FloatTensor([b['writer_id'] for b in batch]),
        "actual_writer_id": torch.FloatTensor([b['actual_writer_id'] for b in batch]),
        "paths": [b["path"] for b in batch],
        "online": online,
        "strokes": [b["strokes"] for b in batch]
    }


class HwDataset(Dataset):
    def __init__(self,
                 data_paths,
                 char_to_idx,
                 img_height=32,
                 num_of_channels=3,
                 root="./data",
                 warp=False,
                 blur=False, blur_level=1.5,
                 random_distortions=False, distortion_sigma = 6.0,
                 writer_id_paths=("prepare_IAM_Lines/writer_IDs.pickle",),
                 max_images_to_load=None,
                 occlusion_size=None, occlusion_freq=None, occlusion_level=1,
                 elastic_distortion=True, elastic_alpha=2.5, elastic_sigma=1.1,
                 logger=None,
                 **kwargs):

        self.load_strokes_flag = True if LOADSTROKES and "loadstrokes" in kwargs and kwargs["loadstrokes"] else False
        self.stroke_path = kwargs["stroke_path"] if "stroke_path" in kwargs else "RESULTS/OFFLINE_PREDS/good/imgs/current/eval/data/all_data.npy"

        data = self.load_data(data_paths, root, images_to_load=max_images_to_load)

        if self.load_strokes_flag:
            self.stroke_dict = self.load_strokes(file_path=self.stroke_path)

        # Data
        # {'gt': 'A MOVE to stop Mr. Gaitskell from',
        #  'image_path': 'prepare_IAM_Lines/lines/a01/a01-000u/a01-000u-00.png', 'err': False,
        #  'author_avg_std': 13.156031815424305, 'author_id': '000'}

        ## Read in all writer IDs
        writer_id_dict = self.join_writer_ids(root, writer_id_paths)

        data, self.classes_count = self.add_writer_ids(data, writer_id_dict)

        self.root = root
        self.img_height = img_height
        self.char_to_idx = char_to_idx
        self.data = data
        self.warp = warp
        self.blur = blur
        self.blur_level = blur_level
        self.random_distortions = random_distortions
        self.distortion_sigma = distortion_sigma
        self.num_of_channels = num_of_channels
        self.occlusion = not (None in (occlusion_size, occlusion_freq))
        self.occlusion_freq = occlusion_freq
        self.occlusion_size = occlusion_size
        self.occlusion_level = occlusion_level
        self.elastic_distortion = elastic_distortion
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma

        self.logger = logger

    @staticmethod
    def load_data(data_paths, root="", images_to_load=None):
        """

        Args:
            root (str):
            data_paths (list): list of str etc.
            images_to_load (int): how many to load

        Returns:

        """
        data = []
        root = Path(root)
        for data_path in data_paths:
            print("Loading JSON: ", data_path)
            with (root / data_path).open(mode="r") as fp:
                new_data = json.load(fp)
                if isinstance(new_data, dict):
                    new_data = [item for key, item in new_data.items()]
                #print(new_data[:100])
                data.extend(new_data)

        if images_to_load:
            data = data[:images_to_load]
        print("Dataloader size", len(data))
        return data

    def join_writer_ids(self, root, writer_id_paths):
        writer_id_dict = {}
        for writer_id_file in writer_id_paths:
            try:
                path = os.path.join(root, writer_id_file)
                # Add files to create super dicitonary
                d = unpickle_it(path)
                writer_id_dict = {**writer_id_dict, **d}
            except:
                print("Error with writer IDs")
        return writer_id_dict

    def add_writer_ids(self, data, writer_dict):
        """

        Args:
            data (json type thing): hw-dataset_
            writer_id_path (str): Path to pickle dictionary of form {Writer_ID: [file_path1,file_path2...] ... }

        Returns:
            tuple: updated data with ID, number of classes
        """
        actual_ids = dict([[v, k] for k, vs in writer_dict.items() for v in vs ]) # {Partial path : Writer ID}
        writer_ids = dict([[v, k] for k, (_, vs) in enumerate(writer_dict.items()) for v in vs])  # {Partial path : IDX}

        for i,item in enumerate(data):
            # Get writer ID from file
            try:
                p,child = os.path.split(item["image_path"])
                child = re.search("([a-z0-9]+-[0-9]+)", child).group(1)[0:7] # take the first 7 characters
                item["actual_writer_id"] = actual_ids[child]
                item["writer_id"] = writer_ids[child]
                data[i] = item
            except:
                item["actual_writer_id"] = -1
                item["writer_id"] = -1
                data[i] = item

        return data, len(set(writer_dict.keys())) # returns dictionary and number of writers

    def __len__(self):
        return len(self.data)


    def read_img(self, image_path):
        if self.num_of_channels == 3:
            img = cv2.imread(image_path)
        elif self.num_of_channels == 1: # read grayscale
            img = cv2.imread(image_path, 0)
        else:
            raise Exception("Unexpected number of channels")

        if img is not None:
            percent = float(self.img_height) / img.shape[0]
            #if random.randint(0, 1):
            #    img = cv2.resize(img, (0,0), fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)
            #else:
            img = cv2.resize(img, (0, 0), fx=percent, fy=percent, interpolation=cv2.INTER_CUBIC)

        return img

    def load_strokes(self, file_path):
        """

        Args:
            file_path:

        Returns:
                Lookup ID, return the strokes
        """
        stroke_list = np.load(file_path, allow_pickle=True)
        output_dict = {}
        for i in stroke_list:
            output_dict[i["id"]] = {"stroke": i["stroke"]}
        return output_dict

    def process_stroke(self, img_id, img_width, cnn_type="default64"):
        cnn_embedding_width = img_width_to_pred_mapping(img_width, cnn_type)

        # output_dict[i["id"]] = {"stroke": i["stroke"]}
        if img_id in self.stroke_dict:
            item = self.stroke_dict[img_id]
            output_dict = edict()
            output_dict.x = np.r_[item["stroke"][:, 0],item["stroke"][-1:, 0]]  # repeat last element
            output_dict.y = np.r_[item["stroke"][:, 1],item["stroke"][-1:, 1]]
            sos =           np.r_[item["stroke"][:, 2],1]
            output_dict.d = reparameterize_as_func_of_distance(output_dict.x, output_dict.y, sos)
            sos[0] = 1 # first one must be a start stroke
            start_times = output_dict.d[np.argwhere(np.round(sos))]
            x_func, y_func = stroke_recovery.create_functions_from_strokes(output_dict, parameter="d")
            x, y, is_start_stroke = stroke_recovery.sample(x_func, y_func, start_times,
                                                           number_of_samples=cnn_embedding_width, noise="random")

            gt = np.array([x, y, is_start_stroke]).transpose([1, 0])
            # Normalize
            gt[:, 1] -= np.min(gt[:, 1])
            gt[:, 0] -= np.min(gt[:, 0])
            factor = np.max(gt[:, 1])
            factor = 1/factor if factor else 1
            gt[:, :2] *= factor
            #draw_from_gt(gt, show=True)
        else:
            gt = np.zeros([cnn_embedding_width, 3])

        # Relativefy and normalize
        gt[:, 0] = stroke_recovery.relativefy(gt[:,0])
        gt[:, 1] = stroke_recovery.relativefy(gt[:,1])
        med = np.median(np.linalg.norm(gt[:,:2], axis=1))
        if med != 0:
            gt[:, :2] /= np.median(np.linalg.norm(gt[:,:2], axis=1))
        gt[0, :2] = 0 # let first position be origin
        return gt

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.root, item['image_path'])

        if "line_img" not in self.data[idx]:
            img = self.read_img(image_path)
            if img is None:
                print("Warning: image is None:", os.path.join(self.root, item['image_path']))
                return None
            self.data[idx]["line_img"] = img

        img = self.data[idx]["line_img"].copy()

        if self.load_strokes_flag and "stroke" not in self.data[idx]:
            img_id = Path(item['image_path']).stem
            self.data[idx]["stroke"] = self.process_stroke(img_id, img.shape[1])
        elif not self.load_strokes_flag:
            self.data[idx]["stroke"] = None

        if self.warp:
            img = distortions.warp_image(img)

        if self.occlusion:
            img = distortions.occlude(img, occlusion_freq=self.occlusion_freq,
                                          occlusion_size=self.occlusion_size,
                                          occlusion_level=self.occlusion_level)

        if self.blur:
            img = distortions.blur(img, max_intensity= self.blur_level)

        if self.random_distortions:
            img = distortions.random_distortions(img, sigma=self.distortion_sigma)

        if self.elastic_distortion:
            img = distortions.elastic_transform(img, alpha=self.elastic_alpha, sigma=self.elastic_sigma)

        img = distortions.crop(img) # trim leading/trailing whitespace


        # Add channel dimension, since resize and warp only keep non-trivial channel axis
        if self.num_of_channels==1:
            img=img[:,:, np.newaxis]

        img = img.astype(np.float32)
        img = img / 128.0 - 1.0


        gt = item['gt'] # actual text

        gt_label = string_utils.str2label(gt, self.char_to_idx) # character loss_indices of text

        #online = item.get('online', False)
        # THIS IS A HACK, FIX THIS (below)
        online = int(item['actual_writer_id']) > 700

        return {
            "line_img": img,
            "gt_label": gt_label,
            "gt": gt,
            "actual_writer_id": int(item['actual_writer_id']),
            "writer_id": int(item['writer_id']),
            "path": image_path,
            "online": online,
            "strokes": self.data[idx]["stroke"]
        }

if __name__=="__main__":
    from torch.utils.data import DataLoader
    from hwr_utils import character_set
    import character_set
    from utils import dict_to_list
    
    m = os.getcwd()
    print(m)
    root = r"../data/"
    json_path = r'prepare_IAM_Lines/gts/lines/txt/training.json'

    out_char_to_idx2, out_idx_to_char2, char_freq = character_set.make_char_set(
        [json_path], root=root)
    # Convert to a list to work with easydict
    idx_to_char = dict_to_list(out_idx_to_char2)

    char_to_idx, idx_to_char, char_freq = out_char_to_idx2, idx_to_char, char_freq

    device="cpu"
    default_collate = lambda x: collate(x, device=device)


    train_dataset = HwDataset([json_path],
                              char_to_idx,
                              img_height=60,
                              num_of_channels=1,
                              root=root,
                              )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=2,
                                  shuffle=False,
                                  num_workers=1,
                                  collate_fn=default_collate,
                                  pin_memory=device == "cpu")

    item = next(iter(train_dataloader))
    print(item.keys())
    print(item["strokes"])