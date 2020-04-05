###########################
### RUN FROM LOCAL FOLDER
###########################
from pathlib import Path
import argparse
import json
import cv2
import matplotlib.pyplot as plt
import sys
import pickle
from easydict import EasyDict as edict
from collections import defaultdict
import traceback
sys.path.insert(0, "../../")
sys.path.insert(0, ".")
from copy import deepcopy
import warnings
import time
from hwr_utils.stroke_recovery import *
from hwr_utils.stroke_plotting import *
from tqdm import tqdm
import multiprocessing
from functools import wraps

normal_error_handling = False

# def error( *args, **kwargs):
#     print("ERROR!")
#     Stop
#     return "error"

### GLOBALS
PARALLEL = True
RENDER = True

def error_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        # Error handler does nothing
        if normal_error_handling:
            return func(*args, **kwargs)
        else:
            try:
                return func(*args, **kwargs)  # exits here if everything is working
            except Exception as e:
                return e

    setattr(sys.modules[func.__module__], func.__name__, wrapper)

    return wrapper

class CreateDataset:
    """ Create a list of dictionaries with the following keys:
                "full_img_path": str of path to image,
                "xml_path": str of path to XML,
                "image_path": str of path to image relative to project,
                "dataset": str training/test,
                "x": list of x coordinates, rescaled to be square with Y
                "y": list of y coordinates, normalized to 0-1
                "t": list of t coordinates, normalized to stroke desired_num_of_strokes
                "start_times": start_times,
                "start_strokes": start_strokes,
                "x_to_y": ratio of x len to y len
                "raw": original stroke data from XML

        Export as JSON/Pickle
    """

    def __init__(self, max_strokes=3, square=True, output_folder_name='.',
                 xml_folder="prepare_online_data/line-level-xml/lineStrokes",
                 json_path="prepare_online_data/online_augmentation.json",
                 img_folder="prepare_online_data/lineImages",
                 data_folder="../../data",
                 render_images=True,
                 train_set_size=None,
                 test_set_size=None,
                 combine_images=False,
                 synthetic=False):

        # Specify data folder:
            # xml, json, and default images relative to the data folder
            # output folder is also relative to the data folder

        self.project_root = Path("../..").resolve()
        self.absolute_data_folder = Path(data_folder).resolve()

        # Abbreviated data folder
        self.relative_data_folder = self.absolute_data_folder.relative_to(self.project_root)

        self.json_path = self.absolute_data_folder / json_path
        self.xml_folder = self.absolute_data_folder / xml_folder
        self.original_img_folder = img_folder # Needs to be relative path to data folder

        #if self.absolute_data_folder not in Path(output_folder).parents:
        current_folder = Path(".").resolve().name
        self.output_folder = self.absolute_data_folder / current_folder / output_folder_name
        print("Output:", self.output_folder.resolve())
        self.new_img_folder = (self.output_folder / "images").resolve()
        self.new_img_folder.mkdir(exist_ok=True, parents=True)

        if self.json_path.suffix == ".json":
            self.data_dict = json.load(self.json_path.open("r"))
        elif self.json_path.suffix == ".npy":
            self.data_dict = np.load(self.json_path, allow_pickle=True)
        elif self.json_path.is_dir():
            self.data_dict = []
            for f in self.json_path.glob("*.json"):
                print(f"Loading from {f}")
                self.data_dict.extend(json.load(f.open("r")))
        else:
            raise Exception("JSON load problem")

        self.output_dict = {"train": [], "test": []}
        self.max_strokes=max_strokes
        self.square=square
        self.render_images=render_images
        self.test_set_size = test_set_size
        self.train_set_size = train_set_size
        self.combine_images = combine_images
        self.synthetic = synthetic
        if self.combine_images:
            self.process_fn = self.process_multiple
        elif self.synthetic:
            self.process_fn = self.process_synthetic
        else:
            self.process_fn = self.process_one

    #@error_handler
    #@staticmethod

    # def process_one(self, item):
    #     #the_dict = edict(self.__dict__)
    #     return self._process_one(self, item)

    # def process_one(self, item):
    #     return self.process_one(item)

    @staticmethod
    def _concat_raw_substrokes(raw1, raw2, x_space=10, time_space=.1, inplace=True):
        """

         Args:
             dict_list:
             last_point_remove (bool): Last point is a duplicate point for interpolation purposes, remove

         Returns:

         """
        # dict_keys(['x', 'y', 't', 'start_times', 'x_to_y', 'start_strokes', 'raw', 'tmin', 'tmax', 'trange'])
        # Get max X
        if not inplace:
            raw1 = raw1.copy()
        time_start = time_space + raw1[-1]["time"][-1]

        # where does raw2 start
        x2_start = np.min([x for stroke in raw2 for x in stroke["x"]])

        # calculate offset
        start_x = int(np.max([x for stroke in raw1 for x in stroke["x"]])+x_space-x2_start)

        # take average point
        y1 = np.average([y_pt for stroke in raw1 for y_pt in stroke["y"]])
        y2 = np.average([y_pt for stroke in raw2 for y_pt in stroke["y"]])
        y_diff = y1-y2

        for i in range(len(raw2)):
            raw1.append({"x":[xx+int(start_x) for xx in raw2[i]["x"]], "y":[y+y_diff for y in raw2[i]["y"]], "time":[tt+time_start for tt in raw2[i]["time"]]})

        return raw1

    @staticmethod
    def concat_raw_substrokes(raw_stroke_list, x_space=150, time_space=.1):
        if len(raw_stroke_list)==1:
            return raw_stroke_list[0]
        else:
            new_list = raw_stroke_list[0].copy()
            for i in range(1,len(raw_stroke_list)):
                new_list = CreateDataset._concat_raw_substrokes(new_list, raw_stroke_list[i], x_space=x_space, time_space=time_space)
            return new_list

    @staticmethod
    def process_one(item, hyperparams):
        self = hyperparams
        file_name = Path(item["image_path"]).stem
        rel_path = Path(item["image_path"]).relative_to(self.original_img_folder).with_suffix(".xml")
        xml_path = self.xml_folder / rel_path

        # For each item, we can extract multiple stroke_lists by using a sliding
        # window across the image.  Thus multiple json items will point to the same
        # image, but different strokes within that image.
        stroke_list, _ = read_stroke_xml(xml_path)
        stroke_dict = prep_stroke_dict(stroke_list, time_interval=0, scale_time_distance=True) # list of dictionaries, 1 per file
        all_substrokes = get_all_substrokes(stroke_dict, desired_num_of_strokes=self.max_strokes) # subdivide file into smaller sets

        if item["dataset"] in ["test", "val1", "val2"]:
            dataset = "test"
        else:
            dataset = "train"

        new_items = []

        for i, sub_stroke_dict in enumerate(all_substrokes):
            x_to_y = sub_stroke_dict.x_to_y

            # Don't warp images too much
            if self.square and (x_to_y < .5 or x_to_y > 2):
                continue

            new_img_path = (self.new_img_folder / (file_name + f"_{i}")).with_suffix(".tif")

            new_item = {
                "full_img_path": item["image_path"],
                "xml_path": xml_path.resolve().relative_to(self.absolute_data_folder).as_posix(),
                "image_path": new_img_path.relative_to(self.absolute_data_folder).as_posix(),
                "dataset": dataset
            }
            new_item.update(sub_stroke_dict) # added to what's already in the substroke dictionary

            # Create images
            ratio = 1 if self.square else x_to_y

            if self.render_images:
                draw_strokes(normalize_stroke_list(sub_stroke_dict.raw), ratio, save_path=new_img_path, line_width=.8)
            new_items.append(new_item)

        ## Add shapes -- the system needs some time to actually perform the writing op before reading it back
        for item in new_items:
            new_img_path = hyperparams.absolute_data_folder / item["image_path"]
            img = cv2.imread(new_img_path.as_posix())
            if img is None:
                print("Image file not found; is render off?")
            item["shape"] = deepcopy(img.shape)
            del img
        return new_items

    @staticmethod
    def process_synthetic(item, max_strokes, square, new_img_folder, absolute_data_folder, render_images, **kwargs):
        """ An item from the master dictionary {"stroke":[[x,y,eos],...], "text":"This is the GT"

        Args:
            item:
            max_strokes
            square
            new_img_folder
            absolute_data_folder
            render_images

        Returns:

        """
        text_key = "text" if "text" in item else "name"
        synthetic_format = isinstance(item["stroke"][0], list) # Synthetic format is a dict-list-list
        if synthetic_format:
            s = np.array(item["stroke"])
            s[-1,2] = 0 # replace last EOS with nothing
            file_name = "".join([c for c in item[text_key] if (c.isalpha() or c.isdigit() or c in [' ', "_"])]).rstrip()
            if "bias" in item:
                file_name += f"{item['bias']}_{item['style']}"
                if item['style']==13: # this style always starts with "is a random"
                    return None
            # if s[0,2]==1:
            #     warnings.warn(f"Stroke shouldn't usually start with end stroke! {file_name}")
            # Synthetic generator has EOS tokens - NOT SOS TOKENS!!!
            eos = np.argwhere(s[:, 2]).reshape(-1) + 1
            new_format = np.split(s[:, :2], eos)

            # Create RAW format
            raw_format = []
            for stroke in new_format:
                t = stroke[:,3] if stroke.shape[-1]>2 else np.array(range(len(stroke))) # put time on col 4
                raw_format += [{"x":stroke[:,0], "y":stroke[:,1], "time":t}]
        else: # assume already in raw (item["stroke"] = list of dicts)
            # THIS IS THE INDIC ONE -- exclude strokes longer than 1!
            new_format = raw_format = item["stroke"]
            # if len(item["stroke"])>1:
            #     return None
            file_name = Path(item[text_key]).stem

        stroke_dict = prep_stroke_dict(new_format, time_interval=0, scale_time_distance=True)  # list of dictionaries, 1 per file
        if stroke_dict is None:
            warnings.warn(f"{item[text_key]} failed")
            return None
        all_substrokes = get_all_substrokes(stroke_dict, desired_num_of_strokes=max_strokes)  # subdivide file into smaller sets
        stroke_dict.raw = raw_format
        dataset = "train" if not "dataset" in item else item["dataset"]
        new_items = []

        for i, sub_stroke_dict in enumerate(all_substrokes):
            x_to_y = sub_stroke_dict.x_to_y

            # Don't warp images too much
            if square and (x_to_y < .5 or x_to_y > 2):
                continue

            new_img_path = (new_img_folder / f"{file_name}_{i}").with_suffix(".tif")

            new_item = {
                "image_path": new_img_path.relative_to(absolute_data_folder).as_posix(),
                "dataset": dataset
            }
            new_item.update(sub_stroke_dict)  # added to what's already in the substroke dictionary

            # Create images
            ratio = 1 if square else x_to_y

            if render_images:
                draw_strokes(normalize_stroke_list(sub_stroke_dict.raw), ratio, save_path=new_img_path, line_width=.8)
            new_items.append(new_item)

        ## Add shapes -- the system needs some time to actually perform the writing op before reading it back
        for it in new_items:
            new_img_path = absolute_data_folder / it["image_path"]
            img = cv2.imread(new_img_path.as_posix())
            it["shape"] = deepcopy(img.shape)
            del img
        return new_items

    @staticmethod
    def concat_substrokes(dict_list, last_point_remove=True):
        """

        Args:
            dict_list:
            last_point_remove (bool): Last point is a duplicate point for interpolation purposes, remove

        Returns:

        """
        #dict_keys(['x', 'y', 't', 'start_times', 'x_to_y', 'start_strokes', 'raw', 'tmin', 'tmax', 'trange'])

        spaces = {"x":.1, "time":.1}
        spaces["start_times"] = spaces["time"]
        spaces["t"] = spaces["time"]

        output_dict = edict() #defaultdict(list)

        ## X and Start Times and Strokes
        # 'start_strokes', 'raw'
        # Calculate tmax
        output_dict["tmax"] = 0
        output_dict.raw = []
        for _dict in dict_list:
            output_dict["tmax"] += _dict["tmax"]-_dict["tmin"]
            assert _dict["tmin"] == 0

            for key in ['x', 't', 'start_times']:
                ## Some of the lists have an extra item on the end
                last_index = -1 if last_point_remove and key != 'start_times' else None
                if key in output_dict:
                    output_dict[key] = np.append(output_dict[key] , (_dict[key][:last_index] + (np.max(output_dict[key]) + spaces[key])))
                else:
                    output_dict[key] = _dict[key]

            ## Don't add on to these
            for key in ['y', 'start_strokes']:
                last_index = -1 if last_point_remove and key != 'start_strokes' else None
                if key in output_dict:
                    output_dict[key] = np.append(output_dict[key] , (_dict[key][:last_index]))
                else:
                    output_dict[key] = _dict[key]

            ## Raw -
            output_dict.raw.append(_dict.raw) # This is now a list of "raw" lists, which are lists of strokes, where each stroke is a dict
            # i.e. data_instances->strokes->stroke_point_dict

            ## Do the same thing here, add the max item etc. ugh

        output_dict.x_to_y = np.max(output_dict.x) / np.max(output_dict.y)
        output_dict.trange = (0, output_dict.tmax)

        return output_dict


    @staticmethod
    def process_multiple(items, hyperparams):
        """ Process multiple images to append together; no substrokes though

        Args:
            items:

        Returns:

        """

        # ONLY COMBINE IMAGES WITH OTHER IMAGES IN THE SAME CATEGORY, e.g. val1 with val1 etc.
        xml_paths = []
        meta_stroke_list = []
        file_names = []

        ## Combine stroke files
        for i, item in enumerate(items["data"]):
            if item["dataset"] != items["data"][i - 1]["dataset"] and i > 0:  # make sure they are all the same dataset, val1, val2, train, test
                break # shorten it if they're from different datasets

            file_names.append(Path(item["image_path"]).stem)
            rel_path = Path(item["image_path"]).relative_to(hyperparams.original_img_folder).with_suffix(".xml")
            xml_path = hyperparams.xml_folder / rel_path

            # For each item, we can extract multiple stroke_lists by using a sliding
            # window across the image.  Thus multiple json items will point to the same
            # image, but different strokes within that image.
            stroke_list, start_times = read_stroke_xml(xml_path)
            meta_stroke_list.append(stroke_list)
            xml_paths.append(xml_path)
        dataset = item["dataset"]

        concat_stroke_list = CreateDataset.concat_raw_substrokes(meta_stroke_list)

        super_stroke_list = prep_stroke_dict(concat_stroke_list, time_interval=0, scale_time_distance=True) # list of dictionaries, 1 per file
        new_items = []

        x_to_y = super_stroke_list.x_to_y

        new_file_name = "_".join(file_names) + ".tif"
        new_img_path = hyperparams.new_img_folder / new_file_name

        new_item = {
            "full_img_path": item["image_path"], # full path of where the image is saved
            "xml_path": [xml_path.resolve().relative_to(hyperparams.absolute_data_folder).as_posix() for xml_path in xml_paths], # relative path to original XML file
            "image_path": new_img_path.relative_to(hyperparams.absolute_data_folder).as_posix(), # relative path
            "dataset": dataset
        }
        new_item.update(super_stroke_list) # added to what's already in the substroke dictionary

        # Create images
        if hyperparams.render_images:
            draw_strokes(normalize_stroke_list(super_stroke_list.raw), x_to_y, save_path=new_img_path, line_width=.8)
        ## NEED TO FIX RAW, UGH
        new_items.append(new_item)

        ## Add shapes -- the system needs some time to actually perform the writing op before reading it back
        for item in new_items:
            new_img_path = hyperparams.absolute_data_folder / item["image_path"]
            img = cv2.imread(new_img_path.as_posix())
            if img is None:
                print("Image file not found; is render off?")
            item["shape"] = deepcopy(img.shape)
            del img

        return new_items

    @staticmethod
    def loop(temp_results, func=None):
        """ The non parallel version, better for error tracking

        Args:
            temp_results:
            func:

        Returns:

        """
        all_results = []
        no_errors = True
        for i, result in enumerate(temp_results):
            if func is not None:
                result = func(result)
            if (isinstance(result, Exception) or result is None):
                if no_errors:
                    print("error", result)
                    no_errors = False
            else:
                all_results.extend(result)
        return all_results

    def final_process(self, all_results):
        for item in all_results:
            if item is None:
                continue
            # If test set is specified to be smaller, add to training set after a certain size
            if item["dataset"] in ["train", "val1", "val2"] or (self.test_set_size and len(self.output_dict["test"]) > self.test_set_size):
                self.output_dict["train"].append(item)
            elif item["dataset"] == "test":
                self.output_dict["test"].append(item)
            ## This could mix training/test sets for partial stroke things -- nbd
            if self.train_set_size and self.test_set_size and len(self.output_dict["test"]) > self.test_set_size and len(self.output_dict["train"]) > self.train_set_size:
                break

        # PICKLE IT
        pickle.dump(self.output_dict["train"], (self.output_folder / "train_online_coords.pickle").open("wb"))
        pickle.dump(self.output_dict["test"], (self.output_folder / "test_online_coords.pickle").open("wb"))

        # ALSO JSON
        self.prep_for_json(self.output_dict["train"])
        self.prep_for_json(self.output_dict["test"])

        print("Creating train_online_coords.json and test_online_coords.json...")
        json.dump(self.output_dict["train"], (self.output_folder / "train_online_coords.json").open("w"))
        json.dump(self.output_dict["test"], (self.output_folder / "test_online_coords.json").open("w"))

        return self.output_dict

    @staticmethod
    def worker_wrapper(arg):
        #args, kwargs = arg
        try:
            return process_fn(arg, **hyper_param_dict)
        except:
            traceback.print_exc()
            return None

    def parallel(self, max_iter=None, start_iter=0, parallel=PARALLEL):
        data_dict = self.data_dict
        if max_iter:
            data_dict = data_dict[start_iter:max_iter]

        if self.combine_images:
            # for every item in the data dict, pick another item and combine them
            new_data_dict = []
            for i,d in enumerate(data_dict):
                new_data_dict.append({"data":((data_dict[i-1], data_dict[i]))})
            data_dict = new_data_dict

        ### Add all the hyperparameters to the item instead of keeping them in a class, seems to be faster
        # hyper_param_dict = {"max_strokes":self.max_strokes, "square":self.square, self.new_img_folder:"new_img_folder",
        #                     "absolute_data_folder":self.absolute_data_folder, "render_images":self.absolute_data_folder,
        #                     "xml_folder": self.xml_folder, "original_img_folder":self.original_img_folder}
        global hyper_param_dict, process_fn
        hyper_param_dict = edict(self.__dict__)
        process_fn = self.process_fn

        del hyper_param_dict["data_dict"]

        all_results = []
        start = 0
        step = 20000
        print(f"Total items: {len(data_dict)}, using batches of 20k")
        while start < len(data_dict):
            subdict = data_dict[:step]
            if subdict:
                if parallel:
                     all_results.extend(self._parallel(subdict))
                else:
                    all_results.extend(self.loop(tqdm(subdict), func=self.worker_wrapper))
            # Shrink data_dict for memory reasons
            data_dict = data_dict[step:]
        return self.final_process(all_results)

    def _parallel(self, data_dict):
        poolcount = multiprocessing.cpu_count() - 3
        pool = multiprocessing.Pool(processes=poolcount)

        if not self.synthetic:
            all_results = pool.imap_unordered(self.worker_wrapper,
                                              tqdm(data_dict))  # iterates through everything all at once
            pool.close()
        else:
            all_results = []

            # for i in range(10):
            #     data_dict2 = data_dict
            count = len(data_dict)
            # count = min(50000, count)
            # r = range(len(data_dict)) if len(data_dict) < 1000 else range(count, 2*count)
            r = range(count)
            callback = lambda x: all_results.extend(x) if not x is None else None
            for i in r:
                pool.apply_async(func=self.worker_wrapper, args=(data_dict[i],), callback=callback)

            pool.close()
            previous = 0
            with tqdm(total=count) as pbar:
                while previous < count - 80:
                    time.sleep(1)
                    new = len(all_results)
                    pbar.update(new - previous)
                    previous = new

            pool.join()
        return all_results

    def prep_for_json(self, iterable):
        if isinstance(iterable, list):
            for i in iterable:
                self.prep_for_json(i)
        elif isinstance(iterable, dict):
            for key, value in iterable.items():
                if isinstance(value, np.ndarray):
                    iterable[key] = value.tolist()
                else:
                    self.prep_for_json(value)
        elif isinstance(iterable, np.ndarray):
            pass

#def out_pickle(f)

def old():
    strokes = 3      # None=MAX stroke
    square = True      # Don't require square images
    instances = None    # None=Use all available instances
    test_set_size = 500 # use leftover test images in Training
    combine_images = False # combine images to make them longer

    variant="2"
    if square:
        variant += "Square"
    if instances is None:
        variant += "Full"
    else:
        variant += f"Small_{instances}"
    number_of_strokes = str(strokes) if isinstance(strokes, int) else "MAX"
    # data_set = CreateDataset(max_strokes=strokes,
    #                          square=square,
    #                          output_folder_name=f"./{number_of_strokes}_stroke_v{variant}",
    #                          render_images=False,
    #                          test_set_size=test_set_size,
    #                          combine_images=combine_images)
    data_set = CreateDataset(max_strokes=strokes,
                             square=square,
                             output_folder_name=f"./3_stroke_64_v2",
                             render_images=False,
                             test_set_size=test_set_size,
                             combine_images=combine_images)

    data_set.parallel(max_iter=instances, parallel=PARALLEL)

def new():
    strokes = 3      # None=MAX stroke
    square = False      # Don't require square images
    instances = None    # None=Use all available instances
    test_set_size = 30 # use leftover test images in Training
    train_set_size = 60
    combine_images = False # combine images to make them longer
    RENDER = False
    variant="verysmall"
    if square:
        variant += "Square"
    if instances is None:
        variant += "Full"
    else:
        variant += f"Small_{instances}"
    number_of_strokes = str(strokes) if isinstance(strokes, int) else "MAX"
    data_set = CreateDataset(max_strokes=strokes,
                             square=square,
                             output_folder_name=f"./{number_of_strokes}_stroke_v{variant}",
                             render_images=RENDER,
                             test_set_size=test_set_size,
                             train_set_size=train_set_size,
                             combine_images=combine_images,
                             #img_folder="prepare_online_data/lineImages",
                             #json_path="online_coordinate_data/3_stroke_64_v2/train_online_coords.json"
                             )
    data_set.parallel(max_iter=instances, parallel=PARALLEL)

def synthetic(vers="random"):
    strokes = None      # None=MAX stroke
    square = False      # Don't require square images
    instances = None    # None=Use all available instances
    test_set_size = None  # use leftover test images in Training
    train_set_size = None
    combine_images = False # combine images to make them longer

    if True: # new version
        variant = f"Boosted2_{vers}"
        source_json_path = f"synthetic_online/boosted2/{vers}"
    else: # Mason's
        variant="FullSynthetic100k"
        source_json_path = "synthetic_online/train_synth_full.json"
    #json_path = "synthetic_online/train_synth_sample.json"
    if "sample" in source_json_path:
        variant += "_sample"

    number_of_strokes = str(strokes) if isinstance(strokes, int) else "MAX"
    data_set = CreateDataset(max_strokes=strokes,
                             square=square,
                             output_folder_name=f"./{number_of_strokes}_stroke_v{variant}",
                             render_images=RENDER,
                             test_set_size=test_set_size,
                             train_set_size=train_set_size,
                             combine_images=combine_images,
                             #img_folder="prepare_online_data/lineImages",
                             json_path=source_json_path,
                             synthetic=True
                             )

    data_set.parallel(max_iter=instances, parallel=PARALLEL)

def indic():
    """
    /media/data/GitHub/simple_hwr/data/indic/devnagari_test.json
    /media/data/GitHub/simple_hwr/data/indic/devnagari_train.json
    /media/data/GitHub/simple_hwr/data/indic/tamil_test.json
    /media/data/GitHub/simple_hwr/data/indic/tamil_train.json
    /media/data/GitHub/simple_hwr/data/indic/telug_test.json
    /media/data/GitHub/simple_hwr/data/indic/telug_train.json
    Returns:

    """
    root = Path("/media/data/GitHub/simple_hwr/data/indic/")
    for language in "devnagari", "tamil", "telug":
        strokes = 1      # None=MAX stroke
        square = False      # Don't require square images
        instances = None    # None=Use all available instances
        test_set_size = None # use leftover test images in Training
        train_set_size = None
        combine_images = False # combine images to make them longer
        RENDER = False
        variant=f"{language}_one_stroke"
        load_path = root / ((language) + "_raw.json")

        data_set = CreateDataset(max_strokes=strokes,
                                 square=square,
                                 output_folder_name=f"./{variant}",
                                 render_images=RENDER,
                                 test_set_size=test_set_size,
                                 train_set_size=train_set_size,
                                 combine_images=combine_images,
                                 json_path=load_path,
                                 synthetic=True
                                 )
        data_set.parallel(max_iter=instances, parallel=PARALLEL)


if __name__ == "__main__":
    #new()

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--testing", action='store_true', help="No parallel", default=False)
    args = parser.parse_args()
    PARALLEL = not args.testing
    print("parallel", PARALLEL)
    synthetic("random")
    synthetic("normal")
    #indic()


import shutil
import os
from pathlib import Path

source = Path(f"/media/data/GitHub/simple_hwr/data/online_coordinate_data/")
dests = ["/fslg_hwr/hw_data/strokes/online_coordinate_data", "/home/taylor/shares/brodie/github/simple_hwr/data/online_coordinate_data"]

for dest in dests:
    for var in "random", "normal":
        subvar = "MAX_stroke_vBoosted2_{var}"
        try:
            (Path(dest) / subvar).mkdir(exist_ok=True, parents=True)
            shutil.copy(source / subvar / "train_online_coords.json", dest)
        except:
            pass


# import cProfile
#
# pr = cProfile.Profile()
# pr.enable()
# your_function_call()
# pr.disable()
# # after your program ends
# pr.print_stats(sort="calls")

