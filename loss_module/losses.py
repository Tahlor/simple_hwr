import warnings
import numpy as np
import torch
# from sdtw import SoftDTW
import torch.multiprocessing as multiprocessing
import torch.nn as nn
from taylor_dtw import custom_dtw as dtw
from scipy.spatial import KDTree
from torch import Tensor, tensor
from robust_loss_pytorch import AdaptiveLossFunction
import logging
from hwr_utils.stroke_dataset import create_gts
from hwr_utils.utils import to_numpy
from hwr_utils.stroke_recovery import relativefy_torch, swap_to_minimize_l1, get_number_of_stroke_pts_from_gt
from loss_module.dev import adaptive_dtw, swap_strokes
import sys
sys.path.append("..")
from unittest import FunctionTestCase
#pip install git+https://github.com/tahlor/pydtw

# def extensions():
#     #__builtins__.__NUMPY_SETUP__ = False
#     from Cython.Distutils import Extension
#     import numpy as np
#     extra_compile_args = ["-O3"]
#     extra_link_args = []
#     if sys.platform == "darwin":
#         extra_compile_args.append("-mmacosx-version-min=10.9")
#         extra_compile_args.append('-stdlib=libc++')
#         extra_link_args.append('-stdlib=libc++')
#     return  {"extra_compile_args":" ".join(extra_compile_args),
#                 "extra_link_args":" ".join(extra_link_args),
#                 "include_dirs":[np.get_include()],
#                 "language":"c++"}
#
# import pyximport
# #pyximport.install(setup_args={"include_dirs":np.get_include()})
# pyximport.install(setup_args=extensions())
# cd loss_module && python taylor_dtw/setup.py install --force
from taylor_dtw.custom_dtw import dtw2d_with_backward
from pydtw import constrained_dtw2d as constrained_dtw2d2

BCELoss = torch.nn.BCELoss()
BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones(1)*5)
SIGMOID = torch.nn.Sigmoid()
# DEVICE???
# x.requires_grad = False

logger = logging.getLogger("root."+__name__)


class CustomLoss(nn.Module):
    def __init__(self, loss_indices, device="cuda", **kwargs):
        super().__init__()
        self.name = self.__class__.__name__
        self.loss_indices = loss_indices
        self.device = "cpu"  # I guess this needs to be CPU? IDK
        self.__dict__.update(**kwargs)
        SIGMOID.to(device)
        if "subcoef" in kwargs:
            subcoef = kwargs["subcoef"]
            if isinstance(subcoef, str):
                subcoef = [float(s) for s in subcoef.split(",")]
            self.subcoef = Tensor(subcoef).to(self.device)
        else:
            # MAY NOT ALWAYS BE 4!!!
            length = len(range(*loss_indices.indices(4))) if isinstance(loss_indices, slice) else len(loss_indices)
            self.subcoef = torch.ones(length).to(self.device)


class CosineSimilarity(CustomLoss):
    """ Use opts to specify "variable_L1" (resample to get the same number of GTs/preds)
    """

    def __init__(self, loss_indices, **kwargs):
        """
        """
        # parse the opts - this will include opts regarding the DTW basis
        # loss_indices - the loss_indices to calculate the actual loss
        super().__init__(loss_indices, **kwargs)
        self.lossfun = self.cosine_similarity

    def cosine_similarity(self, preds, targs, label_lengths, **kwargs):
        # Each pred is multiplied by every GT
        # Soft max for each GT; classify the correct GT
        # Return the best guess GT
        


        pass

class NearestNeighbor(CustomLoss):
    """ Use opts to specify "variable_L1" (resample to get the same number of GTs/preds)
    """

    def __init__(self, loss_indices, **kwargs):
        """
        """
        # parse the opts - this will include opts regarding the DTW basis
        # loss_indices - the loss_indices to calculate the actual loss
        super().__init__(loss_indices, **kwargs)
        self.lossfun = self.nearest_neighbor

    def nearest_neighbor(self, preds, targs, label_lengths, line_imgs, kd_tree, **kwargs):
        bad_indices = line_imgs[torch.round(preds[:,self.loss_indices])]


class DTWLoss(CustomLoss):
    def __init__(self, loss_indices, dtw_mapping_basis=None, **kwargs):
        """

        Args:
            dtw_mapping_basis: The slice on which to match points (e.g. X,Y)
            dtw_loss_slice: The values for which to calculate the loss
                (e.g. after matching on X,Y, calculate the loss based on X,Y,SOS)

        """
        # parse the opts - this will include opts regarding the DTW basis
        # loss_indices - the loss_indices to calculate the actual loss
        super().__init__(loss_indices, **kwargs)
        self.dtw_mapping_basis = loss_indices if dtw_mapping_basis is None else dtw_mapping_basis
        self.lossfun = self.dtw
        self.abs = abs
        self.method = "normal" if not "method" in kwargs else kwargs["method"]

        if "cross_entropy_indices" in kwargs and kwargs["cross_entropy_indices"]:
            self.cross_entropy_indices = kwargs["cross_entropy_indices"]
        else:
            self.cross_entropy_indices = None

        if "barron" in kwargs and kwargs["barron"]:
            logger.info("USING BARRON + DTW!!!")
            self.barron = AdaptiveLossFunction(num_dims=len(loss_indices), float_dtype=np.float32, device='cpu').lossfun
        else:
            self.barron = None

        if "l1_center_of_mass" in kwargs and kwargs["l1_center_of_mass"]:
            logger.info("Taking center of mass")
            self.center_of_mass = True
        else:
            self.center_of_mass = False

        if "window_size" in kwargs:
            self.window_size = kwargs["window_size"]
        else:
            self.window_size = 20
        logger.info(f"DTW Window Size {self.window_size}")

        if "relativefy_cross_entropy_gt" in kwargs and kwargs["relativefy_cross_entropy_gt"]:
            logger.info("Relativefying stroke number + BCE!!!")
            self.relativefy = True
        else:
            self.relativefy = False

        if "low_level_dtw_alg" in kwargs and kwargs["low_level_dtw_alg"]=="invert":
            logger.info("Using INVERT low-level DTW")
            self._dtw_alg = "invert"
        else:
            self._dtw_alg = ""

        if "training_dataset" in kwargs and kwargs["training_dataset"] is not None:
            logger.info("Training dataset provided to loss function...GT adaptation possible")
            self.training_dataset = kwargs["training_dataset"]
            self.updates = 0

        self.swapping = True
        if "no_swapping" in kwargs and kwargs["no_swapping"]:
            self.swapping = False
            logger.info("Swapping strokes disabled")

    # not faster
    def parallel_dtw(self, preds, targs, label_lengths, **kwargs):
        raise Exception("Deprecated -- replace with async")
        loss = 0
        if self.parallel:
            pool = multiprocessing.Pool(processes=self.poolcount)
            as_and_bs = pool.imap(self.dtw_single, iter(zip(
                to_numpy(preds),
                targs)), chunksize=32)  # iterates through everything all at once
            pool.close()
            # print(as_and_bs[0])
            for i, (a, b) in enumerate(as_and_bs):  # loop through BATCH
                loss += abs(preds[i, a, :2] - targs[i][b, :2]).sum()
        return loss


    def dtw_adaptive(self, preds, targs, label_lengths, item, suffix, **kwargs):
        loss = 0
        item = add_preds_numpy(preds, item)
        for i in range(len(preds)):  # loop through BATCH
            pred = item["preds_numpy"][i]
            targ = targs[i]
            # This can be extended to do DTW with just a small buffer
            _targ = item["gt_numpy"][i]

            a,b,_gt, adaptive_instr_dict = adaptive_dtw(item["preds_numpy"][i], _targ,
                                                        constraint=self.window_size,
                                                        buffer=20,
                                                        opt="sample", method=None,
                                                        swapping=self.swapping
                                                        )

            ## Reverse the original GT
            if adaptive_instr_dict:
                if suffix != "_test":
                    original_gt = self.training_dataset[item["gt_idx"][i]]["gt"]
                    #print(original_gt.shape, _targ.shape)
                    #o = original_gt.copy()
                    #print(adaptive_instr_dict.keys())
                    if "reverse" in adaptive_instr_dict:
                        reverse_slice = adaptive_instr_dict["reverse"][1]
                        normal_slice = adaptive_instr_dict["reverse"][0]
                        original_gt[normal_slice,:2] = original_gt[reverse_slice,:2]
                    else:
                        swap_strokes(instruction_dict=adaptive_instr_dict, gt=original_gt, stroke_numbers=True)
                    self.updates += 1
                    if self.updates % 10000 == 0:
                        logger.info(f"Made {self.updates} adaptive GT changes")

                # Make sure something has changed!
                # assert np.any(np.not_equal(o,original_gt))
                # np.testing.assert_allclose(o, original_gt)

                _gt = Tensor(_gt)
            else:
                _gt = targ

            pred = preds[i][a, :][:, self.loss_indices]
            targ = _gt[b, :][:, :2]
            loss_by_point = (abs(pred - targ)* self.subcoef).sum(axis=1)
            if self.barron:
                loss += (self.barron(pred - targ)).sum() * self.subcoef  # AVERAGE pointwise loss for 1 image
            else:
                loss += loss_by_point.sum()  # AVERAGE pointwise loss for 1 image

            if self.cross_entropy_indices:
                loss += self.cross_entropy_loss(preds, targs, a, b, i, loss_by_point) # THIS IS IMPACTED BY SUBCOEF!

        return loss  # , to_value(loss)


    def dtw_l1_swapper_DEPRECATED(self, preds, targs, label_lengths, item, **kwargs):
        loss = 0
        item = add_preds_numpy(preds, item)
        for i in range(len(preds)):  # loop through BATCH
            pred = item["preds_numpy"][i]
            targ = targs[i]
            # This can be extended to do DTW with just a small buffer
            adjusted_targ = swap_to_minimize_l1(pred, targ.detach().numpy().astype("float64"), stroke_numbers=True, center_of_mass=self.center_of_mass)
            a, b = self.dtw_single((item["preds_numpy"][i], adjusted_targ), dtw_mapping_basis=self.dtw_mapping_basis, window_size=self.window_size)
            adjusted_targ = tensor(adjusted_targ)
            # LEN X VOCAB
            if self.method == "normal":
                pred = preds[i][a, :][:, self.loss_indices]
                targ = adjusted_targ[b, :][:, self.loss_indices]

            else:
                raise NotImplemented

            loss_by_point = abs(pred - targ).sum(axis=1)

            if self.barron:
                loss += (self.barron(pred - targ) * self.subcoef).sum()  # AVERAGE pointwise loss for 1 image
            elif True:
                loss += loss_by_point.sum()*self.subcoef  # AVERAGE pointwise loss for 1 image

            if self.cross_entropy_indices:

                pred2 = preds[i][a, :][:, self.cross_entropy_indices]
                targ2 = targs[i][b, :][:, self.cross_entropy_indices]
                pred2 = torch.clamp(pred2, -4, 4)
                if self.relativefy:
                    targ2 = relativefy_torch(targ2, default_value=1)  # default=1 ensures first point is a 1 (SOS);
                    targ2[targ2 != 0] = 1  # everywhere it's not zero, there was a stroke change
                    targ2[loss_by_point>.1] = 0 # if not sufficiently close to new point, don't make it a start point

                loss += BCEWithLogitsLoss(pred2, targ2).sum() * .1  # AVERAGE pointwise loss for 1 image
        return loss  # , to_value(loss)

    def dtw_sos_eos_DEPRECATED(self, preds, targs, label_lengths, item, **kwargs):
        """ Extra weight placed on SOS and EOS points -- not bad in concept though

        Args:
            preds:
            targs:
            label_lengths:
            item:
            **kwargs:

        Returns:

        """
        loss = 0
        item = add_preds_numpy(preds, item)

        for i in range(len(preds)):  # loop through BATCH
            a, b = self.dtw_single((item["preds_numpy"][i], targs[i]), dtw_mapping_basis=self.dtw_mapping_basis, window_size=self.window_size)

            # LEN X VOCAB
            if self.method=="normal":
                pred = preds[i][a, :][:, self.loss_indices]
                targ = targs[i][b, :][:, self.loss_indices]

            else:
                raise NotImplemented

            ## !!! DELETE THIS
            if self.barron:
                loss += (self.barron(pred - targ) * self.subcoef).sum()  # AVERAGE pointwise loss for 1 image
            elif True:
                loss += (abs(pred - targ) * self.subcoef).sum()  # AVERAGE pointwise loss for 1 image

            if self.cross_entropy_indices:
                pred2 = preds[i][a, :][:, self.cross_entropy_indices]
                targ2 = targs[i][b, :][:, self.cross_entropy_indices]

                pred2 = torch.clamp(pred2, -4,4)
                if self.relativefy:
                    targ2 = relativefy_torch(targ2, default_value=1) # default=1 ensures first point is a 1 (SOS);
                    targ2[targ2 != 0] = 1  # everywhere it's not zero, there was a stroke change
                loss += BCEWithLogitsLoss(pred2, targ2).sum() * .1  # AVERAGE pointwise loss for 1 image

                # TEMP HACKY LOSS, WEIGHT START/END POINTS MORE!
                # This will take only the first start point as having extra weight
                start_points = torch.nonzero(targ2.flatten())
                end_points = start_points[1:] - 1
                combined_points = torch.unique(torch.cat([start_points, end_points]))
                loss += (4*abs(pred[combined_points] - targ[combined_points]) * self.subcoef).sum()
        return loss  # , to_value(loss)

    def dtw_sos_eos_L2_DEPRECATED(self, preds, targs, label_lengths, item, **kwargs):
        loss = 0
        item = add_preds_numpy(preds, item)

        for i in range(len(preds)):  # loop through BATCH
            a, b = self.dtw_single(
                (item["preds_numpy"][i], targs[i]),
                dtw_mapping_basis=self.dtw_mapping_basis,
                window_size=self.window_size)

            # LEN X VOCAB
            if self.method=="normal":
                pred = preds[i][a, :][:, self.loss_indices]
                targ = targs[i][b, :][:, self.loss_indices]

            else:
                raise NotImplemented

            ## !!! DELETE THIS
            if self.barron:
                loss += (self.barron(pred - targ) * self.subcoef).sum()  # AVERAGE pointwise loss for 1 image
            elif True:
                loss += (abs(pred - targ)**2 * self.subcoef).sum()  # AVERAGE pointwise loss for 1 image

            if self.cross_entropy_indices:
                pred2 = preds[i][a, :][:, self.cross_entropy_indices]
                targ2 = targs[i][b, :][:, self.cross_entropy_indices]

                pred2 = torch.clamp(pred2, -4,4)
                if self.relativefy:
                    targ2 = relativefy_torch(targ2, default_value=1) # default=1 ensures first point is a 1 (SOS);
                    targ2[targ2 != 0] = 1  # everywhere it's not zero, there was a stroke change
                loss += BCEWithLogitsLoss(pred2, targ2).sum() * .1  # AVERAGE pointwise loss for 1 image

                # TEMP HACKY LOSS, WEIGHT START/END POINTS MORE!
                # This will take only the first start point as having extra weight
                start_points = torch.nonzero(targ2.flatten())
                end_points = start_points[1:] - 1
                combined_points = torch.unique(torch.cat([start_points, end_points]))
                loss += (4*abs(pred[combined_points] - targ[combined_points]) * self.subcoef).sum()
        return loss  # , to_value(loss)


    def dtw(self, preds, targs, label_lengths, item, **kwargs):
        loss = 0
        item = add_preds_numpy(preds, item)
        for i in range(len(preds)):  # loop through BATCH
            a, b = self.dtw_single((item["preds_numpy"][i], targs[i]), dtw_mapping_basis=self.dtw_mapping_basis, window_size=self.window_size)

            # LEN X VOCAB
            if self.method=="normal":
                pred = preds[i][a, :][:, self.loss_indices]
                targ = targs[i][b, :][:, self.loss_indices]

            else:
                raise NotImplemented

            loss_by_point = (abs(pred - targ)* self.subcoef).sum(axis=1) # targ is CPU, pred is GPU
            if self.barron:
                loss += (self.barron(pred - targ)).sum() * self.subcoef  # AVERAGE pointwise loss for 1 image
            else:
                loss += loss_by_point.sum()  # AVERAGE pointwise loss for 1 image

            if self.cross_entropy_indices:
                loss += self.cross_entropy_loss(preds, targs, a, b, i, loss_by_point) # THIS IS IMPACTED BY SUBCOEF!

        return loss  # , to_value(loss)


    def cross_entropy_loss(self, preds, targs, a, b, i, loss_by_point=None):
        """ Preds: 0/1 SOS
            Targs: 1,1,1,2,2... Stroke number
            Align Preds/Targs, calculate GT SOS from stroke number using relativefy

        """
        pred2 = preds[i][a, :][:, self.cross_entropy_indices]
        targ2 = targs[i][b, :][:, self.cross_entropy_indices]
        pred2 = torch.clamp(pred2, -4, 4)
        if self.relativefy:
            targ2 = relativefy_torch(targ2, default_value=1)  # default=1 ensures first point is a 1 (SOS);
            targ2[targ2 != 0] = 1  # everywhere it's not zero, there was a stroke change
            targ2[loss_by_point > .1] = 0  # if not sufficiently close to new point, don't make it a start point
        return BCEWithLogitsLoss(pred2, targ2).sum() * .1  # AVERAGE pointwise loss for 1 image

    def dtw_reverse(self, preds, targs, label_lengths, item, **kwargs):
        loss = 0
        item = add_preds_numpy(preds, item)

        for i in range(len(preds)):  # loop through BATCH
            #a, b = self.dtw_single((preds[i], targs[i]), dtw_mapping_basis=self.dtw_mapping_basis)
            targs_reverse = item["gt_reverse_strokes"][i]
            a, b = self.dtw_single_reverse(
                item["preds_numpy"][i],
                targs[i],
                targs_reverse,
                dtw_mapping_basis=self.dtw_mapping_basis,
                window_size=self.window_size)

            # LEN X VOCAB
            if self.method=="normal":
                pred = preds[i][a, :][:, self.loss_indices]
                targ = targs[i][b, :][:, self.loss_indices]
                targ2 = targs_reverse[b, :][:, self.loss_indices]
                sos_arg = item["sos_args"][i] # the actual start stroke indices
                first_indices_in_b = []
                for ii in sos_arg:
                    first_indices_in_b.append(np.argmax(b >= ii)) # Double check off by one errors etc.

                sos_arg = (np.concatenate([first_indices_in_b[1:], [targ.shape[0]]]) - first_indices_in_b).tolist() # convert indices to array sizes
            else:
                raise NotImplemented

            ## !!! DELETE THIS
            if self.barron:
                loss += (self.barron(pred - targ) * self.subcoef).sum()  # AVERAGE pointwise loss for 1 image
            else:
                a_strokes = (abs(pred - targ)* self.subcoef).split(sos_arg)
                b_strokes = (abs(pred - targ2) * self.subcoef).split(sos_arg)

                loss_a = Tensor([torch.sum(x) for x in a_strokes])
                loss_b = Tensor([torch.sum(x) for x in b_strokes])

                loss += (torch.min(loss_a, loss_b)).sum()  # AVERAGE pointwise loss for 1 image

            if self.cross_entropy_indices:
                pred2 = preds[i][a, :][:, self.cross_entropy_indices]
                targ2 = targs[i][b, :][:, self.cross_entropy_indices]

                pred2 = torch.clamp(pred2, -4,4)
                if self.relativefy:
                    targ2 = relativefy_torch(targ2, default_value=1) # default=1 ensures first point is a 1 (SOS);
                    targ2[targ2 != 0] = 1  # everywhere it's not zero, there was a stroke change
                loss += BCEWithLogitsLoss(pred2, targ2).sum() * .1  # AVERAGE pointwise loss for 1 image

        return loss  # , to_value(loss)


    def dtw_single(self, _input, dtw_mapping_basis, **kwargs):
        """ THIS DOES NOT USE SUBCOEF
        Args:
            _input (tuple): pred, targ, label_length

        Returns:

        """
        _pred, _targ = _input
        pred, targ = to_numpy(_pred[:, dtw_mapping_basis], astype="float64"), \
                     to_numpy(_targ[:, dtw_mapping_basis], astype="float64")

        if self._dtw_alg == "invert":
            gt_stroke_lens = get_number_of_stroke_pts_from_gt(_targ, stroke_numbers=True)
            warnings.simplefilter("ignore")
            dist, cost, a, b = DTWLoss._dtw_with_invert(pred, targ, gt_stroke_lens, **kwargs)
        else:
            dist, cost, a, b = DTWLoss._dtw(pred, targ, **kwargs)

        # Cost is weighted by how many GT stroke points, i.e. how long it is
        return a, b

    @staticmethod
    def dtw_single_reverse(pred, targ, reverse_targ, dtw_mapping_basis, **kwargs):
        """ THIS DOES NOT USE SUBCOEF
        Args:
            _input (tuple): pred, targ, label_length

        Returns:

        """
        pred, targ, reverse_targ = to_numpy(pred[:, dtw_mapping_basis], astype="float64"), \
                                    to_numpy(targ[:, dtw_mapping_basis], astype="float64"), \
                                   to_numpy(reverse_targ[:, dtw_mapping_basis], astype="float64")

        x1 = np.ascontiguousarray(pred)  # time step, batch, (x,y)
        x2 = np.ascontiguousarray(targ)
        x3 = np.ascontiguousarray(reverse_targ)

        dist, cost, a, b = dtw2d_with_backward(x1, x2, x3) # dist, cost, a, b


        # Cost is weighted by how many GT stroke points, i.e. how long it is
        return a, b

    @staticmethod
    def align(gt, gt_next, pred, str_len, window_size=10):
        """ Match the GT stroke + a few more strokes
            Get last match to GT stroke


            gt_next: next stroke
        """
        _comb = np.concatenate([gt, gt_next], axis=0)
        # print("combined", _comb, pred)
        f, cost, a, b = dtw.constrained_dtw2d(_comb, pred, window_size)
        # print(a,b)
        # print("GT", _comb[a])
        # print("Pred", pred[b])
        dist = np.sum(abs(_comb[a] - pred[b]))
        return dist, a, b

    @staticmethod
    def _dtw_with_invert(preds, gt, gt_stroke_lens, next_stroke_sample_len=5, pred_buffer=10, window=10, **kwargs):
        """ Matches each pred/stroke individually

        # Consider a cython function that takes exisiting distance matrix and only refills the last bit

        Args:
            gt:
            preds:
            next_stroke_sample_len: how many points to sample to choose whether to reverse
            pred_buffer: how many extra preds to add to alignment, in case pred is behind; if pred is ahead, hopefully
                this won't detract too much
            window: the DTW window size; since we're forcing strokes to line up, it probably doesn't need to be too big

        Returns:

        Optimizations:
            # gt_stroke_lens done in dataloader
            # store as contiguous arrays

        """
        gt = np.ascontiguousarray(gt)
        preds = np.ascontiguousarray(preds)
        gt_stroke_lens = np.insert(gt_stroke_lens, 0, 0)  # possibly swap first stroke

        match_a = []
        match_b = []

        pos_gt = 0
        pos_pred = 0

        ## FIRST ATTEMPT AT MANUAL DTW MATCH
        for i in range(len(gt_stroke_lens)):
            next_stroke_length = gt_stroke_lens[i + 1] if i + 1 < len(gt_stroke_lens) else 0
            str_len = gt_stroke_lens[i]
            _next_stroke_sample_len = min(next_stroke_sample_len, next_stroke_length)
            end_gt, end_pred = pos_gt + str_len, pos_pred + str_len + _next_stroke_sample_len + pred_buffer
            gt_next = gt[end_gt:end_gt + next_stroke_length]
            gt_stroke = gt[pos_gt:end_gt]
            pred_stroke = preds[pos_pred:end_pred]
            # print(gt_stroke, gt_next, pred_stroke)
            dist, a, b = DTWLoss.align(gt_stroke, gt_next[:_next_stroke_sample_len], pred_stroke, str_len, window)

            if i < len(gt_stroke_lens) - 1:
                gtr_next = np.ascontiguousarray(gt_next[::-1])
                distr, ar, br = DTWLoss.align(gt_stroke, gtr_next[:_next_stroke_sample_len], pred_stroke, str_len, window)
                # Invert the next stroke
                # print("LOSS", dist, distr)
                if distr < dist:
                    # print("REVERSE REVERSE!")
                    gt[end_gt:end_gt + next_stroke_length] = gtr_next
                    a, b = ar, br

                # print("A", a)
                # print("B", b)

                if i > 0:  # first one doesn't save alignment
                    last_stroke_idx_a = len(a) - 1 - np.argmax(
                        a[::-1] == str_len)  # how far to go in the alignment list
                    # -1 because 0 indexing; 10 - np.argmax(np.array([5]*10)[::-1]==5) - 1
                    # print("How far to go in alignment list", last_stroke_idx_a)
                    pos_pred_update = b[last_stroke_idx_a]  # how far the pred matches
                    # print("How far the pred matches", pos_pred_update)
                    # print("How far the gt matches (by def)", str_len)

                    match_a += [x + pos_gt for x in a[:last_stroke_idx_a]]
                    match_b += [x + pos_pred for x in b[:last_stroke_idx_a]]
                    pos_gt += str_len
                    pos_pred += pos_pred_update
            else:
                match_a += [x + pos_gt for x in a]
                match_b += [x + pos_pred for x in b]
        return None, None, match_a, match_b

    @staticmethod
    # ORIGINAL
    def _dtw(pred, targ, window_size=20):
        # Cost is weighted by how many GT stroke points, i.e. how long it is
        x1 = np.ascontiguousarray(pred)  # time step, batch, (x,y)
        x2 = np.ascontiguousarray(targ)
        if window_size:
            return dtw.constrained_dtw2d(x1, x2, window_size) # dist, cost, a, b
        else:
            return dtw.dtw2d(x1, x2) # dist, cost, a, b

    # @staticmethod
    # # FASTER
    # def _dtw(pred, targ):
    #     # Cost is weighted by how many GT stroke points, i.e. how long it is
    #     x1 = np.ascontiguousarray(pred)  # time step, batch, (x,y)
    #     x2 = np.ascontiguousarray(targ)
    #     return dtw.dtw2d(x1, x2) # dist, cost, a, b
    #
    # import dtaidistance.dtw
    # from dtaidistance.dtw_ndim

class NNLoss(CustomLoss):
    """ Use opts to specify "variable_L1" (resample to get the same number of GTs/preds)
    """

    def __init__(self, loss_indices, **kwargs):
        """
        """
        # parse the opts - this will include opts regarding the DTW basis
        # loss_indices - the loss_indices to calculate the actual loss
        super().__init__(loss_indices, **kwargs)

        self.pred_tree, self.gt_tree = True,True
        if "pred_tree" in kwargs and not kwargs["pred_tree"]:
            self.pred_tree = False
            logger.info("Disabled pred_tree; all preds evaluated, not all GTs")
        if "gt_tree" in kwargs and not kwargs["gt_tree"]:
            self.gt_tree = False
            logger.info("Disabled gt_tree; all GTs evaluated, not all preds")

        self.lossfun = self.nn_loss

    def nn_loss(self, preds, targs, label_lengths, item, **kwargs):
        # Forces predictions to be near a GT
        item = add_preds_numpy(preds, item)
        move_preds_to_gt_loss = 0
        move_gts_to_pred_loss = 0
        #kdtrees = create_kdtrees(item["preds_numpy"])

        if True:
            for i,pred in enumerate(preds):
                p = pred[:, self.loss_indices]
                targ = targs[i][:, self.loss_indices]
                pred_numpy = item["preds_numpy"][i][:, self.loss_indices]

                if self.gt_tree: # figure out where preds need to move to match GTs; all preds are used
                    distances, neighbor_indices = item["kdtree"][i].query(pred_numpy)
                    move_preds_to_gt_loss += torch.sum(abs(p - targ[neighbor_indices]) * self.subcoef)

                if self.pred_tree: # figure out where GTs need to move to match Preds; all GTs are used
                    k = KDTree(pred_numpy)
                    #k = kdtrees[i]
                    distances, neighbor_indices = k.query(targ)
                    move_gts_to_pred_loss += torch.sum(abs(p[neighbor_indices] - targ) )

            return move_gts_to_pred_loss + move_preds_to_gt_loss
        # else:
        #     pool = multiprocessing.Pool(processes=12)
        #     it = iter(zip(preds, targs, [item]*len(targs), range(len(preds)), [self.loss_indices]*len(targs)))
        #     sum = pool.imap(self.nn_loss_one, it)  # iterates through everything all at once
        #     pool.close()
        #     return np.sum(sum)

    # @ staticmethod
    # def nn_loss_one(pred, targ, item, i, loss_indices):
    #     p = pred[:, loss_indices]
    #     trg = targ[i][:, loss_indices]
    #
    #     pred_numpy = item["preds_numpy"][i][:, loss_indices]
    #     distances, neighbor_indices = item["kdtree"][i].query(pred_numpy)
    #     move_preds_to_gt_loss = torch.sum(abs(p - trg[neighbor_indices]) )
    #
    #     k = KDTree(pred_numpy)
    #     #k = kdtrees[i]
    #     distances, neighbor_indices = k.query(trg)
    #     move_gts_to_pred_loss = torch.sum(abs(p[neighbor_indices] - trg))
    #     return move_gts_to_pred_loss + move_preds_to_gt_loss

def create_kdtrees(preds, poolcount=24):
    pool = multiprocessing.Pool(processes=poolcount)
    it = iter(zip(preds, range(len(preds))))
    trees = pool.imap(_create_kd_tree, it)  # iterates through everything all at once
    pool.close()

    output = {}
    for i in trees:
        output[i[0]] = i[1]
    return output

def _create_kd_tree(pred_i):
    pred, i = pred_i
    return i, KDTree(pred[:,:2])


def add_preds_numpy(preds, item):
    if "preds_numpy" not in item:
        item["preds_numpy"] = []
        for p in preds:
            item["preds_numpy"].append(to_numpy(p))
    return item

class L1(CustomLoss):
    """ Use opts to specify "variable_L1" (resample to get the same number of GTs/preds)
    """

    def __init__(self, loss_indices, **kwargs):
        """
        """
        # parse the opts - this will include opts regarding the DTW basis
        # loss_indices - the loss_indices to calculate the actual loss
        super().__init__(loss_indices, **kwargs)
        self.lossfun = self.l1

    @staticmethod
    # BATCH x LEN x VOCAB
    def l1_swapper(preds, targs, label_lengths, **kwargs):
        loss = 0
        for i, pred in enumerate(preds):
            targ = targs[i].transpose(1,0) # swap width and vocab -> VOCAB x WIDTH
            pred = pred.transpose(1,0)
            diff  = torch.sum(torch.abs(pred.reshape(-1, 2) - targ.reshape(-1, 2)), axis=1)
            diff2 = torch.sum(torch.abs(pred.reshape(-1, 2) - torch.flip(targ.reshape(-1, 2), dims=(1,))), axis=1)
            loss += torch.sum(torch.min(diff, diff2)) # does not support subcoef
        return loss  # , to_value(loss)

    @staticmethod
    def variable_l1(preds, targs, label_lengths, **kwargs):
        """ Resmaple the targets to match whatever was predicted, i.e. so they have the same number (targs/preds)

        Args:
            preds:
            targs:
            label_lengths:

        Returns:

        """
        targs, label_lengths = resample_gt(preds, targs)
        loss = L1.loss(preds, targs, label_lengths)  # already takes average loss
        return loss  # , to_value(loss)

    def l1(self, preds, targs, label_lengths, **kwargs):
        loss = 0
        for i, pred in enumerate(preds):
            loss += torch.sum(abs(pred[:, self.loss_indices] - targs[i][:, self.loss_indices]) * self.subcoef)
        return loss  # , to_value(loss)

class L2(CustomLoss):
    """ Use opts to specify "variable_L1" (resample to get the same number of GTs/preds)
    """

    def __init__(self, loss_indices, **kwargs):
        """
        """
        # parse the opts - this will include opts regarding the DTW basis
        # loss_indices - the loss_indices to calculate the actual loss
        super().__init__(loss_indices, **kwargs)
        self.lossfun = self.l2

    @staticmethod
    def variable_l2(preds, targs, label_lengths, **kwargs):
        """ Resample the targets to match whatever was predicted, i.e. so they have the same number (targs/preds)

        Args:
            preds:
            targs:
            label_lengths:

        Returns:

        """
        targs, label_lengths = resample_gt(preds, targs)
        loss = L2.loss(preds, targs, label_lengths)  # already takes average loss
        return loss  # , to_value(loss)

    def l2(self, preds, targs, label_lengths, **kwargs):
        loss = 0
        for i, pred in enumerate(preds):  # loop through batch
            loss += torch.sum((pred[:, self.loss_indices] - targs[i][:, self.loss_indices]) ** 2 * self.subcoef) ** (
                        1 / 2)
        return loss  # , to_value(loss)


class CrossEntropy(nn.Module):
    """ Use opts to specify "variable_L1" (resample to get the same number of GTs/preds
    """

    def __init__(self, loss_indices, **kwargs):
        """
        """
        # parse the opts - this will include opts regarding the DTW basis
        # loss_indices - the loss_indices to calculate the actual loss
        super().__init__()
        self.__dict__.update(kwargs)
        self.loss_indices = loss_indices
        self.lossfun = self.cross_entropy

        self._loss = BCELoss
        if "activation" in kwargs.keys():
            if kwargs["activation"] == "sigmoid":
                self._loss = BCEWithLogitsLoss
                # torch.nn.Sigmoid().to(device)


    def cross_entropy(self, preds, targs, label_lengths, **kwargs):
        loss = 0
        for i, pred in enumerate(preds):  # loop through batches, since they are not the same size
            targ = targs[i]
            loss += self._loss(pred[:, self.loss_indices], targ[:, self.loss_indices])
        return loss  # , to_value(loss)


class SSL(nn.Module):
    def __init__(self, loss_indices, **kwargs):
        """
        """
        # parse the opts - this will include opts regarding the DTW basis
        # loss_indices - the loss_indices to calculate the actual loss
        super().__init__()
        self.loss_indices = 2
        self.nn_indices = slice(0, 2)
        self.lossfun = self.ssl

    def ssl(self, preds, targs, label_lengths, **kwargs):
        ### TODO: L1 distance and SOS/EOS are really two different losses, but they both depend on identifying the start points

        # Method
        ## Find the point nearest to the actual start stroke
        ## Assume this point should have been the predicted start stroke
        ## Calculate loss for predicting the start strokes!

        # OTHER LOSS
        # Sometimes the model "skips" stroke points because of DTW
        # Calculate the nearest point to every start and end point
        # Have this be an additional loss

        # start_time = time.time()
        # for each of the start strokes in targs
        loss_tensor = 0

        # Preds are BATCH x LEN x VOCAB
        for i in range(len(preds)):  # loop through batches
            # Get the coords of all start strokes
            targ_start_strokes = targs[i][torch.nonzero(targs[i][:, self.loss_indices]).squeeze(1), self.nn_indices]  #
            # targ_end_strokes = (targ_start_strokes-1)[1:] # get corresponding end strokes - this excludes the final stroke point!!
            k = KDTree(preds[i][:, self.nn_indices])

            # Start loss_indices; get the preds nearest the actual start points
            start_indices = k.query(targ_start_strokes)[1]
            pred_gt_fitted = torch.zeros(preds[i].shape[0])
            pred_gt_fitted[start_indices] = 1

            # End loss_indices
            # end_indices = k.query(targ_end_strokes)[1]
            # pred_end_fitted = torch.zeros(preds[i].shape[0])
            # pred_end_fitted[end_indices] = 1

            # Do L1 distance loss for start strokes and nearest stroke point
            # loss_tensor += abs(preds[i][start_indices, :2] - targ_start_strokes).sum()
            # loss_tensor += abs(preds[i][end_indices, :2] - targ_end_strokes).sum()

            # Do SOStr classification loss
            # print("preds", pred_gt_fitted)
            loss_tensor += BCELoss(preds[i][:, self.loss_indices], pred_gt_fitted)

            # print(targ_start_strokes, start_indices)
            # input()
            # print(pred_gt_fitted)
            # input()
            # print(targs[i][:,2])
            # print(loss_tensor)
            # input()

            # Do EOSeq prediction - not totally fair, again, we should evaluate it based on the nearest point to the last prediction
            # loss_tensor += BCELoss(preds[i][:, 3], targs[i][:, 3])

            # # Do L1 distance loss
            # loss += abs(preds[i][start_indices, :2] - targ_start_strokes).sum() / len(start_indices)
            # loss += abs(preds[i][end_indices, :2] - targ_end_strokes).sum() / len(end_indices)
            # loss += 0.1 * abs(pred_gt_fitted - targs[i][:, 2]).sum()
            # loss += 0.1 * abs(pred_end_fitted - targs[i][:, 3]).sum()
        loss = to_value(loss_tensor)
        # print("Time to compute ssl: ", time.time() - start_time)
        return loss_tensor  # , loss


def resample_gt(preds, targs, gt_format):
    batch = targs
    device = preds.device
    targs = []
    label_lengths = []
    for i in range(0, preds.shape[0]):
        pred_length = preds[i].shape[0]
        t = create_gts(batch["x_func"][i], batch["y_func"][i], batch["start_times"][i],
                       number_of_samples=pred_length, noise=None,
                       gt_format=gt_format)  # .transpose([1,0])
        t = torch.from_numpy(t.astype(np.float32)).to(device)
        targs.append(t)
        label_lengths.append(pred_length)
    return targs, label_lengths


def to_value(loss_tensor):
    return torch.sum(loss_tensor.cpu(), 0, keepdim=False).item()


def tensor_sum(tensor):
    return torch.sum(tensor.cpu(), 0, keepdim=False).item()

