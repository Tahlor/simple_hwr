from losses import CustomLoss, DTWLoss


class IMG2IMG(CustomLoss):
    """ Compare Stroke Module strokes with GT ones
            # This is literally just DTW right?
    """

    def __init__(self, loss_indices, dtw_mapping_basis=None, device="cuda", **kwargs):
        super().__init__()

        # Use cross-entropy start point, DTW x,y
        loss_obj = DTWLoss(loss_indices, dtw_mapping_basis, relativefy_cross_entropy_gt=True)
        self.loss = loss_obj.dtw

    def weighted_mse_loss(self,input,target,weights):
        out = (input-target)**2
        out = out * weights.expand_as(out)
        loss = out.sum(0) # or sum over whatever dimensions
        return loss


class SM2GT(CustomLoss):
    """ Compare Stroke Module strokes with GT ones
            # This is literally just DTW right?
    """

    def __init__(self, loss_indices, dtw_mapping_basis=None, device="cuda", **kwargs):
        super().__init__()

        # Use cross-entropy start point, DTW x,y
        loss_obj = DTWLoss(loss_indices, dtw_mapping_basis, relativefy_cross_entropy_gt=True)
        self.loss = loss_obj.dtw

class SM2SM(CustomLoss):
    """ Compare Stroke Module strokes with other Stroke Module strokes
            # This is literally just DTW right?
    """

    def __init__(self, loss_indices, dtw_mapping_basis=None, device="cuda", **kwargs):
        super().__init__()

        # Use cross-entropy start point, DTW x,y
        loss_obj = DTWLoss(loss_indices, dtw_mapping_basis, relativefy_cross_entropy_gt=True)
        self.loss = loss_obj.dtw

