import torch
import torch.nn.functional as F

from mmdet3d.registry import MODELS


@MODELS.register_module()
class CoTCriterion:
    """Semantic criterion for ScanNet.

    Args:
        ignore_index (int): Ignore index.
        loss_weight (float): Loss weight.
    """

    def __init__(self, loss_weight):
        self.loss_weight = loss_weight

    def __call__(self, language_losses):
        """Calculate loss.

        Args:
            pred (dict): Predictions with List `sem_preds`
                of len batch_size, each of shape
                (n_queries_i, n_classes + 1).
            insts (list): Ground truth of len batch_size, 
                each InstanceData_ with `sp_masks` of shape
                (n_classes + 1, n_queries_i).

        Returns:
            Dict: with semantic loss value.
        """
        # losses = []
        # for language_loss in pred['language_losses']:

        #     losses.append(F.cross_entropy(
        #         pred_mask,
        #         gt_mask.sp_masks.float().argmax(0),
        #         ignore_index=self.ignore_index))
        loss = self.loss_weight * torch.mean(torch.stack(language_losses))
        return dict(cot_loss=loss)
