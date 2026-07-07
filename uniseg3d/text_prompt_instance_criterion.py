from turtle import pd
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .structures import InstanceData_
from mmdet3d.registry import MODELS, TASK_UTILS


def get_iou(inputs, targets):
    """IoU for to equal shape masks.

    Args:
        inputs (Tensor): of shape (n_gts, n_points).
        targets (Tensor): of shape (n_gts, n_points).
    
    Returns:
        Tensor: IoU of shape (n_gts,).
    """
    inputs = inputs.sigmoid()
    binarized_inputs = (inputs >= 0.5).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


def dice_loss(inputs, targets):
    """Compute the DICE loss, similar to generalized IOU for masks.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs.
            Stores the binary classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
    
    Returns:
        Tensor: loss value.
    """
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()




def dice_loss_v2(inputs, targets, eps=1e-6, ignore_nan=True):
    """
    Soft Dice loss with:
      - automatic flattening (works for [N,L], [N,H,W], [N,1,H,W], ...)
      - optional NaN masking (element-wise)
    """
    # inputs: logits
    probs = inputs.sigmoid()

    # flatten to [N, D]
    probs = probs.flatten(1)
    targets = targets.flatten(1).float()

    if ignore_nan:
        valid = torch.isfinite(targets) & torch.isfinite(probs)
        # if no valid pixels for some samples, we handle safely below
        probs = torch.where(valid, probs, torch.zeros_like(probs))
        targets = torch.where(valid, targets, torch.zeros_like(targets))
        valid_sum = valid.sum(dim=1).clamp_min(1)  # avoid div-by-0 in edge cases
    else:
        valid_sum = torch.tensor(probs.shape[1], device=probs.device).repeat(probs.shape[0])

    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)

    # If using ignore_nan, denom/inter already exclude invalid because we zeroed them.
    dice = (2 * inter + eps) / (denom + eps)
    loss = 1 - dice

    # If a sample had 0 valid pixels, loss becomes 1 - (eps/eps)=0, which is fine.
    return loss.mean()


@MODELS.register_module()
class TextPromptInstanceCriterion:
    def __init__(self, loss_weight, non_object_weight, num_classes,
                 fix_dice_loss_weight, fix_mean_loss=False, total_weight=1.0,
                 use_cls_supervise=False):
        class_weight = [1] * num_classes + [non_object_weight]
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.fix_dice_loss_weight = fix_dice_loss_weight
        self.fix_mean_loss = fix_mean_loss
        self.total_weight = total_weight
        if use_cls_supervise:
            self.cls_weight = 1.
        else:
            self.cls_weight = 0.

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_layer_loss(self, aux_outputs, insts, indices=None):
        cls_preds = aux_outputs['cls_preds']
        pred_scores = aux_outputs['scores']
        pred_masks = aux_outputs['masks']

        if indices is None:
            indices = []
            for i in range(len(insts)):
                pred_instances = InstanceData_(
                    scores=cls_preds[i],
                    masks=pred_masks[i])
                gt_instances = InstanceData_(
                    labels=insts[i].labels_3d,
                    masks=insts[i].sp_masks)
                if insts[i].get('query_masks') is not None:
                    gt_instances.query_masks = insts[i].query_masks
                indices.append(self.matcher(pred_instances, gt_instances))

        cls_losses = []
        for cls_pred, inst, (idx_q, idx_gt) in zip(cls_preds, insts, indices):
            if cls_pred is None: continue
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long)
            cls_target[idx_q] = inst.labels_3d[idx_gt]
            cls_losses.append(F.cross_entropy(
                cls_pred, cls_target, cls_pred.new_tensor(self.class_weight)))
        if not cls_losses: return 0
        cls_loss = torch.mean(torch.stack(cls_losses))

        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for mask, score, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores,
                                                      insts, indices):
            if mask is None: continue
            if len(inst) == 0:
                continue

            pred_mask = mask[idx_q]
            tgt_mask = inst.sp_masks[idx_gt]
            mask_bce_losses.append(F.binary_cross_entropy_with_logits(
            pred_mask, tgt_mask.float()))
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask.float()))

        # todo: actually .mean() should be better
        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = 0

        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum() / len(pred_masks)

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4
            
            if self.fix_mean_loss:
                mask_bce_loss  = mask_bce_loss * len(pred_masks) \
                    / len(mask_bce_losses)
                mask_dice_loss  = mask_dice_loss * len(pred_masks) \
                    / len(mask_dice_losses)
        else:
            mask_bce_loss = 0
            mask_dice_loss = 0

        loss = (
            self.loss_weight[0] * cls_loss * self.cls_weight +
            self.loss_weight[1] * mask_bce_loss +
            self.loss_weight[2] * mask_dice_loss +
            self.loss_weight[3] * score_loss)

        return loss

    # todo: refactor pred to InstanceData_
    def __call__(self, pred, insts):
        cls_preds = pred['cls_preds']
        pred_scores = pred['scores']
        pred_masks = pred['masks']

        # match
        indices = []
        for i in range(len(insts)):
            query_index=torch.tensor([m for m in range(len(insts[i].labels_3d))]).to(insts[i].labels_3d.device)
            labels_index=query_index.clone()
            indices.append((query_index,labels_index))

        # class loss
        cls_losses = []
        for cls_pred, inst, (idx_q, idx_gt) in zip(cls_preds, insts, indices):
            if cls_pred is None: continue
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long)
            cls_target[idx_q] = inst.labels_3d[idx_gt]
            cls_losses.append(F.cross_entropy(
                cls_pred, cls_target, cls_pred.new_tensor(self.class_weight)))
        if not cls_losses: return {'text_prompt_inst_loss': torch.tensor(0.).to(insts[0].sp_masks.device)}
        cls_loss = torch.mean(torch.stack(cls_losses))

        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for mask, score, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores,
                                                      insts, indices):
            if mask is None: continue
            if len(inst) == 0:
                continue
            pred_mask = mask[idx_q]
            tgt_mask = inst.sp_masks[idx_gt]
            mask_bce_losses.append(F.binary_cross_entropy_with_logits(
                pred_mask, tgt_mask.float()))
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask.float()))

        # todo: actually .mean() should be better
        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = 0
        
        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum()

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4
            
            if self.fix_mean_loss:
                mask_bce_loss  = mask_bce_loss * len(pred_masks) \
                    / len(mask_bce_losses)
                mask_dice_loss  = mask_dice_loss * len(pred_masks) \
                    / len(mask_dice_losses)
        else:
            mask_bce_loss = 0
            mask_dice_loss = 0

        loss = (
            self.loss_weight[0] * cls_loss * self.cls_weight +
            self.loss_weight[1] * mask_bce_loss +
            self.loss_weight[2] * mask_dice_loss +
            self.loss_weight[3] * score_loss)

        if 'aux_outputs' in pred:
            for i, aux_outputs in enumerate(pred['aux_outputs']):
                loss += self.get_layer_loss(aux_outputs, insts, indices)

        return {'text_prompt_inst_loss': self.total_weight * loss}





@MODELS.register_module()
class LanguageSegCriterion:
    def __init__(self, loss_weight, 
                 fix_dice_loss_weight, fix_mean_loss=False, total_weight=1.0):
        self.loss_weight = loss_weight
        self.fix_dice_loss_weight = fix_dice_loss_weight
        self.fix_mean_loss = fix_mean_loss
        self.total_weight = total_weight

    def get_layer_loss(self, aux_outputs, insts):
        pred_masks = aux_outputs['masks']
        
        # Get device from pred_masks or insts
        device = None
        for mask in pred_masks:
            if mask is not None:
                device = mask.device
                break
        if device is None:
            for inst in insts:
                if hasattr(inst, 'sp_masks') and inst.sp_masks.numel() > 0:
                    device = inst.sp_masks.device
                    break
                elif not hasattr(inst, 'sp_masks') and inst.numel() > 0:
                    device = inst.device
                    break
        if device is None:
            # Fallback: use first inst's device or CPU
            if len(insts) > 0 and hasattr(insts[0], 'sp_masks'):
                device = insts[0].sp_masks.device if insts[0].sp_masks.numel() > 0 else torch.device('cpu')
            else:
                device = torch.device('cpu')

        mask_bce_losses, mask_dice_losses = [], []
        for mask, inst in zip(pred_masks, insts):
            if mask is None:
                continue
            
            # Extract sp_masks if inst is InstanceData_ object, otherwise use inst directly
            if hasattr(inst, 'sp_masks'):
                inst_masks = inst.sp_masks
            else:
                inst_masks = inst
            
            if inst_masks.numel() == 0:
                continue
            
            inst_masks = inst_masks.to(mask.device)
            
            # Find indices that are not NaN in both pred_masks and inst_masks
            none_nan_mask_index = ~torch.isnan(mask).all(dim=1)
            none_nan_inst_index = ~torch.isnan(inst_masks).all(dim=1)
            
            # Get their intersection, which is the valid mask index
            valid_index = none_nan_mask_index & none_nan_inst_index
            
            if valid_index.sum() == 0:
                continue
            
            pred_mask = mask[valid_index]
            tgt_mask = inst_masks[valid_index]
            
            mask_bce_losses.append(F.binary_cross_entropy_with_logits(
                pred_mask, tgt_mask.float()))
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask.float()))

        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum()

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4
            
            if self.fix_mean_loss:
                mask_bce_loss = mask_bce_loss * len(pred_masks) \
                    / len(mask_bce_losses)
                mask_dice_loss = mask_dice_loss * len(pred_masks) \
                    / len(mask_dice_losses)
        else:
            mask_bce_loss = torch.tensor(0.0, device=device)
            mask_dice_loss = torch.tensor(0.0, device=device)

        loss = (
            self.loss_weight[0] * mask_bce_loss +
            self.loss_weight[1] * mask_dice_loss)

        return loss

    def __call__(self, pred, insts):
        pred_masks = pred['masks']
        
        # Get device from pred_masks or insts
        device = None
        for mask in pred_masks:
            if mask is not None:
                device = mask.device
                break
        if device is None:
            for inst in insts:
                if hasattr(inst, 'sp_masks') and inst.sp_masks.numel() > 0:
                    device = inst.sp_masks.device
                    break
                elif not hasattr(inst, 'sp_masks') and inst.numel() > 0:
                    device = inst.device
                    break
        if device is None:
            # Fallback: use first inst's device or CPU
            if len(insts) > 0 and hasattr(insts[0], 'sp_masks'):
                device = insts[0].sp_masks.device if insts[0].sp_masks.numel() > 0 else torch.device('cpu')
            else:
                device = torch.device('cpu')

        mask_bce_losses, mask_dice_losses = [], []
        for mask, inst in zip(pred_masks, insts):
            if mask is None:
                continue
            
            # Extract sp_masks if inst is InstanceData_ object, otherwise use inst directly
            if hasattr(inst, 'sp_masks'):
                inst_masks = inst.sp_masks
            else:
                inst_masks = inst
            
            if inst_masks.numel() == 0:
                continue
            
            inst_masks = inst_masks.to(mask.device)
            
            # Find indices that are not NaN in both pred_masks and inst_masks
            none_nan_mask_index = ~torch.isnan(mask).all(dim=1)
            none_nan_inst_index = ~torch.isnan(inst_masks).all(dim=1)
            
            # Get their intersection, which is the valid mask index
            valid_index = none_nan_mask_index & none_nan_inst_index
            
            if valid_index.sum() == 0:
                continue
            
            pred_mask = mask[valid_index]
            tgt_mask = inst_masks[valid_index]
            
            mask_bce_losses.append(F.binary_cross_entropy_with_logits(
                pred_mask, tgt_mask.float()))
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask.float()))

        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum()

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4
            
            if self.fix_mean_loss:
                mask_bce_loss = mask_bce_loss * len(pred_masks) \
                    / len(mask_bce_losses)
                mask_dice_loss = mask_dice_loss * len(pred_masks) \
                    / len(mask_dice_losses)
        else:
            mask_bce_loss = torch.tensor(0.0, device=device)
            mask_dice_loss = torch.tensor(0.0, device=device)

        loss = (
            self.loss_weight[0] * mask_bce_loss +
            self.loss_weight[1] * mask_dice_loss)

        if 'aux_outputs' in pred:
            for i, aux_outputs in enumerate(pred['aux_outputs']):
                loss += self.get_layer_loss(aux_outputs, insts)

        return {'language_seg_loss': self.total_weight * loss}





def masked_bce_with_logits(inputs, targets, ignore_nan=True, reduction="mean"):
    """
    BCEWithLogits that can ignore NaNs in targets (and inputs).
    Works element-wise, supports any shape.
    """
    targets = targets.float()

    if ignore_nan:
        valid = torch.isfinite(targets) & torch.isfinite(inputs)
        if valid.sum() == 0:
            # no valid supervision -> return 0 on correct device/dtype
            return inputs.sum() * 0.0

        inputs = inputs[valid]
        targets = targets[valid]

    return F.binary_cross_entropy_with_logits(inputs, targets, reduction=reduction)


def focal_bce_with_logits(inputs, targets, alpha=0.25, gamma=2.0, ignore_nan=True, reduction="mean"):
    """
    Focal loss variant built on BCEWithLogits, optional replacement for BCE.
    Keeps same "logits + targets" interface.
    """
    targets = targets.float()

    if ignore_nan:
        valid = torch.isfinite(targets) & torch.isfinite(inputs)
        if valid.sum() == 0:
            return inputs.sum() * 0.0
        inputs = inputs[valid]
        targets = targets[valid]

    # BCE per element
    bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # pt = probability of correct class
    p = torch.sigmoid(inputs)
    pt = p * targets + (1 - p) * (1 - targets)
    # alpha balancing
    at = alpha * targets + (1 - alpha) * (1 - targets)
    loss = at * (1 - pt).pow(gamma) * bce

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss




@MODELS.register_module()
class LanguageSegCriterion_v2:
    """
    v2 changes ONLY loss function definitions/usage:
      - BCE: masked BCEWithLogits (element-wise NaN ignore)
      - Dice: flattened Dice with element-wise NaN ignore and eps=1e-6

    All internal logic (including aux_outputs handling, scaling, etc.) is preserved.
    """
    def __init__(self, loss_weight,
                 fix_dice_loss_weight, fix_mean_loss=False, total_weight=1.0,
                 dice_eps=1e-6, ignore_nan=True,
                 use_focal_loss=True, focal_alpha=0.25, focal_gamma=2.0):
        self.loss_weight = loss_weight
        self.fix_dice_loss_weight = fix_dice_loss_weight
        self.fix_mean_loss = fix_mean_loss
        self.total_weight = total_weight

        # v2-only knobs (do not change internal criterion logic)
        self.dice_eps = dice_eps
        self.ignore_nan = ignore_nan
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def get_layer_loss(self, aux_outputs, insts):
        pred_masks = aux_outputs['masks']

        # Get device from pred_masks or insts
        device = None
        for mask in pred_masks:
            if mask is not None:
                device = mask.device
                break
        if device is None:
            for inst in insts:
                if hasattr(inst, 'sp_masks') and inst.sp_masks.numel() > 0:
                    device = inst.sp_masks.device
                    break
                elif not hasattr(inst, 'sp_masks') and inst.numel() > 0:
                    device = inst.device
                    break
        if device is None:
            # Fallback: use first inst's device or CPU
            if len(insts) > 0 and hasattr(insts[0], 'sp_masks'):
                device = insts[0].sp_masks.device if insts[0].sp_masks.numel() > 0 else torch.device('cpu')
            else:
                device = torch.device('cpu')

        mask_bce_losses, mask_dice_losses = [], []
        for mask, inst in zip(pred_masks, insts):
            if mask is None:
                continue

            # Extract sp_masks if inst is InstanceData_ object, otherwise use inst directly
            if hasattr(inst, 'sp_masks'):
                inst_masks = inst.sp_masks
            else:
                inst_masks = inst

            if inst_masks.numel() == 0:
                continue

            inst_masks = inst_masks.to(mask.device)

            # Find indices that are not NaN in both pred_masks and inst_masks
            none_nan_mask_index = ~torch.isnan(mask).all(dim=1)
            none_nan_inst_index = ~torch.isnan(inst_masks).all(dim=1)

            # Get their intersection, which is the valid mask index
            valid_index = none_nan_mask_index & none_nan_inst_index

            if valid_index.sum() == 0:
                continue

            pred_mask = mask[valid_index]
            tgt_mask = inst_masks[valid_index]

            # v2: use masked BCE + improved Dice (definitions only)
            if self.use_focal_loss:
                mask_bce_losses.append(
                    focal_bce_with_logits(
                        pred_mask, tgt_mask,
                        alpha=self.focal_alpha,
                        gamma=self.focal_gamma,
                        ignore_nan=self.ignore_nan,
                        reduction="mean"
                    )
                )
            else:
                mask_bce_losses.append(
                    masked_bce_with_logits(
                        pred_mask, tgt_mask,
                        ignore_nan=self.ignore_nan,
                        reduction="mean"
                    )
                )
            mask_dice_losses.append(
                dice_loss_v2(
                    pred_mask, tgt_mask,
                    eps=self.dice_eps,
                    ignore_nan=self.ignore_nan
                )
            )

        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum()

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4

            if self.fix_mean_loss:
                mask_bce_loss = mask_bce_loss * len(pred_masks) \
                    / len(mask_bce_losses)
                mask_dice_loss = mask_dice_loss * len(pred_masks) \
                    / len(mask_dice_losses)
        else:
            mask_bce_loss = torch.tensor(0.0, device=device)
            mask_dice_loss = torch.tensor(0.0, device=device)

        loss = (
            self.loss_weight[0] * mask_bce_loss +
            self.loss_weight[1] * mask_dice_loss)

        return loss

    def __call__(self, pred, insts):
        pred_masks = pred['masks']

        # Get device from pred_masks or insts
        device = None
        for mask in pred_masks:
            if mask is not None:
                device = mask.device
                break
        if device is None:
            for inst in insts:
                if hasattr(inst, 'sp_masks') and inst.sp_masks.numel() > 0:
                    device = inst.sp_masks.device
                    break
                elif not hasattr(inst, 'sp_masks') and inst.numel() > 0:
                    device = inst.device
                    break
        if device is None:
            # Fallback: use first inst's device or CPU
            if len(insts) > 0 and hasattr(insts[0], 'sp_masks'):
                device = insts[0].sp_masks.device if insts[0].sp_masks.numel() > 0 else torch.device('cpu')
            else:
                device = torch.device('cpu')

        mask_bce_losses, mask_dice_losses = [], []
        for mask, inst in zip(pred_masks, insts):
            if mask is None:
                continue

            # Extract sp_masks if inst is InstanceData_ object, otherwise use inst directly
            if hasattr(inst, 'sp_masks'):
                inst_masks = inst.sp_masks
            else:
                inst_masks = inst

            if inst_masks.numel() == 0:
                continue

            inst_masks = inst_masks.to(mask.device)

            # Find indices that are not NaN in both pred_masks and inst_masks
            none_nan_mask_index = ~torch.isnan(mask).all(dim=1)
            none_nan_inst_index = ~torch.isnan(inst_masks).all(dim=1)

            # check whether there is an all-zero gt
            none_zero_inst_index=inst_masks.sum(dim=1) != 0

            # if there is any false inside, drop into pdb
            # if False in none_zero_inst_index:

            # Get their intersection, which is the valid mask index
            # print('none_nan_mask_index.shape: ', none_nan_mask_index.shape)
            # print('none_nan_inst_index.shape: ', none_nan_inst_index.shape)
            # print('mask.shape: ', mask.shape)
            # print('inst_masks.shape: ', inst_masks.shape)
            valid_index = none_nan_mask_index & none_nan_inst_index & none_zero_inst_index

            if valid_index.sum() == 0:
                continue

            pred_mask = mask[valid_index]
            tgt_mask = inst_masks[valid_index]

            # v2: use masked BCE + improved Dice (definitions only)
            if self.use_focal_loss:
                mask_bce_losses.append(
                    focal_bce_with_logits(
                        pred_mask, tgt_mask,
                        alpha=self.focal_alpha,
                        gamma=self.focal_gamma,
                        ignore_nan=self.ignore_nan,
                        reduction="mean"
                    )
                )
            else:
                mask_bce_losses.append(
                    masked_bce_with_logits(
                        pred_mask, tgt_mask,
                        ignore_nan=self.ignore_nan,
                        reduction="mean"
                    )
                )
            mask_dice_losses.append(
                dice_loss_v2(
                    pred_mask, tgt_mask,
                    eps=self.dice_eps,
                    ignore_nan=self.ignore_nan
                )
            )

        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum()

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4

            if self.fix_mean_loss:
                mask_bce_loss = mask_bce_loss * len(pred_masks) \
                    / len(mask_bce_losses)
                mask_dice_loss = mask_dice_loss * len(pred_masks) \
                    / len(mask_dice_losses)
        else:
            mask_bce_loss = torch.tensor(0.0, device=device)
            mask_dice_loss = torch.tensor(0.0, device=device)
        
        loss = (
            self.loss_weight[0] * mask_bce_loss +
            self.loss_weight[1] * mask_dice_loss)

        if 'aux_outputs' in pred:
            for i, aux_outputs in enumerate(pred['aux_outputs']):
                aux_loss = self.get_layer_loss(aux_outputs, insts)
                aux_weight = 1.0 / (i + 1)
                loss = loss + aux_weight * aux_loss
       

        return {'seg_loss': self.total_weight * loss}





def dice_loss(inputs, targets, eps=1e-6):
    """
    Compute Dice loss from logits inputs and binary targets.
    inputs: logits tensor, arbitrary shape. Last dimension is treated as "pixels".
    targets: same shape as inputs, {0,1} floats.
    Returns scalar tensor (mean over samples).
    """
    # apply sigmoid to logits
    probs = inputs.sigmoid()
    # compute per-sample dice along the last dim
    numerator = 2.0 * (probs * targets).sum(dim=-1)
    denominator = probs.sum(dim=-1) + targets.sum(dim=-1)
    dice = (numerator + eps) / (denominator + eps)
    loss = 1.0 - dice
    return loss.mean()

@MODELS.register_module()
class TextPrompt_Criterion_Glamm:
    def __init__(self, loss_weight, non_object_weight=None,
                 fix_dice_loss_weight=False, fix_mean_loss=False, total_weight=1.0,
                 use_cls_supervise=False, dice_scale=4.0):
        """
        loss_weight: [bce_weight, dice_weight]
        fix_dice_loss_weight: if True (or numeric), scale dice by dice_scale
        fix_mean_loss: legacy flag kept - implemented as optional rescaling by expected count
        dice_scale: numeric scale applied to dice when fix_dice_loss_weight is truthy
        """
        self.loss_weight = loss_weight
        self.fix_dice_loss_weight = fix_dice_loss_weight
        self.fix_mean_loss = fix_mean_loss
        self.total_weight = total_weight
        self.dice_scale = dice_scale if dice_scale is not None else 4.0

        self.cls_weight = 1.0 if use_cls_supervise else 0.0

    def _get_src_permutation_idx(self, indices):
        # same as original: helper to permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_layer_loss(self, aux_outputs, insts, origin_queries):
        """
        aux_outputs: dict with key 'masks' similar shape as main pred
        insts: list of Instance objects (with attribute sp_masks)
        origin_queries: list of origin_query tensors used to filter valid queries
        """
        pred_masks = aux_outputs.get('masks', [])

        mask_bce_losses = []
        mask_dice_losses = []

        # iterate over images (assumes pred_masks aligns with insts & origin_queries)
        for mask, inst, i_origin_query in zip(pred_masks, insts, origin_queries):
            if mask is None:
                continue
            if len(inst) == 0:
                continue

            # ensure device/dtype alignment
            device = mask.device

            # none_nan_mask_index = (~torch.isnan(i_origin_query)).all(dim=-1)
            none_zero_mask_index = (i_origin_query.sum(dim=1) != 0)

            none_nan_gt_index = ~torch.isnan(inst).all(dim=1).to(device)


            # Get their intersection, which is the valid mask index
            valida_ids = none_zero_mask_index & none_nan_gt_index

            if valida_ids.sum() == 0:
                continue
            
            pred_mask = mask[valida_ids]           # logits: [num_queries, pixels]
            tgt_mask = inst.to(device)[valida_ids]  # targets: same shape

            # compute losses for this image (over all matched queries)
            # if there are zero valid queries, skip
            if pred_mask.numel() == 0 or tgt_mask.numel() == 0:
                continue

            bce = F.binary_cross_entropy_with_logits(pred_mask, tgt_mask.float(), reduction='mean')
            dice = dice_loss(pred_mask, tgt_mask.float())

            mask_bce_losses.append(bce)
            mask_dice_losses.append(dice)

        device_for_zero = None
        if len(pred_masks):
            # get device from first available mask if possible
            for m in pred_masks:
                if m is not None:
                    device_for_zero = m.device
                    break

        if len(mask_bce_losses) > 0:
            # mean over valid images
            mask_bce_loss = torch.stack(mask_bce_losses).mean()
            mask_dice_loss = torch.stack(mask_dice_losses).mean()

            # optional dice scaling (legacy behavior preserved)
            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss * float(self.dice_scale)

            # optional fix_mean_loss: if enabled, rescale by ratio of expected to actual valid
            if self.fix_mean_loss and len(pred_masks) > 0:
                expected = float(len(pred_masks))
                actual = float(len(mask_bce_losses))
                if actual > 0:
                    scale = expected / actual
                    mask_bce_loss = mask_bce_loss * scale
                    mask_dice_loss = mask_dice_loss * scale
        else:
            device = device_for_zero if device_for_zero is not None else torch.device('cpu')
            mask_bce_loss = torch.tensor(0., device=device)
            mask_dice_loss = torch.tensor(0., device=device)

        loss = (self.loss_weight[0] * mask_bce_loss +
                self.loss_weight[1] * mask_dice_loss)
        return loss

    def __call__(self, pred, insts, origin_queries):
        """
        pred: dict containing 'masks' (list of per-image masks) and optionally 'aux_outputs'
        insts: list of ground-truth instances (each with .sp_masks and .labels_3d)
        origin_queries: list of origin_query tensors used to filter valid queries (NaN rows ignored)
        """
        pred_masks = pred.get('masks', [])

        # create trivial indices: 0..N-1 per image (keeps original behavior)
        # indices = []
        # for i in range(len(insts)):
        #     device = insts[i].labels_3d.device if hasattr(insts[i], 'labels_3d') else torch.device('cpu')
        #     n = 0
        #     if hasattr(insts[i], 'labels_3d'):
        #         n = len(insts[i].labels_3d)
        #     query_index = torch.arange(n, device=device)
        #     labels_index = query_index.clone()
        #     indices.append((query_index, labels_index))

        mask_bce_losses = []
        mask_dice_losses = []

        for mask, inst, origin_query in zip(pred_masks, insts, origin_queries):
            if mask is None:
                continue
            if len(inst) == 0:
                continue

            device = mask.device
            # idx_q = idx_q.to(device)
            # idx_gt = idx_gt.to(device)

            pred_mask = mask
            tgt_mask = inst.to(device)

            # valida_ids: which queries in origin_query are valid (no NaN in any dim)


            none_zero_mask_index = (origin_query.sum(dim=1) != 0)

            none_nan_gt_index = ~torch.isnan(inst).all(dim=1).to(device)


            # Get their intersection, which is the valid mask index
            valida_ids = none_zero_mask_index & none_nan_gt_index

            # if no valid query rows, skip
            if valida_ids.sum() == 0:
                continue

            pm_valid = pred_mask[valida_ids, :]
            tm_valid = tgt_mask[valida_ids, :]


            # compute losses (mean reduction)
            i_bce_loss = F.binary_cross_entropy_with_logits(pm_valid, tm_valid.float(), reduction='mean')
            i_dice_loss = dice_loss(pm_valid, tm_valid.float())

            mask_bce_losses.append(i_bce_loss)
            mask_dice_losses.append(i_dice_loss)

        # aggregate main losses
        device_for_zero = None
        for m in pred_masks:
            if m is not None:
                device_for_zero = m.device
                break

        if len(mask_bce_losses) > 0:
            mask_bce_loss = torch.stack(mask_bce_losses).mean()
            mask_dice_loss = torch.stack(mask_dice_losses).mean()

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss * float(self.dice_scale)

            if self.fix_mean_loss and len(pred_masks) > 0:
                expected = float(len(pred_masks))
                actual = float(len(mask_bce_losses))
                if actual > 0:
                    scale = expected / actual
                    mask_bce_loss = mask_bce_loss * scale
                    mask_dice_loss = mask_dice_loss * scale
        else:
            device = device_for_zero if device_for_zero is not None else torch.device('cpu')
            mask_bce_loss = torch.tensor(0., device=device)
            mask_dice_loss = torch.tensor(0., device=device)

        loss = (self.loss_weight[0] * mask_bce_loss +
                self.loss_weight[1] * mask_dice_loss)
        

        # add aux outputs if present. scale each aux layer by 1/(i+1) to avoid explosion
        if 'aux_outputs' in pred:
            for i, aux_outputs in enumerate(pred['aux_outputs']):
                aux_loss = self.get_layer_loss(aux_outputs, insts, origin_queries)
                aux_weight = 1.0 / (i + 1)
                loss = loss + aux_weight * aux_loss
        return {'text_seg_loss': self.total_weight * loss}






@MODELS.register_module()
class TextPrompt_Criterion_Glamm_v2:
    def __init__(self, loss_weight, non_object_weight=None,
                 fix_dice_loss_weight=False, fix_mean_loss=False, total_weight=1.0,
                 use_cls_supervise=False, dice_scale=4.0):
        """
        loss_weight: [bce_weight, dice_weight]
        fix_dice_loss_weight: if True (or numeric), scale dice by dice_scale
        fix_mean_loss: legacy flag kept - implemented as optional rescaling by expected count
        dice_scale: numeric scale applied to dice when fix_dice_loss_weight is truthy
        """
        self.loss_weight = loss_weight
        self.fix_dice_loss_weight = fix_dice_loss_weight
        self.fix_mean_loss = fix_mean_loss
        self.total_weight = total_weight
        self.dice_scale = dice_scale if dice_scale is not None else 4.0

        self.cls_weight = 1.0 if use_cls_supervise else 0.0

    def _get_src_permutation_idx(self, indices):
        # same as original: helper to permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_layer_loss(self, aux_outputs, insts):
        """
        aux_outputs: dict with key 'masks' similar shape as main pred
        insts: list of Instance objects (with attribute sp_masks)
        origin_queries: list of origin_query tensors used to filter valid queries
        """
        pred_masks = aux_outputs.get('masks', [])

        mask_bce_losses = []
        mask_dice_losses = []

        # iterate over images (assumes pred_masks aligns with insts & origin_queries)
        for mask, inst in zip(pred_masks, insts):
            if mask is None:
                continue
            if len(inst) == 0:
                continue

            # ensure device/dtype alignment
            device = mask.device

            inst = inst.to(device)

            none_nan_mask_index = ~torch.isnan(mask).all(dim=1)
            none_nan_inst_index = ~torch.isnan(inst).all(dim=1)
            none_zero_inst_index = inst.sum(dim=1) != 0



            # Get their intersection, which is the valid mask index
            valida_ids = none_nan_mask_index & none_nan_inst_index & none_zero_inst_index

            if valida_ids.sum() == 0:
                continue
            
            pred_mask = mask[valida_ids]           # logits: [num_queries, pixels]
            tgt_mask = inst.to(device)[valida_ids]  # targets: same shape

            # compute losses for this image (over all matched queries)
            # if there are zero valid queries, skip
            if pred_mask.numel() == 0 or tgt_mask.numel() == 0:
                continue

            bce = F.binary_cross_entropy_with_logits(pred_mask, tgt_mask.float(), reduction='mean')
            dice = dice_loss(pred_mask, tgt_mask.float())

            mask_bce_losses.append(bce)
            mask_dice_losses.append(dice)

        device_for_zero = None
        if len(pred_masks):
            # get device from first available mask if possible
            for m in pred_masks:
                if m is not None:
                    device_for_zero = m.device
                    break

        if len(mask_bce_losses) > 0:
            # mean over valid images
            mask_bce_loss = torch.stack(mask_bce_losses).mean()
            mask_dice_loss = torch.stack(mask_dice_losses).mean()

            # optional dice scaling (legacy behavior preserved)
            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss * float(self.dice_scale)

            # optional fix_mean_loss: if enabled, rescale by ratio of expected to actual valid
            if self.fix_mean_loss and len(pred_masks) > 0:
                expected = float(len(pred_masks))
                actual = float(len(mask_bce_losses))
                if actual > 0:
                    scale = expected / actual
                    mask_bce_loss = mask_bce_loss * scale
                    mask_dice_loss = mask_dice_loss * scale
        else:
            device = device_for_zero if device_for_zero is not None else torch.device('cpu')
            mask_bce_loss = torch.tensor(0., device=device)
            mask_dice_loss = torch.tensor(0., device=device)

        loss = (self.loss_weight[0] * mask_bce_loss +
                self.loss_weight[1] * mask_dice_loss)
        return loss

    def __call__(self, pred, insts):
        """
        pred: dict containing 'masks' (list of per-image masks) and optionally 'aux_outputs'
        insts: list of ground-truth instances (each with .sp_masks and .labels_3d)
        origin_queries: list of origin_query tensors used to filter valid queries (NaN rows ignored)
        """
        pred_masks = pred.get('masks', [])

        # create trivial indices: 0..N-1 per image (keeps original behavior)
        # indices = []
        # for i in range(len(insts)):
        #     device = insts[i].labels_3d.device if hasattr(insts[i], 'labels_3d') else torch.device('cpu')
        #     n = 0
        #     if hasattr(insts[i], 'labels_3d'):
        #         n = len(insts[i].labels_3d)
        #     query_index = torch.arange(n, device=device)
        #     labels_index = query_index.clone()
        #     indices.append((query_index, labels_index))

        mask_bce_losses = []
        mask_dice_losses = []

        for mask, inst in zip(pred_masks, insts):
            if mask is None:
                continue
            if len(inst) == 0:
                continue

            device = mask.device
            # idx_q = idx_q.to(device)
            # idx_gt = idx_gt.to(device)

            pred_mask = mask
            tgt_mask = inst.to(device)

            # valida_ids: which queries in origin_query are valid (no NaN in any dim)


            none_nan_mask_index = ~torch.isnan(pred_mask).all(dim=1)
            none_nan_inst_index = ~torch.isnan(tgt_mask).all(dim=1)

            # check whether there is an all-zero gt
            none_zero_inst_index=tgt_mask.sum(dim=1) != 0


            # Get their intersection, which is the valid mask index
            valida_ids = none_nan_mask_index & none_nan_inst_index & none_zero_inst_index

            # if no valid query rows, skip
            if valida_ids.sum() == 0:
                continue

            pm_valid = pred_mask[valida_ids, :]
            tm_valid = tgt_mask[valida_ids, :]


            

            # compute losses (mean reduction)
            i_bce_loss = F.binary_cross_entropy_with_logits(pm_valid, tm_valid.float(), reduction='mean')
            i_dice_loss = dice_loss(pm_valid, tm_valid.float())

            mask_bce_losses.append(i_bce_loss)
            mask_dice_losses.append(i_dice_loss)

        # aggregate main losses
        device_for_zero = None
        for m in pred_masks:
            if m is not None:
                device_for_zero = m.device
                break

        if len(mask_bce_losses) > 0:
            mask_bce_loss = torch.stack(mask_bce_losses).mean()
            mask_dice_loss = torch.stack(mask_dice_losses).mean()

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss * float(self.dice_scale)

            if self.fix_mean_loss and len(pred_masks) > 0:
                expected = float(len(pred_masks))
                actual = float(len(mask_bce_losses))
                if actual > 0:
                    scale = expected / actual
                    mask_bce_loss = mask_bce_loss * scale
                    mask_dice_loss = mask_dice_loss * scale
        else:
            device = device_for_zero if device_for_zero is not None else torch.device('cpu')
            mask_bce_loss = torch.tensor(0., device=device)
            mask_dice_loss = torch.tensor(0., device=device)

        loss = (self.loss_weight[0] * mask_bce_loss +
                self.loss_weight[1] * mask_dice_loss)
        

        # add aux outputs if present. scale each aux layer by 1/(i+1) to avoid explosion
        if 'aux_outputs' in pred:
            for i, aux_outputs in enumerate(pred['aux_outputs']):
                aux_loss = self.get_layer_loss(aux_outputs, insts)
                aux_weight = 1.0 / (i + 1)
                loss = loss + aux_weight * aux_loss

        return {'text_seg_loss': self.total_weight * loss}