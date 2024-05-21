import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


def cross_entropy(pred,
                  label,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    loss = F.cross_entropy(pred, label, weight=class_weight, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_onehot_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(
        (labels >= 0) & (labels < label_channels), as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1

    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)

    return bin_labels, bin_label_weights


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    if pred.dim() != label.dim():
        label, weight = _expand_onehot_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss

def soft_binary_cross_entropy(pred,
                              label,
                              weight=None,
                              reduction='mean',
                              avg_factor=None,
                              class_weight=None):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss

def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    if pred.dim() != label.dim():
        label, weight = _expand_onehot_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C, *), C is the
            number of classes. The trailing * indicates arbitrary shape.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss

    Example:
        >>> N, C = 3, 11
        >>> H, W = 2, 2
        >>> pred = torch.randn(N, C, H, W) * 1000
        >>> target = torch.rand(N, H, W)
        >>> label = torch.randint(0, C, size=(N,))
        >>> reduction = 'mean'
        >>> avg_factor = None
        >>> class_weights = None
        >>> loss = mask_cross_entropy(pred, target, label, reduction,
        >>>                           avg_factor, class_weights)
        >>> assert loss.shape == (1,)
    """
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction='mean')[None]

    

@LOSSES.register_module()
class LNLLoss(nn.Module):

    def __init__(self,
                 loss_type='GCE',
                 use_sigmoid=False,
                 q=0.7,
                 alpha=6.0,
                 beta=0.1,
                 gamma=0.5,
                 size_average=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super(LNLLoss, self).__init__()
        LNL_loss_bank = ['MAE', 'GCE', 'RCE', 'SCE', 'DMI', 'NCE', 'NFL', 'NCEandMAE', 'NCEandRCE', 'NFLandMAE', 'NFLandRCE']
        function_bank = [MAEloss, GCELoss, RCELoss, SCELoss, DMILoss, NCELoss, NFLLoss, NCEandMAELoss, NCEandRCELoss, NFLandMAELoss, NFLandRCELoss]
        assert (loss_type in LNL_loss_bank)

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        self.q = q
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.size_average = size_average
        self.cls_criterion = function_bank[LNL_loss_bank.index(loss_type)]

        
    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight, device=cls_score.device)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            self.q,
            self.alpha,
            self.beta,
            self.gamma,
            self.size_average,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls


def MAEloss(preds,
            labels,
            q=0.7,
            alpha=6.0,
            beta=0.1,
            gamma=0.5,
            size_average=True,
            weight=None,
            reduction='mean',
            avg_factor=None,
            class_weight=None):
    """
        MAE: Mean Absolute Error
        2017 AAAI | Robust Loss Functions under Label Noise for Deep Neural Networks
        Ref: https://github.com/HanxunH/Active-Passive-Losses/blob/master/loss.py
    """
    num_classes = preds.shape[1]
    pred = F.softmax(preds, dim=1)
    # pred = preds.sigmoid()
    pred = torch.clamp(pred, min=1e-7, max=1.0)
    label_oh = F.one_hot(labels.long(), num_classes).float()
    loss = 1. - torch.sum(label_oh * pred, dim=1)

    # element-wise losses
    loss = loss.mean()

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def GCELoss(preds,
            labels,
            q=0.7,
            alpha=6.0,
            beta=0.1,
            gamma=0.5,
            size_average=True,
            weight=None,
            reduction='mean',
            avg_factor=None,
            class_weight=None):
    """
        GCE: Generalized Cross Entropy
        2018 NeurIPS | Generalized cross entropy loss for training deep neural networks with noisy labels
        Ref: https://github.com/AlanChou/Truncated-Loss/blob/master/TruncatedLoss.py
    """
    num_classes = preds.shape[1]
    pred = F.softmax(preds, dim=1)
    # pred = preds.sigmoid()
    pred = torch.clamp(pred, min=1e-7, max=1.0)

    Yg = torch.gather(pred, 1, torch.unsqueeze(labels, 1).long())
    loss = ((1-(Yg**q))/q)

    # element-wise losses
    # loss = torch.mean(loss)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss

def RCELoss(preds,
            labels,
            q=0.7,
            alpha=6.0,
            beta=0.1,
            gamma=0.5,
            size_average=True,
            weight=None,
            reduction='mean',
            avg_factor=None,
            class_weight=None):
    """
        RCE: Reverse Cross Entropy
        Ref: https://github.com/HanxunH/Active-Passive-Losses/blob/master/loss.py
    """
    num_classes = preds.shape[1]
    pred = F.softmax(preds, dim=1)
    # pred = preds.sigmoid()
    pred = torch.clamp(pred, min=1e-7, max=1.0)

    label_one_hot = torch.nn.functional.one_hot(labels, num_classes).float().to(preds.device)
    label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
    rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

    # element-wise losses
    loss = rce.mean()

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def SCELoss(preds,
            labels,
            q=0.7,
            alpha=6.0,
            beta=0.1,
            gamma=0.5,
            size_average=True,
            weight=None,
            reduction='mean',
            avg_factor=None,
            class_weight=None):
    """
        SCE: Symmetric Cross Entropy
        2019 ICCV | Symmetric cross entropy for robust learning with noisy labels
        Ref: https://github.com/HanxunH/SCELoss-Reproduce/blob/master/loss.py
    """
    num_classes = preds.shape[1]
    # CCE
    # ce = torch.nn.CrossEntropyLoss(preds, labels.long())
    ce = F.cross_entropy(preds, labels, weight=class_weight, reduction='none')
    # RCE
    pred = F.softmax(preds, dim=1)
    pred = torch.clamp(pred, min=1e-7, max=1.0)
    label_one_hot = F.one_hot(labels.long(), num_classes).float()
    label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
    rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
    # Loss
    loss = alpha * ce + beta * rce.mean()

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss

def DMILoss(preds,
            labels,
            q=0.7,
            alpha=6.0,
            beta=0.1,
            gamma=0.5,
            size_average=True,
            weight=None,
            reduction='mean',
            avg_factor=None,
            class_weight=None):
    """
        DMI: Determinant based Mutual Information Loss
        2019 NeurlIPS | L_dmi: A novel information-theoretic loss function for training deep nets robust to label noise
        Ref: https://github.com/Newbeeer/L_DMI/blob/master/CIFAR-10/DMI.py
    """
    num_classes = preds.shape[1]
    pred = F.softmax(preds, dim=1)
    pred = torch.clamp(pred, min=1e-7, max=1.0) 
    label_one_hot = F.one_hot(labels.long(), num_classes).float()
    label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
    label_one_hot = label_one_hot.transpose(0, 1)
    mat = label_one_hot @ pred
    loss = -1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.001) 
    ce = F.cross_entropy(preds, labels, weight=class_weight, reduction='none')

    loss = alpha * ce + beta * loss
    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss

def NCELoss(preds,
            labels,
            q=0.7,
            alpha=6.0,
            beta=0.1,
            gamma=0.5,
            size_average=True,
            weight=None,
            reduction='mean',
            avg_factor=None,
            class_weight=None):
    """
        NCELoss: Normalized Cross Entropy
        2020 ICML | Normalized loss functions for deep learning with noisy labels
        Ref: https://github.com/HanxunH/Active-Passive-Losses/blob/master/loss.py
    """
    num_classes = preds.shape[1]
    pred = F.log_softmax(preds, dim=1)
    label_one_hot = F.one_hot(labels.long(), num_classes).float()
    nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))

    # loss = nce.mean()
    loss = nce

    ce = F.cross_entropy(preds, labels, weight=class_weight, reduction='none')
    loss = alpha * ce + beta * loss

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss

def NFLLoss(preds,
            labels,
            q=0.7,
            alpha=6.0,
            beta=0.1,
            gamma=0.5,
            size_average=True,
            weight=None,
            reduction='mean',
            avg_factor=None,
            class_weight=None):
    """
        NFLLoss: Normalized Focal Loss
        2020 ICML | Normalized loss functions for deep learning with noisy labels
        Ref: https://github.com/HanxunH/Active-Passive-Losses/blob/master/loss.py
    """
    num_classes = preds.shape[1]
    target = labels.view(-1, 1)
    logpt = F.log_softmax(preds, dim=1)
    normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** gamma * logpt, dim=1)
    logpt = logpt.gather(1, target.long())
    logpt = logpt.view(-1)
    pt = torch.autograd.Variable(logpt.data.exp())
    loss = -1 * (1-pt)**gamma * logpt
    loss = loss / normalizor

    if size_average:
        loss = loss.mean()
    else:
        loss = loss.sum()

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


def NCEandMAELoss(preds,
                labels,
                q=0.7,
                alpha=6.0,
                beta=0.1,
                gamma=0.5,
                size_average=True,
                weight=None,
                reduction='mean',
                avg_factor=None,
                class_weight=None):
    """
        NCEandMAE: APL - Normalized Cross Entropy + MAE Loss
        2020 ICML | Normalized loss functions for deep learning with noisy labels
        Ref: https://github.com/HanxunH/Active-Passive-Losses/blob/master/loss.py
    """
    loss = 0.5 * NCELoss(preds, labels, q, alpha, beta, gamma, size_average, weight, reduction, avg_factor, class_weight) + \
                0.5 * MAEloss(preds, labels, q, alpha, beta, gamma, size_average, weight, reduction, avg_factor, class_weight)
    ce = F.cross_entropy(preds, labels, weight=class_weight, reduction='none')
    loss = alpha * ce + beta * loss
    return loss

def NCEandRCELoss(preds,
                labels,
                q=0.7,
                alpha=6.0,
                beta=0.1,
                gamma=0.5,
                size_average=True,
                weight=None,
                reduction='mean',
                avg_factor=None,
                class_weight=None):
    """
        NCEandRCE: APL - Normalized Cross Entropy + Reverse Cross Entropy
        2020 ICML | Normalized loss functions for deep learning with noisy labels
        Ref: https://github.com/HanxunH/Active-Passive-Losses/blob/master/loss.py
    """
    loss = 0.5 * NCELoss(preds, labels, q, alpha, beta, gamma, size_average, weight, reduction, avg_factor, class_weight) + \
                0.5 * RCELoss(preds, labels, q, alpha, beta, gamma, size_average, weight, reduction, avg_factor, class_weight)
    ce = F.cross_entropy(preds, labels, weight=class_weight, reduction='none')
    loss = alpha * ce + beta * loss
    return loss

def NFLandMAELoss(preds,
                labels,
                q=0.7,
                alpha=6.0,
                beta=0.1,
                gamma=0.5,
                size_average=True,
                weight=None,
                reduction='mean',
                avg_factor=None,
                class_weight=None):
    """
        NFLandMAE: APL - Normalized Focal Loss + MAE Loss
        2020 ICML | Normalized loss functions for deep learning with noisy labels
        Ref: https://github.com/HanxunH/Active-Passive-Losses/blob/master/loss.py
    """   
    loss = 0.5 * NFLLoss(preds, labels, q, alpha, beta, gamma, size_average, weight, reduction, avg_factor, class_weight) + \
                0.5 * MAEloss(preds, labels, q, alpha, beta, gamma, size_average, weight, reduction, avg_factor, class_weight)
    ce = F.cross_entropy(preds, labels, weight=class_weight, reduction='none')
    loss = alpha * ce + beta * loss
    return loss

def NFLandRCELoss(preds,
                labels,
                q=0.7,
                alpha=6.0,
                beta=0.1,
                gamma=0.5,
                size_average=True,
                weight=None,
                reduction='mean',
                avg_factor=None,
                class_weight=None):
    """
        NFLandRCE: APL - Normalized Focal Loss + Reverse Cross Entropy
        2020 ICML | Normalized loss functions for deep learning with noisy labels
        Ref: https://github.com/HanxunH/Active-Passive-Losses/blob/master/loss.py
    """  
    loss = 0.5 * NFLLoss(preds, labels, q, alpha, beta, gamma, size_average, weight, reduction, avg_factor, class_weight) + \
                0.5 * RCELoss(preds, labels, q, alpha, beta, gamma, size_average, weight, reduction, avg_factor, class_weight)
    ce = F.cross_entropy(preds, labels, weight=class_weight, reduction='none')
    loss = alpha * ce + beta * loss
    return loss