import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np


from mmcv.cnn import Scale
from mmcv.runner import force_fp32

from mmdet.core import distance2bbox, multi_apply, multiclass_nms, reduce_mean, build_assigner
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead

INF = 1e8

@HEADS.register_module()
class DNTOD_FCOSHead(AnchorFreeHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 fpn_layer = 'p3', # new
                 fraction = 1/3,    # new
                 w1 = 1.0, # new
                 cls_alpha = 1.0, # new
                 TLS = [True, True],  # new
                 RBR = True,  # new
                 with_CLC = True, # new
                 store_path = '/xxx/xxx/DN-TOD/train_detail/dict', # new
                 frozen_iter = 10000,  # new
                 samples = 10000, # new
                 strategy = 0,  # new
                 threshold = (1, 1),  # new
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        # RFLA
        self.fpn_layer = fpn_layer  # new
        self.fraction = fraction    # new
        # TLS head
        self.w1 = w1
        self.cls_alpha = cls_alpha
        self.csr = TLS
        self.rbr = RBR
        self.dict_last_GT_1 = {}
        self.store_path = store_path
        # CLC head
        self.with_ALC = with_CLC
        self.strategy = strategy
        self.samples = samples
        self.threshold = threshold
        self.frozen_iter = frozen_iter
        self.iters = 0
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.confidence_matrix = torch.zeros(self.samples, self.num_classes, self.num_classes).float()
        self.memory = torch.full((self.num_classes, self.num_classes), float('nan')).float()
        self.loss_centerness = build_loss(loss_centerness)
        self.assigner = build_assigner(self.train_cfg.assigner) # 不同于传统的MaxIou

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                centernesses (list[Tensor]): centerness for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = F.relu(bbox_pred)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        return cls_score, bbox_pred, centerness

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        def DCM(bbox_preds):
            threshold_matrix = torch.ones(self.num_classes, self.num_classes).float().to(bbox_preds[0].device)
            if self.iters >= self.frozen_iter:
                mean = torch.nanmean(self.confidence_matrix, axis=0).to(bbox_preds[0].device)
                temp = self.confidence_matrix.numpy()
                std = torch.Tensor(np.nanstd(temp, ddof=1, axis=0)).float().to(bbox_preds[0].device)
                threshold_matrix = self.threshold[0] * mean + self.threshold[1] * std
                for i in range(0, self.num_classes):
                    for j in range(0, self.num_classes):
                        if (torch.isnan(threshold_matrix[i,j]) == True) and (torch.isnan(self.memory[i,j]) == False):
                            threshold_matrix[i,j] = self.memory[i,j]
                        elif (torch.isnan(threshold_matrix[i,j]) == True) and (torch.isnan(self.memory[i,j]) == True):
                            threshold_matrix[i,j] = 0.5
                        elif torch.isnan(threshold_matrix[i,j]) == False:
                            self.memory[i,j] = threshold_matrix[i,j]
                        threshold_matrix[i,j] = max(threshold_matrix[i,j], 0.1)
            return threshold_matrix
        
        def label_assignment(cls_scores, bbox_preds, centernesses, all_level_points, gt_bboxes, gt_labels, img_metas):
            flatten_cls_scores_img = [
                cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
                for cls_score in cls_scores
            ]
            flatten_bbox_preds_img = [
                bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
                for bbox_pred in bbox_preds
            ]
            flatten_centerness_img = [
                centerness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                for centerness in centernesses
            ]
            
            labels, bbox_targets, weights, new_GT_list = self.get_targets(all_level_points, gt_bboxes,
                                                    gt_labels, img_metas, flatten_cls_scores_img, flatten_bbox_preds_img, flatten_centerness_img)
            return labels, bbox_targets, weights, new_GT_list
        
        def CLC_statics(flatten_cls_scores, flatten_labels):
            confidence_matrix = torch.zeros(self.num_classes, self.num_classes).float()
            s_temp = []
            for i in range(self.num_classes):
                s_temp.append([])
            for i in range(0, len(pos_inds)):
                score = flatten_cls_scores[pos_inds[i],:].sigmoid().tolist()
                confidence_index = int(flatten_labels[pos_inds[i]])
                s_temp[confidence_index].append(score)
            for i in range(self.num_classes):
                temp = torch.tensor(s_temp[i]).float()
                confidence_matrix[i,:] = torch.mean(temp, axis=0) 
            self.confidence_matrix[(self.iters-1)%self.samples,:,:] = confidence_matrix

        def CLC_reweighting(weight_input_alc, pos_inds, flatten_cls_scores, flatten_labels, threshold_matrix):
            del_need = []
            new_label = []
            pos_inds_list = pos_inds.tolist()
            if self.with_ALC == True:
                if (self.iters-1) >= self.frozen_iter: 
                    for i in range(0, len(pos_inds_list)):
                        confidence_index = int(flatten_labels[pos_inds_list[i]])
                        a = flatten_cls_scores[pos_inds_list[i],:].sigmoid().tolist()
                        sorted_id = sorted(range(len(a)), key=lambda k: a[k], reverse=True) 
                        if sorted_id[0] != confidence_index:
                            for j in range(0, len(sorted_id)):
                                j_index = sorted_id[j]
                                confidence_j = float(a[j_index])
                                if j_index == confidence_index:
                                    break
                                if (confidence_j >= threshold_matrix[confidence_index, j_index]) and (confidence_j >= threshold_matrix[j_index, j_index]):
                                    del_need.append(pos_inds_list[i])
                                    new_label.append(j_index)
                                    break 
                    if self.strategy == 1:
                        for i in range(len(del_need)):
                            flatten_labels[del_need[i]] = new_label[i]
                    elif self.strategy == 0:
                        weight_input_alc[del_need] = 0   
            return weight_input_alc, flatten_labels

        threshold_matrix = DCM(bbox_preds)
        self.iters = self.iters + 1
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        num_imgs = cls_scores[0].size(0)
        
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]

        labels, bbox_targets, weights, new_GT_list = label_assignment(cls_scores, bbox_preds, centernesses, all_level_points, gt_bboxes, gt_labels, img_metas)
        
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_weights = torch.cat(weights)

        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        
        # CLC statics
        CLC_statics(flatten_cls_scores, flatten_labels)
        # CLC reweighting
        weight_input_alc = torch.ones_like(flatten_labels).long().to(bbox_preds[0].device)
        weight_input_alc, flatten_labels = CLC_reweighting(weight_input_alc, pos_inds, flatten_cls_scores, flatten_labels, threshold_matrix)   
        
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, weight=flatten_weights*weight_input_alc, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)

        cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
        bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]
        centerness_pred_list = [
            centernesses[i].detach() for i in range(num_levels)
        ]
        if torch.onnx.is_in_onnx_export():
            assert len(
                img_metas
            ) == 1, 'Only support one input image while in exporting to ONNX'
            img_shapes = img_metas[0]['img_shape_for_onnx']
        else:
            img_shapes = [
                img_metas[i]['img_shape']
                for i in range(cls_scores[0].shape[0])
            ]
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]
        result_list = self._get_bboxes(cls_score_list, bbox_pred_list,
                                       centerness_pred_list, mlvl_points,
                                       img_shapes, scale_factors, cfg, rescale,
                                       with_nms)
        return result_list

    def _get_bboxes(self,
                    cls_scores,
                    bbox_preds,
                    centernesses,
                    mlvl_points,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False,
                    with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (N, num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (N, num_points, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shapes (list[tuple[int]]): Shape of the input image,
                list[(height, width, 3)].
            scale_factors (list[ndarray]): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1), device=device, dtype=torch.long)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(0, 2, 3,
                                            1).reshape(batch_size,
                                                       -1).sigmoid()

            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)
            points = points.expand(batch_size, -1, 2)
            # Get top-k prediction
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:
                max_scores, _ = (scores * centerness[..., None]).max(-1)
                _, topk_inds = max_scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds).long()
                # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
                if torch.onnx.is_in_onnx_export():
                    transformed_inds = bbox_pred.shape[
                        1] * batch_inds + topk_inds
                    points = points.reshape(-1,
                                            2)[transformed_inds, :].reshape(
                                                batch_size, -1, 2)
                    bbox_pred = bbox_pred.reshape(
                        -1, 4)[transformed_inds, :].reshape(batch_size, -1, 4)
                    scores = scores.reshape(
                        -1, self.num_classes)[transformed_inds, :].reshape(
                            batch_size, -1, self.num_classes)
                    centerness = centerness.reshape(
                        -1, 1)[transformed_inds].reshape(batch_size, -1)
                else:
                    points = points[batch_inds, topk_inds, :]
                    bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                    scores = scores[batch_inds, topk_inds, :]
                    centerness = centerness[batch_inds, topk_inds]

            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
                scale_factors).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        batch_mlvl_centerness = torch.cat(mlvl_centerness, dim=1)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        if torch.onnx.is_in_onnx_export() and with_nms:
            from mmdet.core.export import add_dummy_nms_for_onnx
            batch_mlvl_scores = batch_mlvl_scores * (
                batch_mlvl_centerness.unsqueeze(2))
            max_output_boxes_per_class = cfg.nms.get(
                'max_output_boxes_per_class', 200)
            iou_threshold = cfg.nms.get('iou_threshold', 0.5)
            score_threshold = cfg.score_thr
            nms_pre = cfg.get('deploy_nms_pre', -1)
            return add_dummy_nms_for_onnx(batch_mlvl_bboxes, batch_mlvl_scores,
                                          max_output_boxes_per_class,
                                          iou_threshold, score_threshold,
                                          nms_pre, cfg.max_per_img)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = batch_mlvl_scores.new_zeros(batch_size,
                                              batch_mlvl_scores.shape[1], 1)
        batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores,
                 mlvl_centerness) in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                         batch_mlvl_centerness):
                det_bbox, det_label = multiclass_nms(
                    mlvl_bboxes,
                    mlvl_scores,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img,
                    score_factors=mlvl_centerness)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                   batch_mlvl_centerness)
            ]
        return det_results

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    def get_targets(self, points, gt_bboxes_list, gt_labels_list, img_metas, flatten_cls_scores, flatten_bbox_preds, flatten_centerness):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        
        # convert points to rf according to its layer
        rfields = []
        trfs = self.gen_trf()
        # the j_i = [1, 2, 4, 8, 16, 32, 64, 128]
        # r = [r0, r1, r2, r3, r4, r5]
        # r = [1, 7, 11, 43, 107, 299]
        # trfs = [35, 91, 267, 427, 555, 811]
        # fpn_layer = 'p3', # bottom FPN layer P3
        # fraction = 1/2,
        for num in range(len(points)):
            rfield=[]
            if self.fpn_layer == 'p3':
                rfnum = num +1
            else:
                rfnum = num

            if rfnum == 0:
                rf = trfs[0]*self.fraction
            elif rfnum == 1:
                rf = trfs[1]*self.fraction
            elif rfnum == 2:
                rf = trfs[2]*self.fraction
            elif rfnum == 3:
                rf = trfs[3]*self.fraction 
            elif rfnum == 4:
                rf = trfs[4]*self.fraction
            else:
                rf = trfs[5]*self.fraction
                      
            point = points[num]
            px1 = point[...,0] - rf/2
            py1 = point[...,1] - rf/2
            px2 = point[...,0] + rf/2
            py2 = point[...,1] + rf/2
            rfield = torch.cat((px1[...,None], py1[...,None]), dim=1)
            rfield = torch.cat((rfield, px2[...,None]), dim=1)
            rfield = torch.cat((rfield, py2[...,None]), dim=1)
            rfields.append(rfield)
        rfields = torch.cat(rfields, dim=0)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(  #[[-1,64],[-1,64]] num*2*2
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0) 
        concat_points = torch.cat(points, dim=0)
        # flatten_cls_scores, flatten_bbox_preds, flatten_centerness
        concat_cls_scores = []
        concat_bbox_preds = []
        concat_centerness = []
        for i in range(flatten_cls_scores[0].shape[0]):
            temp_scores = []
            temp_preds = []
            temp_centerness = []
            for j in range(num_levels):
                temp_scores.append(flatten_cls_scores[j][i,:,:])
                temp_preds.append(flatten_bbox_preds[j][i,:,:])
                temp_centerness.append(flatten_centerness[j][i,:])
            concat_cls_scores.append(torch.cat(temp_scores, dim=0))
            concat_bbox_preds.append(torch.cat(temp_preds, dim=0))
            concat_centerness.append(torch.cat(temp_centerness, dim=0))

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]
        
        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, weight_input_list, temp_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            img_metas,
            concat_cls_scores,
            concat_bbox_preds,
            concat_centerness,
            points_ori=points,
            points=concat_points,
            rfields=rfields, 
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        weight_input_list = [weights.split(num_points, 0) for weights in weight_input_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_weights = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_weights.append(
                torch.cat([weights[i] for weights in weight_input_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_weights, temp_list

    def _get_target_single(self, gt_bboxes, gt_labels, img_metas, flatten_cls_scores, flatten_bbox_preds, flatten_centerness,
                           points_ori, points, rfields, regress_ranges, num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        imagefile = img_metas['ori_filename']
        num_points_list = [center.size(0) for center in points_ori]

        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        gt_ori = gt_bboxes
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1) # points, gts
        
        regress_ranges_clone = regress_ranges.clone()
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0] #numpoints, num_gt
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1) # numpoints, num_gt, 4

        assign_result = self.assigner.assign(rfields, gt_ori, gt_bboxes_ignore=None)
        inds = assign_result.gt_inds #num_points
        inds = inds - 1

        inds_mask = inds[...,None].repeat(1, num_gts).to(flatten_bbox_preds.device)
        point_mask = torch.arange(num_gts).repeat(num_points, 1).to(flatten_bbox_preds.device)

        assigned_mask = (inds_mask == point_mask)
        areas[assigned_mask == False] = INF
        min_area, min_area_inds = areas.min(dim=1)
 
        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().reshape(-1)
        neg_inds = (labels == bg_class_ind).nonzero().reshape(-1)
        pos_masks = min_area_inds[pos_inds]
        weight_input = torch.ones_like(labels).float()
        flatten_cls_scores_clone = flatten_cls_scores.clone().detach().sigmoid()
        flatten_bbox_preds_clone = flatten_bbox_preds.clone().detach()

        if imagefile in self.dict_last_GT_1.keys():
            path_json_temp = os.path.join(self.store_path, imagefile + '.npy')
            data = np.load(path_json_temp)
            max_pred_last = torch.tensor(data, device=flatten_cls_scores.device)
            if self.csr[0] == True:
                for i in range(len(gt_labels)):
                    gt_labels_now = gt_labels[i]
                    N_target = gt_ori[i].clone()
                    pos_GT_inds = (pos_masks == i).nonzero().reshape(-1)
                    pos_GT_inds = pos_inds[pos_GT_inds]
                    pos_GT_cls_pred = flatten_cls_scores_clone[pos_GT_inds, gt_labels_now]
                    if pos_GT_cls_pred.numel() == 0:
                        continue
                    pos_GT_cls_pred_last = max_pred_last[pos_GT_inds, gt_labels_now]
                    weight_1 = 1 - pos_GT_cls_pred_last / pos_GT_cls_pred
                    inds_plus_zero = (weight_1 > 0).nonzero().reshape(-1)
                    if inds_plus_zero.numel() == 0:
                        inds_minus_zero = (weight_1 <= 0).nonzero().reshape(-1)
                        weight_1[inds_minus_zero] = 1.0
                        weight_1 = torch.where(torch.isnan(weight_1), torch.full_like(weight_1, 1.0), weight_1)
                    else:
                        min_neg = torch.min(weight_1[inds_plus_zero])
                        inds_minus_zero = (weight_1 <= 0).nonzero().reshape(-1)
                        weight_1[inds_minus_zero] = min_neg / 2
                        weight_1 = weight_1 / torch.max(weight_1)
                        weight_1 = torch.where(torch.isnan(weight_1), torch.full_like(weight_1, 1.0), weight_1)

                    weight_2 = pos_GT_cls_pred / torch.sum(pos_GT_cls_pred)
                    weight_2 = weight_2 / torch.max(weight_2)
                    weight_2 = torch.where(torch.isnan(weight_2), torch.full_like(weight_2, 1.0), weight_2)

                    weight_all = self.cls_alpha * weight_1 + (1-self.cls_alpha) * weight_2
                    weight_all = weight_all / torch.max(weight_all)
                    weight_input[pos_GT_inds] = weight_all
            if self.csr[1] == True:
                neg_cls_pred = (flatten_cls_scores_clone[neg_inds,:]).max(1)[0]
                neg_cls_pred_last = (max_pred_last[neg_inds,:]).max(1)[0]
                weight_1 = neg_cls_pred_last / neg_cls_pred
                inds_one = (weight_1 >= 1).nonzero().reshape(-1)
                weight_1[inds_one] = 1.0
                weight_1 = torch.where(torch.isnan(weight_1), torch.full_like(weight_1, 1.0), weight_1)
                weight_input[neg_inds] = weight_1
        
        # For regression
        flatten_bbox_preds_clone = list(flatten_bbox_preds_clone.split(num_points_list, 0))
        for i in range(len(num_points_list)):
            flatten_bbox_preds_clone[i] = flatten_bbox_preds_clone[i] * self.strides[i]
        flatten_bbox_preds_clone = torch.cat(flatten_bbox_preds_clone)
        GT_j = []
        grades_j = []
        for i in range(len(gt_labels)):
            gt_labels_now = gt_labels[i]
            N_target = gt_ori[i].clone()
            pos_GT_inds = (pos_masks == i).nonzero().reshape(-1)
            pos_GT_inds = pos_inds[pos_GT_inds]
            pos_GT_cls_pred = flatten_cls_scores_clone[pos_GT_inds, gt_labels_now]
            if pos_GT_cls_pred.numel() == 0:
                GT_j.append(N_target.tolist())
                grades_j.append(1.0)
                continue
            else:
                topk_values, topk_indices = torch.topk(pos_GT_cls_pred, pos_GT_cls_pred.numel())
                denominator = 0
                for xyz in range(pos_GT_cls_pred.numel()):
                    denominator = denominator + topk_values[xyz]
                pred_temp = []
                grade_temp = []
                for mn in range(pos_GT_cls_pred.numel()):
                    inds = pos_GT_inds[topk_indices[mn]]
                    R_target = flatten_bbox_preds_clone[inds,:].clone().tolist()
                    x1, y1 = points[inds,:].tolist()
                    if mn == 0: 
                        pred_temp.append([x1-R_target[0], y1-R_target[1], R_target[2]+x1, R_target[3]+y1])
                        grade_temp.append(float(topk_values[0]/denominator))
                        grades_j.append(float(topk_values[0]))
                    else:
                        pred_temp.append([x1-R_target[0], y1-R_target[1], R_target[2]+x1, R_target[3]+y1])
                        grade_temp.append(float(topk_values[mn]/denominator))
                pred_temp = torch.tensor(pred_temp, device=flatten_bbox_preds.device)
                temp_GT_j = grade_temp[0]*pred_temp[0,:]
                for xyz in range(len(grade_temp)-1):
                    temp_GT_j = temp_GT_j + grade_temp[xyz+1]*pred_temp[xyz+1,:]
                GT_j.append((temp_GT_j).tolist())
                if torch.isnan(torch.tensor(GT_j, device=flatten_bbox_preds.device)).any():
                    GT_j[i] = N_target.tolist()
                    grades_j[i] = 1.0  
        GT_j = torch.tensor(GT_j, device=gt_bboxes.device)
        grades_j = torch.tensor(grades_j, device=gt_bboxes.device)
        
        if imagefile in self.dict_last_GT_1.keys():
            path_json_temp = os.path.join(self.store_path, imagefile + '_GT.npy')
            data = np.load(path_json_temp)
            GT_pred_last = torch.tensor(data, device=flatten_cls_scores.device)
            path_json_temp = os.path.join(self.store_path, imagefile + '_grades.npy')
            data = np.load(path_json_temp)
            grades_pred_last = torch.tensor(data, device=flatten_cls_scores.device)
            
            grade_ori = self.w1 * torch.ones_like(grades_j).float()
            grade_last = grades_pred_last
            grade_now = grades_j
            grade_all = grade_ori + grade_last + grade_now
            temp = ((grade_ori/grade_all).reshape(-1,1).expand(gt_ori.shape[0], gt_ori.shape[1])) * gt_ori + ((grade_last/grade_all).reshape(-1,1).expand(gt_ori.shape[0], gt_ori.shape[1])) * GT_pred_last + ((grade_now/grade_all).reshape(-1,1).expand(gt_ori.shape[0], gt_ori.shape[1])) * GT_j
        else:
            grade_ori = self.w1 * torch.ones_like(grades_j).float()
            grade_now = grades_j
            grade_all = grade_ori + grade_now
            temp = ((grade_ori/grade_all).reshape(-1,1).expand(gt_ori.shape[0], gt_ori.shape[1])) * gt_ori + ((grade_now/grade_all).reshape(-1,1).expand(gt_ori.shape[0], gt_ori.shape[1])) * GT_j
        
        # store 
        self.dict_last_GT_1[imagefile] = 1
        # scores
        array_save = np.array(flatten_cls_scores_clone.cpu())
        path_img = os.path.join(self.store_path, imagefile)
        np.save(path_img, arr=array_save)
        # GT
        array_save = np.array(temp.cpu())
        path_img = os.path.join(self.store_path, imagefile + '_GT')
        np.save(path_img, arr=array_save)
        # grades
        array_save = np.array(grades_j.cpu())
        path_img = os.path.join(self.store_path, imagefile + '_grades')
        np.save(path_img, arr=array_save)
        
        if self.rbr:
            areas = (temp[:, 2] - temp[:, 0]) * (
                temp[:, 3] - temp[:, 1])
            areas = areas[None].repeat(num_points, 1) # points, gts
            gt_bboxes = temp[None].expand(num_points, num_gts, 4)
            left = xs - gt_bboxes[..., 0] #numpoints, num_gt
            right = gt_bboxes[..., 2] - xs
            top = ys - gt_bboxes[..., 1]
            bottom = gt_bboxes[..., 3] - ys
            
            bbox_targets = torch.stack((left, top, right, bottom), -1) # numpoints, num_gt, 4

            assign_result = self.assigner.assign(rfields, temp, gt_bboxes_ignore=None)
            inds = assign_result.gt_inds #num_points
            inds = inds - 1

            inds_mask = inds[...,None].repeat(1, num_gts).to(flatten_bbox_preds.device)
            point_mask = torch.arange(num_gts).repeat(num_points, 1).to(flatten_bbox_preds.device)

            assigned_mask = (inds_mask == point_mask)
            areas[assigned_mask == False] = INF
            min_area, min_area_inds = areas.min(dim=1)
    
            labels = gt_labels[min_area_inds]
            labels[min_area == INF] = self.num_classes  # set as BG
            bbox_targets = bbox_targets[range(num_points), min_area_inds]
            return labels, bbox_targets, weight_input, temp
        return labels, bbox_targets, weight_input, temp

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0].clamp(min=0.01) / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0].clamp(min=0.01) / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    def gen_trf(self):
        '''
        Calculate the theoretical receptive field from P2-p7 of a standard ResNet-50-FPN.
        # ref: https://distill.pub/2019/computing-receptive-fields/
        '''

        j_i = [1]
        for i in range(7):
            j = j_i[i]*2
            j_i.append(j)
        # the j_i = [1, 2, 4, 8, 16, 32, 64, 128]
        # r = [r0, r1, r2, r3, r4, r5]
        # r = [1, 7, 11, 43, 107, 299]
        # trfs = [35, 91, 267, 427, 555, 811]
        r0 = 1
        r1 = r0 + (7-1)*j_i[0]  
        
        r2 = r1 + (3-1)*j_i[1]
        trf_p2 = r2 + (3-1)*j_i[2]*3

        r3 = trf_p2 + (3-1)*j_i[2]
        trf_p3 = r3 + (3-1)*j_i[3]*3

        r4 = trf_p3 + (3-1)*j_i[3]
        trf_p4 = r4 + (3-1)*j_i[4]*5

        r5 = trf_p4 + (3-1)*j_i[4]
        trf_p5 = r5 + (3-1)*j_i[5]*2
 
        trf_p6 = trf_p5 + (3-1)*j_i[6]

        trf_p7 = trf_p6 + (3-1)*j_i[7]

        trfs = [trf_p2, trf_p3, trf_p4, trf_p5, trf_p6, trf_p7]

        return trfs
