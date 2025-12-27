import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.models.dense_heads import RPNHead
from mmdet.utils.typing_utils import MultiConfig
from mmcv.cnn import ConvModule
from mmdet.structures.bbox import get_box_tensor
from typing import List


@MODELS.register_module()
class MutipRPNHead(RPNHead):
    def __init__(self,
                 in_channels: int,
                 num_classes: int = 1,
                 init_cfg: MultiConfig = dict(
                     type='Normal', layer='Conv2d', std=0.01),
                 num_convs: int = 1,
                 **kwargs) -> None:
        self.num_convs = num_convs
        super(RPNHead, self).__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        if self.num_convs > 1:
            rpn_convs = []
            for i in range(self.num_convs):
                if i == 0:
                    in_channels = self.in_channels
                else:
                    in_channels = self.feat_channels
                # use ``inplace=False`` to avoid error: one of the variables
                # needed for gradient computation has been modified by an
                # inplace operation.
                rpn_convs.append(
                    ConvModule(
                        in_channels,
                        self.feat_channels,
                        3,
                        padding=1,
                        inplace=False))
            self.rpn_conv = nn.Sequential(*rpn_convs)
        else:
            self.rpn_conv = nn.Conv2d(
                self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls1 = nn.Conv2d(self.feat_channels,
                                  self.num_base_priors * self.cls_out_channels, 1)
        self.rpn_cls2 = nn.Conv2d(self.feat_channels,
                                  self.num_base_priors * self.cls_out_channels, 1)
        
        reg_dim = self.bbox_coder.encode_size
        self.rpn_reg1 = nn.Conv2d(self.feat_channels,
                                 self.num_base_priors * reg_dim, 1)
        self.rpn_reg2 = nn.Conv2d(self.feat_channels,
                                 self.num_base_priors * reg_dim, 1)

    def forward_single(self, x: Tensor):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level \
                    the channels number is num_base_priors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale \
                    level, the channels number is num_base_priors * 4.
        """
        x = self.rpn_conv(x)
        x = F.relu(x)
        rpn_base_score = self.rpn_cls1(x)
        rpn_move_score = self.rpn_cls2(x)
        base_bbox_pred = self.rpn_reg1(x)
        move_bbox_pred = self.rpn_reg2(x)
        return torch.stack([rpn_base_score, rpn_move_score]), \
            torch.stack([base_bbox_pred, move_bbox_pred])

    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            anchors: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            bbox_weights: Tensor, avg_factor: int) -> tuple:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            avg_factor (int): Average factor that is used to average the loss.

        Returns:
            tuple: loss components.
        """
        bz = cls_score.size(1)
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        # test_cls_score = torch.zeros_like(cls_score)
        # for i in range(bz):
        #     test_cls_score[i, ...] = cls_score[i*2, ...]
        #     test_cls_score[i+8, ...] = cls_score[i*2+1, ...]
        # test_cls_score = test_cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        cls_score = cls_score.permute(0, 1, 3, 4,
                                      2).reshape(-1, self.cls_out_channels)
        # reshape(-1, 2*bz, self.cls_out_channels).permute(0, 2, 1).reshape(-1, self.cls_out_channels, 2, bz).permute(0, 3, 2, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=avg_factor)
        # regression loss
        target_dim = bbox_targets.size(-1)
        bbox_targets = bbox_targets.reshape(-1, target_dim)
        bbox_weights = bbox_weights.reshape(-1, target_dim)
        bbox_pred = bbox_pred.permute(0, 1, 3, 4,
                                      2).reshape(-1, self.bbox_coder.encode_size)
        # bbox_pred = bbox_pred.permute(0, 2, 3,
        #                               1).reshape(-1, 2*bz, self.bbox_coder.encode_size).permute(0, 2, 1).reshape(-1, self.bbox_coder.encode_size, 2, bz).permute(0, 3, 2, 1).reshape(-1, self.bbox_coder.encode_size)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, anchors.size(-1))
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
            bbox_pred = get_box_tensor(bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred, bbox_targets, bbox_weights, avg_factor=avg_factor)
        return loss_cls, loss_bbox


    def predict_by_feat(self, cls_scores,
                        bbox_preds, score_factors=None,
                        batch_img_metas = None,
                        cfg = None,
                        rescale = False,
                        with_nms = True):
        cls_scores_flat = []
        bbox_preds_flat = []
        for cls_levl_fea, bbox_level_fea in zip(cls_scores, bbox_preds):
            _, bz, ak, h, w = cls_levl_fea.shape
            cls_scores_flat.append(cls_levl_fea.reshape(2*bz, -1, h, w))
            bbox_preds_flat.append(bbox_level_fea.reshape(2*bz, -1, h, w))
        
        return super().predict_by_feat(
            cls_scores_flat,
            bbox_preds_flat,
            score_factors,
            batch_img_metas,
            cfg,
            rescale,
            with_nms
        )

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg,
                                rescale: bool = False,
                                with_nms: bool = True):
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Be compatible with
                BaseDenseHead. Not used in RPNHead.
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (ConfigDict, optional): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg['nms_pre'] = cfg.get('nms_pre', -1) // 2

        return super(MutipRPNHead, self)._predict_by_feat_single(
            cls_score_list,
            bbox_pred_list,
            score_factor_list,
            mlvl_priors,
            img_meta=img_meta,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms
        )
