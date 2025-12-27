import copy
import warnings
import torch
from mmdet.structures import SampleList
from torch import Tensor
from mmdet.registry import MODELS
from mmdet.models.detectors import RPN, SingleStageDetector
from mmdet.models.dense_heads import RPNHead, AnchorHead
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.utils.typing_utils import MultiConfig
from mmcv.cnn import ConvModule
import torch.nn as nn
import torch.nn.functional as F
from mmdet.structures.bbox import get_box_tensor
from typing import List


@MODELS.register_module()
class CustomRPN(RPN):
    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 rpn_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 **kwargs) -> None:
        super(SingleStageDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck) if neck is not None else None
        rpn_train_cfg = train_cfg['rpn'] if train_cfg is not None else None
        rpn_head_num_classes = rpn_head.get('num_classes', 1)
        # if rpn_head_num_classes != 1:
        #     warnings.warn('The `num_classes` should be 1 in RPN, but get '
        #                   f'{rpn_head_num_classes}, please set '
        #                   'rpn_head.num_classes = 1 in your config file.')
        rpn_head.update(train_cfg=rpn_train_cfg)
        rpn_head.update(test_cfg=test_cfg['rpn'])
        self.bbox_head = MODELS.build(rpn_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, batch_inputs: Tensor):
        batch_size = batch_inputs.shape[0] // 2
        base, move = batch_inputs[0:batch_size, ...], batch_inputs[batch_size:, ...]
        x1 = self.backbone(base)
        x2 = self.backbone(move)
        x = []
        for level_fea1, level_fea2 in zip(x1, x2):
            x.append(torch.cat([level_fea1, level_fea2], dim=1))
        if self.with_neck:
            x = self.neck(x)
        return x

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        batch_size = batch_inputs.shape[0] // 2
        assert batch_size > 0
        x = self.extract_feat(batch_inputs)
        # set cat_id of gt_labels to 0 in RPN
        rpn_data_samples = copy.deepcopy(batch_data_samples)
        for i, data_sample in enumerate(rpn_data_samples):
            data_sample.gt_instances.labels = \
                torch.zeros_like(data_sample.gt_instances.labels)  # + i // batch_size

        losses = self.bbox_head.loss(x, rpn_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        batch_size = batch_inputs.shape[0] // 2

        x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
