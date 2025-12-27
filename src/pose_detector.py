import copy
from torch import Tensor
import torch
import torch.nn as nn
from mmdet.structures import SampleList
from mmdet.registry import MODELS
from mmdet.models.detectors import TwoStageDetector
from typing import List, Tuple


@MODELS.register_module()
class TwoStagePose(TwoStageDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """
    def __init__(self, freeze_modules=None, **kwargs):
        super().__init__(**kwargs)
        if freeze_modules is not None:
            for module_name in freeze_modules:
                self._freeze_module(module_name)    
    
    def _freeze_module(self, module_name):
        """冻结模块并确保其参与前向计算"""
        module = dict(self.named_modules())[module_name]
        # 设置eval模式
        module.eval()
        
        # 冻结所有参数
        for param in module.parameters():
            param.requires_grad = False
        
        # 冻结BN层
        for m in module.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                m.eval()
                m.track_running_stats = False
        print(f"初始化时冻结: {module_name}")


    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
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
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        assert batch_inputs.shape[0] // 2 > 0, 'Batch size must be greater than 1.'
        x = self.extract_feat(batch_inputs)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)

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
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """

        x = self.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=rescale)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, rpn_results_list)
        
        for data_sample, pred_pose in zip(batch_data_samples, results_list):
            data_sample.pred_poses = pred_pose
        # custom 
        
        return batch_data_samples
