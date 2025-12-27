from typing import Tuple, List
from torch import Tensor
import torch
import torch.nn.functional as F
from mmdet.models import StandardRoIHead
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import get_box_tensor, bbox2roi
from mmdet.utils import ConfigType, InstanceList
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.utils import unpack_gt_instances
from mmengine.structures import InstanceData
from .utils import PoseMetricsCalculator


@MODELS.register_module()
class PoseHead(StandardRoIHead):
    def __init__(
        self,
        roi_extractor,
        attention_head,
        rotation_head,
        translation_head,
        loss_rot,
        loss_angle,
        loss_tran,
        train_cfg,
        test_cfg,
        roi_topk=32,
        init_cfg=None,
        rot_use_add_loss=False,
        tran_use_center=False,
        **kwargs
    ):
        super(PoseHead, self).__init__(
            None, None, None, None, None,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            
        )
        self.roi_extractor = MODELS.build(roi_extractor)
        self.attention_head = MODELS.build(attention_head)
        self.rotation_head = MODELS.build(rotation_head)
        self.translation_head = MODELS.build(translation_head)
        self.loss_rot = MODELS.build(loss_rot)
        self.loss_tran = MODELS.build(loss_tran)
        self.loss_angle = MODELS.build(loss_angle)
        self.rot_use_add_loss = rot_use_add_loss
        self.tran_use_center = tran_use_center
        self.symmetry_ids = kwargs.get('symmetry_ids', [])
        self.topk = roi_topk

    def _center_xy2real(self, tran_preds, bt_rois, rois_per_img, camera_params, scores):
        # x = tran_pred[:, 0]
        # y = tran_pred[:, 1]
        # z = tran_pred[:, 2]
        bt_rois = torch.split(bt_rois, rois_per_img)
        translation = []
        for tran, rois, s, camera_intri in zip(tran_preds, bt_rois, scores, camera_params):
            rois_center = (rois[:, 0:2] + rois[:, 2:4]) / 2
            rois_whs = rois[:, 2:4] - rois[:, 0:2]
            x, y ,z = tran
            s = F.softmax(s / 0.1, dim=0)
            # oxys = torch.stack(
            #     [
            #         ((x * rois_whs[:, 0] + rois_center[:, 0]) * s).sum(dim=1),
            #         ((y * rois_whs[:, 1] + rois_center[:, 1]) * s).sum(dim=1),
            #     ],
            #     dim=1
            # )
            ox = ((x * rois_whs[:, 0] + rois_center[:, 0]) * s).sum(dim=-1)
            oy = ((y * rois_whs[:, 1] + rois_center[:, 1]) * s).sum(dim=-1)
            translation += [
                torch.stack([
                    z * (ox - camera_intri[0, 0, 2]) / camera_intri[0, 0, 0], 
                    z * (oy - camera_intri[0, 1, 2]) / camera_intri[0, 1, 1], 
                    z
            ])]
        translation = torch.stack(translation)
        return translation

    def forward(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList = None) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            x (List[Tensor]): Multi-level features that may have different
                resolutions.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
            the meta information of each image and corresponding
            annotations.

        Returns
            tuple: A tuple of features from ``bbox_head`` and ``mask_head``
            forward.
        """
        results = ()
        proposals = [rpn_results.bboxes for rpn_results in rpn_results_list]
        rois = self._custom_bbox2roi(proposals)
        # bbox head
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            results = results + (bbox_results['cls_score'],
                                 bbox_results['bbox_pred'])
        # pose head
        rois_per_img = [len(res.bboxes) for res in rpn_results_list]
        roi_scores = [len(res.scores) for res in rpn_results_list]
        pose_results = self._pose_forward(x, rois, rois_per_img, roi_scores)
        results = results + (pose_results['rot_pred'], pose_results['tran_pred'])
        return results

    def _pose_forward(self, x: Tuple[Tensor], rois, rois_per_img, roi_scores = None, camera_intri = None) -> dict:
        """Pose head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

        Returns:
             dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
        """
        # TODO: a more flexible way to decide which feature maps to use
        roi_feats = self.roi_extractor(x[:self.roi_extractor.num_inputs], rois)
        fea_base, fea_move = self.attention_head(roi_feats, rois_per_img)
        rot_pred_b, prob_pred_b = self.rotation_head(fea_base)
        tran_pred_b = self.translation_head(fea_base)
        rot_pred_m, prob_pred_m = self.rotation_head(fea_move)
        tran_pred_m = self.translation_head(fea_move)
        rot_pred = torch.cat([rot_pred_b, rot_pred_m], dim=0)
        tran_pred = torch.cat([tran_pred_b, tran_pred_m], dim=0)
        if self.tran_use_center:
            assert roi_scores is not None
            tran_pred = self._center_xy2real(tran_pred, rois[:,1:], rois_per_img, camera_intri, roi_scores)

        results = dict(
            rot_pred=rot_pred,
            tran_pred=tran_pred,
            pose_feats=roi_feats
        )
        if prob_pred_b is not None and prob_pred_m is not None:
            prob_pred = torch.cat([prob_pred_b, prob_pred_m], dim=0)
            results['prob_pred'] = prob_pred
        return results

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: List[DetDataSample]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs
        batch_gt_rotations = []
        batch_gt_translations = []
        batch_gt_pts = []
        batch_gt_obj_ids = []
        batch_gt_labels = []
        batch_gt_cameras = []
        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        real_bz = num_imgs // 2
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')
            assign_result = self.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i%real_bz][None] for lvl_feat in x])
            rpn_sample_ids = torch.cat([sampling_result.pos_inds, sampling_result.neg_inds], dim=0)
            sampling_result.scores = torch.ones_like(rpn_sample_ids).type(torch.float32)
            si = 1 if sampling_result.pos_is_gt[0] else 0
            rpn_sample_ids = rpn_sample_ids[si:] - 1
            assert rpn_sample_ids.min() > -1 and rpn_sample_ids.max() < len(rpn_results['scores']), f'wrong id: {rpn_sample_ids}'
            sampling_result.scores[si:] = rpn_results['scores'][rpn_sample_ids]
            sampling_results.append(sampling_result)
            # custom batch gt_instances
        for i in range(num_imgs):
            batch_gt_rotations.append(batch_gt_instances[i].rotation)
            trans_using_center = batch_gt_instances[i].translation.clone()
            # if self.tran_use_center:
            #     trans_using_center[:, 0:2] = batch_gt_instances[i].center
            batch_gt_translations.append(trans_using_center)
            batch_gt_pts.append(batch_gt_instances[i].point_cloud.squeeze(0))
            batch_gt_obj_ids.append(batch_gt_instances[i].labels)
            batch_gt_labels.append(batch_gt_instances[i].angle_labels)
            batch_gt_cameras.append(batch_gt_instances[i].intrinsic_matrix)

        losses = dict()
        # bbox head loss
        if self.with_bbox:
            bbox_results = self.bbox_loss(x, sampling_results)
            losses.update(bbox_results['loss_bbox'])
        # pose head forward and loss
        pose_results = self._pose_loss(
            x, sampling_results,
            batch_gt_rotations, batch_gt_translations,
            batch_gt_obj_ids if self.rot_use_add_loss else None,
            batch_gt_pts if self.rot_use_add_loss else None,
            batch_gt_labels if self.rotation_head.angle_or_se3 == 'angle' else None,
            batch_gt_cameras if self.tran_use_center else None
        )
        losses['loss_rot'] = pose_results['loss_rot']
        losses['loss_tran'] = pose_results['loss_tran']
        return losses

    def _compute_add(self, pts_pred: List[Tensor], pts_gt: List[Tensor], obj_ids: List) -> Tensor:
        """计算对称物体的ADD-S"""
        distances = []
        for i, id in enumerate(obj_ids):
            id = id.item()
            if id in self.symmetry_ids:
                ch_dis = []
                for pt_pred in pts_pred:
                    # 找到最近的点
                    min_dist = (pt_pred[i] - pts_gt[i]).norm(dim=-1).min()
                    ch_dis.append(min_dist)
                distances.append(torch.cat(distances).mean())
            else:
                distances.append((pts_pred[i] - pts_gt[i]).norm(dim=-1).mean())
        return sum(distances) / len(distances)

    def _pose_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult],
                  gt_rotations: List[Tensor],
                  gt_translations: List[Tensor],
                  label_ids = None,
                  gt_points = None,
                  gt_angle_labels = None,
                  camera_params = None) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        rois = self._custom_bbox2roi([res.priors for res in sampling_results])
        rois_per_img = [len(res.priors) for res in sampling_results]
        roi_scores = [res.scores for res in sampling_results]
        pose_results = self._pose_forward(x, rois, rois_per_img, roi_scores, camera_params)
        gt_rotations = torch.cat(gt_rotations, dim=0)
        gt_translations = torch.cat(gt_translations, dim=0)
        if self.rot_use_add_loss:
            assert gt_points is not None
            # 计算ADD (Average Distance of Model Points)
            R_pred = pose_results['rot_pred'].reshape(-1, 3, 3)
            R_gt = gt_rotations # .reshape(-1, 3, 3)

            # t_pred = pose_results['tran_pred']
            # t_gt = gt_translations

            pts_pred = [(R_pred[i] @ gt_points[i].T).T for i in range(len(gt_points))]
            pts_gt = [(R_gt[i] @ gt_points[i].T).T for i in range(len(gt_points))]

            loss_rot = self._compute_add(pts_pred, pts_gt, label_ids)
            if self.rotation_head.angle_or_se3 == 'angle':
                gt_angle_labels = torch.cat(gt_angle_labels, dim=0)
                loss_rot += 100 * self.loss_angle(pose_results['prob_pred'], gt_angle_labels)
        else:
            gt_rotations = gt_rotations.reshape(len(rois_per_img), -1)
            loss_rot = self.loss_rot(pose_results['rot_pred'], gt_rotations)
        pose_results.update(loss_rot=loss_rot)
        loss_tran = self.loss_tran(pose_results['tran_pred'], gt_translations)
        pose_results.update(loss_tran=loss_tran)
        return pose_results

    def _custom_bbox2roi(self, bbox_list: List) -> List[Tensor]:
        """Convert a list of bboxes to roi format.

        Args:
            bbox_list (List[Union[Tensor, :obj:`BaseBoxes`]): a list of bboxes
                corresponding to a batch of images.

        Returns:
            Tensor: shape (n, box_dim + 1), where ``box_dim`` depends on the
            different box types. For example, If the box type in ``bbox_list``
            is HorizontalBoxes, the output shape is (n, 5). Each row of data
            indicates [batch_ind, x1, y1, x2, y2].
        """
        rois_list = []
        bz = len(bbox_list) // 2
        for img_id, bboxes in enumerate(bbox_list):
            bboxes = get_box_tensor(bboxes)
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id % bz)
            rois = torch.cat([img_inds, bboxes], dim=-1)
            rois_list.append(rois)
        rois = torch.cat(rois_list, 0)
        return rois

    def predict_pose(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                     rpn_results_list):
        proposals = []
        roi_scores = []
        rois_per_img = []
        for res in rpn_results_list:
            if len(res.bboxes) < self.topk:
                proposals.append(res.bboxes)
                roi_scores.append(res.scores)
                rois_per_img.append(len(res.bboxes))
            else:
                proposals.append(res.bboxes[:self.topk])
                roi_scores.append(res.scores[:self.topk])
                rois_per_img.append(self.topk)

        rois = self._custom_bbox2roi(proposals)
        camera_params = [torch.tensor([meta['intrinsic_matrix']]).to(x[0].device) for meta in batch_img_metas]
        pose_results = self._pose_forward(x, rois, rois_per_img, roi_scores, camera_params)
        rot_pred = pose_results['rot_pred']
        tran_pred = pose_results['tran_pred']
        pred = [
            InstanceData(
                rot_pred=rot_pred[i].unsqueeze(0),
                tran_pred=tran_pred[i].unsqueeze(0),
            ) for i in range(len(rot_pred))
        ]
        return pred

    def predict(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        """Perform forward propagation of the roi head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (N, C, H, W).
            rpn_results_list (list[:obj:`InstanceData`]): list of region
                proposals.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results to
                the original image. Defaults to True.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        # TODO: nms_op in mmcv need be enhanced, the bbox result may get
        #  difference when not rescale in bbox_head

        # If it has the mask branch, the bbox branch does not need
        # to be scaled to the original image scale, because the mask
        # branch will scale both bbox and mask at the same time.
        bbox_rescale = rescale if not self.with_mask else False
        results_list = self.predict_pose(
            x,
            batch_img_metas,
            rpn_results_list,
        )

        if self.with_bbox:
            results_list = self.predict_bbox(
                x, batch_img_metas, rpn_results_list, 
                rcnn_test_cfg=self.test_cfg,
                rescale=bbox_rescale
            )

        return results_list