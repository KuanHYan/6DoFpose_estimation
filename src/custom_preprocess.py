import torch
from mmengine.utils import is_seq_of
from mmdet.registry import MODELS
from mmdet.models.data_preprocessors import DetDataPreprocessor
import numpy as np

@MODELS.register_module()
class MergeImgDataPreprocessor(DetDataPreprocessor):
    """Data pre-processor for pose RoI
    """
    def __init__(self,
                 mean: tuple = (123.675, 116.28, 103.53),
                 std: tuple = (58.395, 57.12, 57.375),
                 bgr_to_rgb: bool = True,
                 pad_size_divisor: int = 32,
                 **kwargs) -> None:
        super().__init__(
            mean=mean,
            std=std,
            bgr_to_rgb=bgr_to_rgb,
            pad_size_divisor=pad_size_divisor,
            **kwargs)
        self.match_pairs_number = kwargs.get('match_pairs_number', 1)

    def forward(self, data: dict, training: bool = False) -> dict:
        """The forward function to process data.

        Args:
            data (dict): The input data.
            training (bool): Whether in training mode. Defaults to False.

        Returns:
            dict: The processed data.
        """
        data = super().forward(data, training=training)
        # return dict: {'inputs': inputs, 'data_samples': data_samples}
        # TODO: get gt_boxes and labels and then crop proposals
        # TODO: proposals is designed to N pairs of (image1, image2)
        inputs, data_samples = data['inputs'], data['data_samples']
        gt_labels = data_samples['gt_labels']
        proposals = self._crop_proposals_from_images(data_samples, inputs)
        assert len(proposals) > 1
        pairs = []
        pair_labels = []
        match_pairs_n = min(self.match_pairs_number, len(proposals))
        for _ in range(match_pairs_n):
            pi = np.random.choice(len(proposals)-1)
            pj = np.random.choice(len(proposals)-1)
            while pi == pj:
                pj = np.random.choice(len(proposals)-1)
            pairs.append((proposals[pi], proposals[pj]))
            pair_labels.append((gt_labels[pi], gt_labels[pj]))
        data['inputs'] = torch.stack(pairs).to(inputs.device)
        data['data_samples']['gt_labels'] = 'test'
        return data

    def _crop_proposals_from_images(self, data_samples, inputs):
        '''
        从原始图像中裁剪出 proposal '''
        # keys: gt_boxes, gt_labels
        gt_boxes = data_samples['gt_boxes']
        rgbd_imgs = inputs
        crop_imgs = []
        for i in range(len(gt_boxes)):
            rmin, rmax, cmin, cmax = gt_boxes[i]
            crop_imgs.append(rgbd_imgs[rmin:rmax, cmin:cmax])

        return crop_imgs


@MODELS.register_module()
class MergeFeatureDataPreprocessor(DetDataPreprocessor):
    """Data pre-processor for pose RoI
    """
    def __init__(self,
                 mean: tuple = (123.675, 116.28, 103.53, 0.0),
                 std: tuple = (58.395, 57.12, 57.375, 1.0),
                 bgr_to_rgb: bool = True,
                 pad_size_divisor: int = 32,
                 **kwargs) -> None:
        super().__init__(
            bgr_to_rgb=bgr_to_rgb,
            pad_size_divisor=pad_size_divisor,
            **kwargs)
        self.match_pairs_number = kwargs.get('match_pairs_number', 1)
        # mean=torch.tensor([123.675, 116.28, 103.53]).to(self.device)
        # std=torch.tensor([58.395, 57.12, 57.375]).to(self.device)
        self.register_buffer('mean',
                                torch.tensor(mean).view(-1, 1, 1), False)
        self.register_buffer('std',
                                torch.tensor(std).view(-1, 1, 1), False)

        self._enable_normalize = True
        self._channel_conversion = False

    def custom_normalize(self, data: dict) -> dict:
        batch_inputs = data['inputs']
        if is_seq_of(batch_inputs, torch.Tensor):
            inputs = [(_batch_input - self.mean) / self.std for _batch_input in batch_inputs]
            data['inputs'] = torch.stack(inputs)
        elif isinstance(batch_inputs, torch.Tensor):
            data['inputs'] = (batch_inputs - self.mean) / self.std
        else:
            raise TypeError('batch_inputs should be a tensor or a list of tensors')
        return data

    def custom_bgr_to_rgb(self, data: dict) -> dict:
        batch_inputs = data['inputs']
        if is_seq_of(batch_inputs, torch.Tensor):
            inputs = [_batch_input[[2, 1, 0, 3], ...] for _batch_input in batch_inputs]
            data['inputs'] = inputs
        elif isinstance(batch_inputs, torch.Tensor):
            data['inputs'] = batch_inputs[:, [2, 1, 0, 3], ...]
        else:
            raise TypeError('batch_inputs should be a tensor or a list of tensors')
        return data

    def pose2tensor(self, data: dict):
        base = data['base']['data_samples']
        move = data['move']['data_samples']
        for bs_, mv_ in zip(base, move):
            base_rot = torch.tensor(bs_.metainfo['rotation'], dtype=torch.float32)
            base_trans = torch.tensor(bs_.metainfo['translation'], dtype=torch.float32)
            move_rot = torch.tensor(mv_.metainfo['rotation'], dtype=torch.float32)
            move_trans = torch.tensor(mv_.metainfo['translation'], dtype=torch.float32)
            # inv_base = torch.transpose(base_rot, 0, 1)
            # relative_rot = torch.matmul(inv_base, move_rot)
            # relative_tran = torch.matmul(inv_base, (move_trans - base_trans).unsqueeze(-1)).squeeze(-1)
            # if len(relative_rot.shape) == 2:
            #     relative_rot = relative_rot.unsqueeze(0)
            # if len(relative_tran.shape) == 1:
            #     relative_tran = relative_tran.unsqueeze(0)
            bs_.gt_instances.translation = base_trans.unsqueeze(0)
            bs_.gt_instances.rotation = base_rot.unsqueeze(0)
            mv_.gt_instances.translation = move_trans.unsqueeze(0)
            mv_.gt_instances.rotation = move_rot.unsqueeze(0)

    def center_resize(self, data: dict):
        base = data['base']['data_samples']
        move = data['move']['data_samples']
        for dt in base + move:
            scale_factor = torch.tensor(dt.scale_factor, dtype=torch.float32)
            # 目前代码似乎没有pad
            center_raw = torch.tensor(dt.metainfo['center'], dtype=torch.float32)
            center_new = center_raw * scale_factor
            dt.gt_instances.center = center_new
            dt.gt_instances.intrinsic_matrix = torch.tensor(dt.metainfo['intrinsic_matrix'], dtype=torch.float32).unsqueeze(0)

    def pts2tesnsor(self, data: dict):
        base = data['base']['data_samples']
        move = data['move']['data_samples']
        for dt in base + move:
            pts = torch.tensor(dt.metainfo['point_cloud'], dtype=torch.float32)
            dt.gt_instances.point_cloud = pts.unsqueeze(0)

    def angle_labels2tensor(self, data: dict):
        base = data['base']['data_samples']
        move = data['move']['data_samples']
        for dt in base + move:
            pts = torch.tensor(np.stack(dt.metainfo['angle_labels']), dtype=torch.float32)
            dt.gt_instances.angle_labels = pts.unsqueeze(0)

    def forward(self, data: dict, training: bool = False) -> dict:
        """The forward function to process data.

        Args:
            data (dict): The input data.
            training (bool): Whether in training mode. Defaults to False.

        Returns:
            dict: The processed data.
        """
        assert 'base' in data and 'move' in data
        pairs = []
        pair_labels = []
        data_process = {}
        for key, inputs in data.items():
            data_ = self.custom_bgr_to_rgb(inputs)
            pairs += data_['inputs']
            for data_sample in data_['data_samples']:
                data_sample.set_metainfo({
                    'obj_flag': key,
                })
            pair_labels += data_['data_samples']
        self.pose2tensor(data)
        self.center_resize(data)
        self.pts2tesnsor(data)
        self.angle_labels2tensor(data)
        data_process['inputs'] = pairs
        data_process['data_samples'] = pair_labels
        data = super(MergeFeatureDataPreprocessor, self).forward(data_process, training=training)
        return data

# class AngleClassificationLabel():
#     def __init__(self, n_theta=180, n_phi=360):
#         super().__init__()
#         self.n_theta = n_theta
#         self.n_phi = n_phi
#         # 构建高斯核，用于标签平滑
#         self.gaussian_kernel = self._create_gaussian_kernel(sigma=1.0)
        
#     def _create_gaussian_kernel(self, sigma):
#         """创建二维高斯核，用于标签平滑"""
#         x = torch.arange(self.n_theta).float()
#         y = torch.arange(self.n_phi).float()
#         xx, yy = torch.meshgrid(x, y, indexing='ij')
#         kernel = torch.exp(-((xx - self.n_theta//2)**2 + (yy - self.n_phi//2)**2) / (2*sigma**2))
#         return kernel / kernel.sum()
    
#     def forward(self, gt_u, gt_r):
#         """
#         logits: [B, 2, N, M]
#         gt_u, gt_r: [B, 3] - 真实的单位向量
#         """
#         probs_u, probs_r = probs[:, 0], probs[:, 1]
        
#         # 将真实向量转换为bin索引
#         gt_theta_u, gt_phi_u = self._vector_to_bins(gt_u)  # [B]
#         gt_theta_r, gt_phi_r = self._vector_to_bins(gt_r)
        
#         # 转换为平滑标签
#         labels_u = self._create_smooth_labels(gt_theta_u, gt_phi_u)  # [B, N, M]
#         labels_r = self._create_smooth_labels(gt_theta_r, gt_phi_r)
        
#         # 分类损失
#         loss_u = F.binary_cross_entropy(probs_u, labels_u)
#         loss_r = F.binary_cross_entropy(probs_r, labels_r)
#         loss = loss_u + loss_r
#         # 几何一致性损失：确保u和r不平行
#         # 计算预测的u和r的点积，鼓励它们垂直
#         # pred_u = self._logits_to_vector(logits_u)  # [B, 3]
#         # pred_r = self._logits_to_vector(logits_r)  # [B, 3]
#         # dot_product = torch.abs(torch.sum(pred_u * pred_r, dim=1))
#         # ortho_loss = torch.mean(torch.clamp(dot_product - 0.3, min=0))  # 鼓励点积小于0.3
#         # loss += 0.1 * ortho_loss
#         return loss * self.loss_weight
    
#     def _vector_to_bins(self, vec):
#         """单位向量转换为bin索引"""
#         # vec: [B, 3]
#         theta = torch.acos(vec[:, 2].clamp(-1, 1))  # [0, π]
#         phi = torch.atan2(vec[:, 1], vec[:, 0])     # [-π, π]
#         phi = torch.where(phi < 0, phi + 2*torch.pi, phi)  # [0, 2π)
        
#         # 转换为bin索引
#         theta_idx = (theta / torch.pi * self.n_theta).long().clamp(0, self.n_theta-1)
#         phi_idx = (phi / (2*torch.pi) * self.n_phi).long().clamp(0, self.n_phi-1)
        
#         return theta_idx, phi_idx
    
#     def _create_smooth_labels(self, theta_idx, phi_idx):
#         """创建平滑的标签分布"""
#         B = theta_idx.shape[0]
#         labels = torch.zeros(B, self.n_theta, self.n_phi, device=theta_idx.device)
        
#         for b in range(B):
#             # 将高斯核中心放在目标位置
#             theta_center, phi_center = theta_idx[b].item(), phi_idx[b].item()
#             kernel = self.gaussian_kernel.clone()
            
#             # 计算核的偏移
#             theta_start = theta_center - self.n_theta//2
#             phi_start = phi_center - self.n_phi//2
            
#             # 将核复制到标签中（处理边界情况）
#             for i in range(self.n_theta):
#                 for j in range(self.n_phi):
#                     src_i = i - theta_start
#                     src_j = j - phi_start
#                     if 0 <= src_i < self.n_theta and 0 <= src_j < self.n_phi:
#                         labels[b, i, j] = kernel[src_i, src_j]
            
#             # 归一化
#             labels[b] = labels[b] / labels[b].sum()
        
#         return labels