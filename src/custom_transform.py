from typing import Optional, Tuple, Dict
import numpy as np
import torch
from mmdet.registry import TRANSFORMS
from mmcv.transforms import LoadImageFromFile, BaseTransform
from mmdet.datasets.transforms import Resize, RandomFlip
import cv2
import json
from .utils import get_point_cloud_from_depth


@TRANSFORMS.register_module()
class LoadDepthImg(LoadImageFromFile):
    """Load depth image.

    """
    def __init__(self, factor_depth, **kwargs) -> None:
        super().__init__(**kwargs)
        self.factor_depth = factor_depth

    def transform(self, results: Dict) -> Optional[Dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        rgb_img = results['img']
        rgb_ori_shape = rgb_img.shape
        depth_name = results['img_path'].replace('_color', '_depth')
        # if bgr_2_rgb depth should transpose(1, 0).
        # The default order is bgr, that means the channel index is 2.
        hw_to_wh_flag = rgb_ori_shape.index(3) == 0
        depth = cv2.imread(depth_name, cv2.IMREAD_ANYDEPTH)
        if len(depth.shape) == 3:
            # This is encoded depth image, let's convert
            # NOTE: RGB is actually BGR in opencv
            depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
            depth16 = np.where(depth16 >= 32001, 0, depth16)
        elif len(depth.shape) == 2 and depth.dtype == 'uint16':
            depth16 = depth
        else:
            assert False, '[ Error ]: Unsupported depth type.'
        if hw_to_wh_flag:
            depth16 = depth16.transpose(1, 0)
        assert rgb_ori_shape[:-1] == depth16.shape
        if len(depth.shape) == 2:
            depth16 = depth16[..., np.newaxis]
        depth16 = depth16.astype(np.float32) / self.factor_depth
        if isinstance(rgb_img, np.ndarray):
            if rgb_img.dtype != np.float32:
                rgb_img = rgb_img.astype(np.float32)
            results['img'] = np.concatenate([rgb_img, depth16], axis=-1)
        elif isinstance(rgb_img, torch.Tensor):
            depth16 = torch.from_numpy(depth16)
            if rgb_img.dtype != torch.float32:
                rgb_img = rgb_img.to(torch.float32)
            results['img'] = torch.cat([rgb_img, depth16], dim=-1)
        else:
            print('[ Error ]: rgb_img is not numpy or torch.Tensor')

        return results


@TRANSFORMS.register_module()
class LoadPointCloud(BaseTransform):
    """
    Load point cloud.
    """
    def __init__(self,
                 **kwargs) -> None:
        super().__init__(**kwargs)

    def transform(self, results: Dict) -> Optional[Dict]:
        # NOTE: this is RGB-D image, with 4 channels
        depth_name = results['img_path'].replace('_color', '_depth')
        mask_name = results['img_path'].replace('_color', '_mask')
        meta_file = results['img_path'].replace('_color.png', '_meta.json')
        with open(meta_file, 'r') as meta:
            data = json.load(meta)
            intrinsic = data['intrinsic_matrix']
            depth_scale = data['factor_depth']
            rot = data['rotation']
            tran = data['translation']
        pts = get_point_cloud_from_depth(
            depth_name,
            mask_name,
            np.array(intrinsic, dtype=np.float32),
            depth_scale,
            np.array(rot, dtype=np.float32),
            np.array(tran, dtype=np.float32)
        )
        results['point_cloud'] = pts
        results['intrinsic_matrix'] = intrinsic
        # results['rotation'] = 
        return results


@TRANSFORMS.register_module()
class CustomResize(Resize):
    """
    Inherit from mmdet.datasets.transforms.Resize
    What is different: resize for cropped objects images
    """

    def transform(self, results: Dict) -> Optional[Dict]:
        """
            Args:
                results (dict): Result dict from loading pipeline.
                It supports 
        """
        res = super(CustomResize, self).transform(results)
        # resize camera intrinsic matrix
        mat = res['intrinsic_matrix']
        scale_w, scale_h = res['scale_factor']
        mat[0][0] = mat[0][0] * scale_w
        mat[0][2] = mat[0][2] * scale_w
        mat[1][1] = mat[1][1] * scale_h
        mat[1][2] = mat[1][2] * scale_h
        return res


# @TRANSFORMS.register_module()
# class CenterProposal(BaseTransform):
#     """"
#     Crop proposals from images"""
#     def __init__(self,
#                  match_pairs_number: int = 1,
#                  scale_factor: Tuple[int, int] = (1, 1),
#                  ):
#         self.match_pairs_number = match_pairs_number
#         self.scale_factor = scale_factor

#     def transform(self, results: Dict) -> Optional[Dict]:
#         assert 'img' in results and 'gt_bboxes' in results
#         gt_boxes = results['gt_bboxes']
#         rgbd_imgs = results['img']
#         crop_imgs = []
#         for i in range(len(gt_boxes)):
#             rmin, rmax, cmin, cmax = gt_boxes[i]
#             crop_imgs.append(rgbd_imgs[rmin:rmax, cmin:cmax])
#         results['img'] = crop_imgs
#         return results
    

class CustomRandomFlip(RandomFlip):
    """
    The input is a list of rgbd images.
    So we need to flip them for each image.
    """
    def transform(self, results: Dict) -> Dict:
        img_list = results['img']
        results['img'] = [
            super()._flip_img(img) for img in img_list
        ]
        return super().transform(results)


@TRANSFORMS.register_module()
class AngleClassificationLabel(BaseTransform):
    def __init__(self, n_theta=180, n_phi=360):
        super().__init__()
        self.n_theta = n_theta
        self.n_phi = n_phi
        # 构建高斯核，用于标签平滑
        # self.gaussian_kernel = self._create_gaussian_kernel(sigma=1.0)
        
    def _create_gaussian_kernel(self, sigma):
        """创建二维高斯核，用于标签平滑"""
        x = np.arange(self.n_theta, dtype=np.float32)
        y = np.arange(self.n_phi, dtype=np.float32)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        kernel = np.exp(-((xx - self.n_theta//2)**2 + (yy - self.n_phi//2)**2) / (2*sigma**2))
        return kernel / kernel.sum()
    
    def transform(self, results: Dict) -> Optional[Dict]:
        """
        logits: [B, 2, N, M]
        gt_u, gt_r: [B, 3] - 真实的单位向量
        """
        rot_mats = results['rotation']
        gt_u = np.clip(np.array(rot_mats[0], dtype=np.float32), -1, 1)
        gt_r = np.clip(np.array(rot_mats[1], dtype=np.float32), -1, 1)
        # 将真实向量转换为bin索引
        gt_theta_u, gt_phi_u = self._vector_to_bins(gt_u)  # [B]
        gt_theta_r, gt_phi_r = self._vector_to_bins(gt_r)
        
        # 转换为平滑标签
        labels_u = self._create_smooth_labels(gt_theta_u, gt_phi_u)  # [B, N, M]
        labels_r = self._create_smooth_labels(gt_theta_r, gt_phi_r)
        results['angle_labels'] = (labels_u, labels_r)
        return results
    
    def _vector_to_bins(self, vec):
        """单位向量转换为bin索引"""
        # vec: [B, 3]
        theta = np.arccos(vec[2])  # [0, π]
        phi = np.arctan2(vec[1], vec[0])     # [-π, π]
        if phi < 0:
            phi = phi + 2*np.pi
        
        # 转换为bin索引
        theta_idx = np.clip(np.int64(theta / np.pi * self.n_theta), 0, self.n_theta-1)
        phi_idx = np.clip(np.int64(phi / (2*np.pi) * self.n_phi), 0, self.n_phi-1)
        
        return theta_idx, phi_idx
    
    def _create_smooth_labels(self, theta_idx, phi_idx):
        """创建平滑的标签分布"""
        labels = np.zeros((self.n_theta, self.n_phi), dtype=np.float32)
        
        # 将高斯核中心放在目标位置
        theta_center, phi_center = theta_idx, phi_idx
        kernel = self._create_gaussian_kernel(sigma=1.0)
        
        # 计算核的偏移
        theta_start = theta_center - self.n_theta//2
        phi_start = phi_center - self.n_phi//2
        
        # 将核复制到标签中（处理边界情况）
        for i in range(self.n_theta):
            for j in range(self.n_phi):
                src_i = i - theta_start
                src_j = j - phi_start
                if 0 <= src_i < self.n_theta and 0 <= src_j < self.n_phi:
                    labels[i, j] = kernel[src_i, src_j]
        
        # 归一化
        labels = labels / labels.sum()
        
        return labels