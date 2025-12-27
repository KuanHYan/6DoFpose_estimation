import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from typing import List, Optional


def load_depth(img_path, norm_scale):
    """ Load depth image from img_path. """
    depth = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        # NOTE: RGB is actually BGR in opencv
        depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
        depth16 = np.where(depth16==32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    return depth16 / norm_scale

def get_point_cloud_from_depth(
        depth_file, mask_file,
        intrinsic, norm_scale,
        rotate, translate):
    depth = load_depth(depth_file, norm_scale)
    mask = cv2.imread(mask_file)
    assert mask is not None, '[ Error ]: Failed to load mask image.'
    mask = mask[:, :, 2]
    mask = np.array(mask, dtype=np.int32)
    all_inst_ids = sorted(list(np.unique(mask)))
    if len(all_inst_ids) < 2:
        return None
    # 0 is background and 1 denotes instance.
    pts, _ = backproject(depth, intrinsic, mask==all_inst_ids[1])
    # transform to zero-centered coordinate frame
    pts = (rotate.squeeze(0).T @ (pts - translate).T).T
    return pts.astype(np.float32)

def backproject(depth, intrinsics, instance_mask):
    """ Back-projection, use opencv camera coordinate frame.
    """
    cam_fx = intrinsics[0, 0]
    cam_fy = intrinsics[1, 1]
    cam_cx = intrinsics[0, 2]
    cam_cy = intrinsics[1, 2]

    non_zero_mask = (depth > 0)
    final_instance_mask = np.logical_and(instance_mask, non_zero_mask)
    idxs = np.where(final_instance_mask)

    z = depth[idxs[0], idxs[1]]
    x = (idxs[1] - cam_cx) * z / cam_fx
    y = (idxs[0] - cam_cy) * z / cam_fy
    pts = np.stack((x, y, z), axis=1)

    return pts, idxs


class PoseMetricsCalculator:
    """6D位姿估计指标计算器"""
    
    def __init__(self, model_points: np.ndarray, symmetry_objects: List[int] = None):
        """
        初始化
        
        Args:
            model_points: 3D模型点云 [N, 3]
            symmetry_objects: 对称物体的ID列表
        """
        self.model_points = model_points
        self.symmetry_objects = symmetry_objects or []
        self.num_points = model_points.shape[0]
        # self.pts_in_camera_view = pts_in_camera_view

    @staticmethod
    def compute_rotation_error(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
        """计算旋转误差（角度）"""
        assert R_pred.shape == (3, 3) and R_gt.shape == (3, 3)
        
        # 计算旋转矩阵之间的差异
        error_cos = (np.trace(R_pred @ R_gt.T) - 1) / 2
        error_cos = np.clip(error_cos, -1.0, 1.0)
        error_deg = np.arccos(error_cos) * 180.0 / np.pi
        return error_deg
    
    @staticmethod
    def compute_translation_error(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
        """计算平移误差（厘米）"""
        assert t_pred.shape == (3,) and t_gt.shape == (3,)
        error_cm = np.linalg.norm(t_pred - t_gt) * 100  # 假设输入单位为米
        return error_cm

    def compute_add(
        self, 
        R_pred: np.ndarray, 
        t_pred: np.ndarray, 
        R_gt: np.ndarray, 
        t_gt: np.ndarray,
        obj_id: Optional[int] = None
    ) -> float:
        """
        计算ADD (Average Distance of Model Points)
        
        Args:
            R_pred, t_pred: 预测的旋转矩阵和平移向量
            R_gt, t_gt: 真实旋转矩阵和平移向量
            obj_id: 物体ID（用于判断是否对称）
        """
        # 变换模型点云
        # pts_pred = (R_pred @ self.model_points.T).T + t_pred
        # pts_gt = (R_gt @ self.model_points.T).T + t_gt
        # if self.pts_in_camera_view:
        #     pts_gt = self.model_points
        #     rel_R = R_pred @ R_gt.T
        #     pts_pred = (rel_R @ self.model_points.T).T + t_pred - rel_R @ t_gt
        # else:
        pts_pred = (R_pred @ self.model_points.T).T + t_pred
        pts_gt = (R_gt @ self.model_points.T).T + t_gt
    
        if obj_id in self.symmetry_objects:
            # 对称物体使用ADD-S
            return self.compute_add_symmetric(pts_pred, pts_gt)
        else:
            # 非对称物体使用标准ADD
            distances = np.linalg.norm(pts_pred - pts_gt, axis=1)
            return np.mean(distances) * 100  # 转换为厘米

    def compute_add_symmetric(self, pts_pred: np.ndarray, pts_gt: np.ndarray) -> float:
        """计算对称物体的ADD-S"""
        distances = []
        for pt_pred in pts_pred:
            # 找到最近的点
            min_dist = np.min(np.linalg.norm(pt_pred - pts_gt, axis=1))
            distances.append(min_dist)
        return np.mean(distances) * 100
    
    def compute_n_d_n_cm(
        self,
        R_pred: np.ndarray,
        t_pred: np.ndarray,
        R_gt: np.ndarray,
        t_gt: np.ndarray,
        rot_threshold: float = 5.0,  # 角度阈值
        trans_threshold: float = 5.0  # 距离阈值（厘米）
    ) -> bool:
        """
        检查是否满足 n° 和 n cm 条件
        """
        rot_error = self.compute_rotation_error(R_pred, R_gt)
        trans_error = self.compute_translation_error(t_pred, t_gt)
        
        return (rot_error < rot_threshold) and (trans_error < trans_threshold)
