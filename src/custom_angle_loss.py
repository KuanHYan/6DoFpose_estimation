import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS


@MODELS.register_module()
class AngleClassificationLoss(nn.Module):
    def __init__(self, n_theta=180, n_phi=360, loss_weight=1.0):
        super().__init__()
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.loss_weight = loss_weight
        # 构建高斯核，用于标签平滑
        self.gaussian_kernel = self._create_gaussian_kernel(sigma=1.0)
        
    def _create_gaussian_kernel(self, sigma):
        """创建二维高斯核，用于标签平滑"""
        x = torch.arange(self.n_theta).float()
        y = torch.arange(self.n_phi).float()
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        kernel = torch.exp(-((xx - self.n_theta//2)**2 + (yy - self.n_phi//2)**2) / (2*sigma**2))
        return kernel / kernel.sum()
    
    def forward(self, probs, gt_u, gt_r):
        """
        logits: [B, 2, N, M]
        gt_u, gt_r: [B, 3] - 真实的单位向量
        """
        probs_u, probs_r = probs[:, 0], probs[:, 1]
        
        # 将真实向量转换为bin索引
        gt_theta_u, gt_phi_u = self._vector_to_bins(gt_u)  # [B]
        gt_theta_r, gt_phi_r = self._vector_to_bins(gt_r)
        
        # 转换为平滑标签
        labels_u = self._create_smooth_labels(gt_theta_u, gt_phi_u)  # [B, N, M]
        labels_r = self._create_smooth_labels(gt_theta_r, gt_phi_r)
        
        # 分类损失
        loss_u = F.binary_cross_entropy(probs_u, labels_u)
        loss_r = F.binary_cross_entropy(probs_r, labels_r)
        loss = loss_u + loss_r
        # 几何一致性损失：确保u和r不平行
        # 计算预测的u和r的点积，鼓励它们垂直
        # pred_u = self._logits_to_vector(logits_u)  # [B, 3]
        # pred_r = self._logits_to_vector(logits_r)  # [B, 3]
        # dot_product = torch.abs(torch.sum(pred_u * pred_r, dim=1))
        # ortho_loss = torch.mean(torch.clamp(dot_product - 0.3, min=0))  # 鼓励点积小于0.3
        # loss += 0.1 * ortho_loss
        return loss * self.loss_weight
    
    def _vector_to_bins(self, vec):
        """单位向量转换为bin索引"""
        # vec: [B, 3]
        theta = torch.acos(vec[:, 2].clamp(-1, 1))  # [0, π]
        phi = torch.atan2(vec[:, 1], vec[:, 0])     # [-π, π]
        phi = torch.where(phi < 0, phi + 2*torch.pi, phi)  # [0, 2π)
        
        # 转换为bin索引
        theta_idx = (theta / torch.pi * self.n_theta).long().clamp(0, self.n_theta-1)
        phi_idx = (phi / (2*torch.pi) * self.n_phi).long().clamp(0, self.n_phi-1)
        
        return theta_idx, phi_idx
    
    def _create_smooth_labels(self, theta_idx, phi_idx):
        """创建平滑的标签分布"""
        B = theta_idx.shape[0]
        labels = torch.zeros(B, self.n_theta, self.n_phi, device=theta_idx.device)
        
        for b in range(B):
            # 将高斯核中心放在目标位置
            theta_center, phi_center = theta_idx[b].item(), phi_idx[b].item()
            kernel = self.gaussian_kernel.clone()
            
            # 计算核的偏移
            theta_start = theta_center - self.n_theta//2
            phi_start = phi_center - self.n_phi//2
            
            # 将核复制到标签中（处理边界情况）
            for i in range(self.n_theta):
                for j in range(self.n_phi):
                    src_i = i - theta_start
                    src_j = j - phi_start
                    if 0 <= src_i < self.n_theta and 0 <= src_j < self.n_phi:
                        labels[b, i, j] = kernel[src_i, src_j]
            
            # 归一化
            labels[b] = labels[b] / labels[b].sum()
        
        return labels