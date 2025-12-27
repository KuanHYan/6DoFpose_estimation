from typing import Tuple, List, Union
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from mmdet.registry import MODELS


@MODELS.register_module()
class RotationHead(nn.Module):
    def __init__(self, angle_or_se3: str = 'se3', in_channels: int = 256,
                 output_dim: Union[int, Tuple[int]] = 6,
                 temperature = 0.1, anneal_steps = -1, **kwargs):
        super().__init__(**kwargs)
        self.angle_or_se3 = angle_or_se3
        self.act = nn.ReLU()
        self.temperature = AnnealedGumbelSoftmax(
            start_temp=1.0, end_temp=temperature,
            anneal_steps=anneal_steps,
        )
        if angle_or_se3 == 'angle':
            self.head1 = JointAngleClassificationHead(in_channels, *output_dim)
            self.head2 = JointAngleClassificationHead(in_channels, *output_dim)
        elif angle_or_se3 == 'se3':
            self.head1 = nn.Linear(in_channels, 6)
            # self.head2 = nn.Linear(in_channels, 3)
        else:
            raise ValueError(f'{angle_or_se3} is not supported.')

    def _uv2se3(self, u: Tensor, v: Tensor):
        rot_u = F.normalize(u, dim=1)
        rot_w = torch.cross(rot_u, v, dim=-1)
        rot_w = F.normalize(rot_w, dim=1)
        rot_v = torch.cross(rot_w, rot_u, dim=-1)
        rot_v = F.normalize(rot_v, dim=-1)
        return torch.stack([rot_u, rot_v, rot_w], dim=1)

    def _angles_to_vector(self, theta, phi):
        """将球坐标角度转换为单位向量"""
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        return torch.stack([x, y, z], dim=1)

    def forward(self, x: Tensor) -> Tuple:
        tmp = self.temperature.get_temperature()
        if self.angle_or_se3 == 'angle':
            # 预测u的角度
            theta_u, phi_u, logits_u = self.head1(x, tmp)
            # 预测r的角度
            theta_r, phi_r, logits_v = self.head2(x, tmp)
            # 从角度恢复单位向量
            rot_u = self._angles_to_vector(theta_u, phi_u)
            rot_v = self._angles_to_vector(theta_r, phi_r)
            logits = torch.cat((logits_u.unsqueeze(1), logits_v.unsqueeze(1)), dim=1)
        else:
            out = self.head1(x)
            rot_u, rot_v = out[:, 0:3], out[:, 3:]
            logits = None
        rot_M = self._uv2se3(rot_u, rot_v)
        return rot_M.reshape(x.shape[0], -1), logits

    def predict(self, x: Tensor) -> Tensor:
        if self.angle_or_se3 == 'angle':
            # 预测u的角度
            theta_u, phi_u, _ = self.head1(x)
            # 预测r的角度
            theta_r, phi_r, _ = self.head2(x)
            # 从角度恢复单位向量
            rot_u = self._angles_to_vector(theta_u, phi_u)
            rot_v = self._angles_to_vector(theta_r, phi_r)
        else:
            out = self.head1(x)
            rot_u, rot_v = out[:, 0:3], out[:, 3:]
        rot_M = self._uv2se3(rot_u, rot_v)
        return rot_M.reshape(x.shape[0], -1)


class AngleClassificationHead(nn.Module):
    def __init__(self, in_channels, n_theta, n_phi):
        super().__init__()
        self.n_theta = n_theta
        self.n_phi = n_phi

        self.classifier = nn.Linear(in_channels, n_theta*n_phi)        
        # 角度bin值
        self.register_buffer('theta_bins', torch.linspace(0, torch.pi, n_theta))
        self.register_buffer('phi_bins', torch.linspace(0, 2 * torch.pi, n_phi))

    def forward(self, x, tmp=0.01):
        # 分类logits
        logits = self.classifier(x)  # [B, N_theta*N_phi]
        B, _ = logits.shape
        logits = logits.view(B, self.n_theta, self.n_phi)
        
        # 使用Gumbel-Softmax得到可微的argmax
        if self.training:
            gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
            probs = F.softmax((logits + gumbel) / tmp, dim=1)
        else:
            # 测试时用argmax
            _, max_indices = torch.max(logits.flatten(1, 2), dim=1)
            probs = torch.zeros_like(logits.flatten(1, 2))
            probs.scatter_(1, max_indices.unsqueeze(1), 1.0)
            probs = probs.view(B, self.n_theta, self.n_phi)
        
        # 计算期望角度
        theta = (probs * self.theta_bins.view(1, -1, 1)).sum(dim=1)  # [B, N_phi, 1]
        phi = (probs * self.phi_bins.view(1, 1, -1)).sum(dim=2)      # [B, N_theta, 1]
                
        # 最终角度（取主要方向）
        theta = theta.mean(dim=1)  # [B]
        phi = phi.mean(dim=1)      # [B]
        
        return theta, phi, logits


class JointAngleClassificationHead(nn.Module):
    def __init__(self, in_channels=256, n_theta=180, n_phi=360):
        super().__init__()
        self.n_theta = n_theta
        self.n_phi = n_phi

        # 直接输出n_theta*n_phi个类别
        self.classifier = nn.Linear(in_channels, n_theta * n_phi)

        # 创建角度bin的网格
        theta_bins = torch.linspace(0, torch.pi, n_theta)
        phi_bins = torch.linspace(0, 2 * torch.pi, n_phi)
        theta_grid, phi_grid = torch.meshgrid(theta_bins, phi_bins, indexing='ij')
        
        self.register_buffer('theta_grid', theta_grid)  # [n_theta, n_phi]
        self.register_buffer('phi_grid', phi_grid)      # [n_theta, n_phi]
    
    def forward(self, x, tmp=0.01):
        # 获得所有bin的logits
        logits = self.classifier(x)  # [B, n_theta*n_phi]
        # 联合softmax - 所有bin的概率和为1
        if self.training:
            gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
            probs_flat = F.softmax((logits + gumbel) / tmp, dim=1)
        else:
            probs_flat = F.softmax(logits / tmp, dim=1)
        
        # 重塑为2D
        joint_probs = probs_flat.view(-1, self.n_theta, self.n_phi)
        
        # 基于联合分布计算期望角度
        # 扩展网格以匹配batch维度
        theta_grid_exp = self.theta_grid.unsqueeze(0)  # [1, n_theta, n_phi]
        phi_grid_exp = self.phi_grid.unsqueeze(0)      # [1, n_theta, n_phi]
        
        # 期望计算
        theta_exp = (joint_probs * theta_grid_exp).sum(dim=(1, 2))
        phi_exp = (joint_probs * phi_grid_exp).sum(dim=(1, 2))
        
        return theta_exp, phi_exp, logits.view(-1, self.n_theta, self.n_phi)
    

class AnnealedGumbelSoftmax:
    def __init__(self, start_temp=1.0, end_temp=0.1, anneal_steps=1000):
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.anneal_steps = anneal_steps
        self.step = 0

    def get_temperature(self):
        # 线性退火
        if self.step >= self.anneal_steps:
            return self.end_temp

        frac = self.step / self.anneal_steps
        return self.start_temp - frac * (self.start_temp - self.end_temp)

    def update(self):
        self.step += 1
