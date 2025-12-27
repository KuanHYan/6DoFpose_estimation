from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from mmengine.evaluator import BaseMetric
from mmdet.evaluation import CocoMetric
from mmengine.registry import METRICS
from mmdet.structures.mask import mask2bbox
import json
import os
import numpy as np
from .utils import PoseMetricsCalculator, get_point_cloud_from_depth

# 注册评估器
@METRICS.register_module()
class Pose6DEvaluator(CocoMetric):
    """OpenMMLab风格的6D位姿评估器"""
    def __init__(self,
                 ann_file: Optional[str],
                 custom_metric: Union[str, List[str]] = 'pose',
                 symmetry_ids: List[int] = None,
                 add_threshold: float = 0.1,  # 10% of diameter
                 add_s_threshold: float = 0.1,
                 rot_threshold: float = 5.0,
                 trans_threshold: float = 5.0,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwags):
        super(Pose6DEvaluator, self).__init__(
            ann_file,
            collect_device=collect_device,
            prefix=prefix,
            **kwags
        )
        self.custom_metric = custom_metric
        images_dict = self._coco_api.imgs
        self.model_points_dict = {}
        parent_path = os.path.dirname(ann_file)
        for id, img in images_dict.items():
            file_name = img['file_name']
            if 'point_cloud' in img:
                pt_value = self._load_points(os.path.join(parent_path, img['point_cloud']))
            else:
                full_name = os.path.join(parent_path, file_name.replace('color', 'depth'))
                pt_value = self._load_points(full_name)
            self.model_points_dict[id] = pt_value

        self.symmetry_ids = symmetry_ids or []
        self.add_threshold = add_threshold
        self.add_s_threshold = add_s_threshold
        self.rot_threshold = rot_threshold
        self.trans_threshold = trans_threshold
        
        # 初始化存储
        self.pose_errors = []
        self.add_scores = []
        self.add_s_scores = []
        self.n_d_n_cm_scores = []

    def _load_points(self, file: str):
        if file.endswith('ply') or file.endswith('npy'):
            pts = np.load(file)
        elif file.endswith('png'):
            assert 'depth' in file, f"cannot load pts from {file} image"
            meta_file = file.replace('depth.png', 'meta.json')
            assert os.path.exists(meta_file), 'If using depth images, camera intrinsic is necesary'
            with open(meta_file, 'r') as meta:
                data = json.load(meta)
                intrinsic = data['intrinsic_matrix']
                depth_scale = data['factor_depth']
                rot = data['rotation']
                tran = data['translation']
            
            depth = file
            mask = file.replace('depth', 'mask')
            pts = get_point_cloud_from_depth(
                depth, mask,
                np.array(intrinsic, dtype=np.float32),
                depth_scale,
                np.array(rot, dtype=np.float32),
                np.array(tran, dtype=np.float32)
            )
        else:
            raise ValueError('cannot load pts')
        return pts

    def process(self, data_batch: Any, data_samples: Sequence[dict]):
        """处理一批预测结果"""
        super(Pose6DEvaluator, self).process(data_batch, data_samples)
        if 'pose' not in self.custom_metric: return
        for data_sample in data_samples:
            img_id = data_sample['img_id']
            pred_instances = data_sample.get('pred_poses', {})
            if not len(pred_instances): continue
            # 提取预测和真实位姿
            pred_rots = pred_instances.get('rot_pred', []).cpu().numpy().reshape(-1, 3, 3)  # [N, 3, 3]
            pred_ts = pred_instances.get('tran_pred', []).cpu().numpy()  # [N, 3]
            # pred_labels = pred_instances.get('labels', [])
            if 'rotation' not in data_sample \
                or 'translation' not in data_sample:
                continue
            gt_rots = np.array(data_sample.get('rotation'), dtype=np.float32)
            gt_trans = np.array(data_sample.get('translation'), dtype=np.float32)            
            # TODO: 匹配预测和真实位姿（这里简化处理，实际需要根据IoU匹配）
            for p_rot, p_t in zip(pred_rots, pred_ts):
                self._compute_metrics(
                    img_id,
                    p_rot, p_t,
                    gt_rots, gt_trans)
    
    def _compute_metrics(self, obj_id: int,
                         R_pred: np.ndarray, t_pred: np.ndarray, 
                         R_gt: np.ndarray, t_gt: np.ndarray):
        """计算单个实例的指标"""
        
        # 获取模型点云
        model_points = self.model_points_dict.get(obj_id)
        if model_points is None:
            return
        
        # 创建计算器
        calculator = PoseMetricsCalculator(
            model_points=model_points,
            symmetry_objects=self.symmetry_ids
        )
        
        # 计算ADD/ADD-S
        is_symmetric = obj_id in self.symmetry_ids
        if is_symmetric:
            add_error = calculator.compute_add_symmetric(
                (R_pred @ model_points.T).T + t_pred,
                (R_gt @ model_points.T).T + t_gt
            )
        else:
            add_error = calculator.compute_add(R_pred, t_pred, R_gt, t_gt, obj_id)

        # 计算ADD精度
        diameter = self._compute_diameter(model_points)
        add_thresh = diameter * self.add_threshold
        
        add_acc = float(add_error < add_thresh)
        
        # 计算n°ncm精度
        n_d_n_cm = calculator.compute_n_d_n_cm(
            R_pred, t_pred, R_gt, t_gt,
            rot_threshold=self.rot_threshold,
            trans_threshold=self.trans_threshold
        )
        
        # 存储结果
        self.results.append({
            'obj_id': obj_id,
            'add_error': add_error,
            'add_acc': add_acc,
            'n_d_n_cm': float(n_d_n_cm),
            'rot_error': calculator.compute_rotation_error(R_pred, R_gt),
            'trans_error': calculator.compute_translation_error(t_pred, t_gt)
        })
        
        self.add_scores.append(add_acc)
        self.add_s_scores.append(add_acc if not is_symmetric else 0)
        self.n_d_n_cm_scores.append(float(n_d_n_cm))
    
    @staticmethod
    def _compute_diameter(points: np.ndarray) -> float:
        """计算模型点云直径"""
        min_vals = points.min(axis=0)
        max_vals = points.max(axis=0)
        diameter = np.linalg.norm(max_vals - min_vals).item()
        return diameter * 100  # 转换为厘米
    
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """计算所有指标"""
        if not results:
            return {}
        pose_res = []
        det_res = []
        for item in results:
            if isinstance(item, Dict):
                pose_res.append(item)
            elif isinstance(item, Tuple):
                det_res.append(item)
            else:
                raise TypeError('The type of one result item is wrong')
        # parent class metrics
        det_metrics = super(Pose6DEvaluator, self).compute_metrics(det_res)
        if 'pose' not in self.custom_metric: return det_metrics
        # # 按物体ID分组
        # metrics_by_obj = {}
        # for res in pose_res:
        #     obj_id = res['obj_id']
        #     if obj_id not in metrics_by_obj:
        #         metrics_by_obj[obj_id] = {
        #             'add_errors': [],
        #             'add_accs': [],
        #             'n_d_n_cms': []
        #         }
        #     metrics_by_obj[obj_id]['add_errors'].append(res['add_error'])
        #     metrics_by_obj[obj_id]['add_accs'].append(res['add_acc'])
        #     metrics_by_obj[obj_id]['n_d_n_cms'].append(res['n_d_n_cm'])
        
        # 计算总体指标. Here, the 100 is for converting to percentage
        total_metrics = {
            'ADD': np.mean(self.add_scores).item() * 100 if self.add_scores else 0.,
            'ADD-S': np.mean(self.add_s_scores).item() * 100 if self.add_s_scores else 0.,
            f'{self.rot_threshold}deg_{self.trans_threshold}cm': np.mean(self.n_d_n_cm_scores).item() * 100 if self.n_d_n_cm_scores else 0.,
            'mean_ADD_error': np.mean([r['add_error'] for r in pose_res]).item() if pose_res else 0.,
            'mean_rotation_error': np.mean([r['rot_error'] for r in pose_res]).item() if pose_res else 0.,
            'mean_translation_error': np.mean([r['trans_error'] for r in pose_res]).item() if pose_res else 0.,
        }
        
        # # 计算每个物体的指标
        # for obj_id, obj_metrics in metrics_by_obj.items():
        #     prefix = f'obj_{obj_id}_'
        #     total_metrics[prefix + 'ADD'] = np.mean(obj_metrics['add_accs']).item() * 100
        #     total_metrics[prefix + 'ADD_error'] = np.mean(obj_metrics['add_errors']).item()
        #     total_metrics[prefix + 'n_d_n_cm'] = np.mean(obj_metrics['n_d_n_cms']).item() * 100
        total_metrics.update(det_metrics)
        return total_metrics
