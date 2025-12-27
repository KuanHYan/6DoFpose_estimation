import json
from typing import List, Union

import cv2
import numpy as np
from mmdet.datasets.coco import CocoDataset
from mmdet.registry import DATASETS
import os.path as osp


@DATASETS.register_module()
class CustomDataset(CocoDataset):
    """Custom dataset for detection.

    This dataset class is modified from coco dataset.
    """
    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        data_list = super().load_data_list()
        data_list = self._exclude_low_quality_mask(data_list)
        img_to_idx = {data['img_path']: i for i, data in enumerate(data_list)}
        wrong_list = []
        self.pair_idx = {}
        for img, id in img_to_idx.items():
            if '_base' in img:
                tar_img = img.replace('_base', '_move')
            elif '_move' in img:
                tar_img = img.replace('_move', '_base')
            else:
                raise ValueError(f'cannot find pair img for {img}')
            if tar_img in img_to_idx:
                self.pair_idx[id] = img_to_idx[tar_img]
            else:
                wrong_list.append(id)
                continue
        wrong_list.sort(reverse=True)
        for id in wrong_list:
            data_list.pop(id)  
        return data_list

    def _exclude_low_quality_mask(self, data_list):
        wrong_list = []
        for id, data in enumerate(data_list):
            img =  data['img_path']
            mask_name1 = img.replace('_color', '_mask')
            mask = cv2.imread(mask_name1)
            if mask is None:
                wrong_list.append(id)
                continue
            mask = mask[:, :, 2]
            mask = np.array(mask, dtype=np.int32)
            all_inst_ids1 = sorted(list(np.unique(mask)))
            if len(all_inst_ids1) < 2:
                wrong_list.append(id)
        wrong_list.sort(reverse=True)
        for id in wrong_list:
            data_list.pop(id)
        return data_list


    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        data_info = super().parse_data_info(raw_data_info)
        # add target_img_path
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']
        if type(data_info) is dict:
            data_info['target_img_path'] = osp.join(self.data_prefix['img'], img_info['target_file_name'])
            data_info['rotation'] = ann_info[0]['rotation']
            data_info['translation'] = ann_info[0]['translation']
            data_info['center'] = ann_info[0]['center']
            # data_info['intrinsic_matrix'] = ann_info[0]['intrinsic_matrix']
        else:
            for i in range(len(data_info)):
                data_info[i]['target_img_path'] = osp.join(self.data_prefix['img'], img_info[i]['target_file_name'])
                # add raw pose data, including 3d rotations, 3d translations, (3d scales)
                data_info[i]['rotation'] = ann_info[0]['rotation']
                data_info[i]['translation'] = ann_info[0]['translation']
                data_info[i]['center'] = ann_info[0]['center']
        return data_info

    def __getitem__(self, idx: int):  # TODO: -> List(dict) This is return type of base class method
        data = super().__getitem__(idx)
        target_id = self.pair_idx[idx]
        target_data = super().__getitem__(target_id)
        return {'base': data, 'move': target_data}
