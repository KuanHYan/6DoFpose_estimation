from typing import Tuple, List, Union
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from mmdet.registry import MODELS


@MODELS.register_module()
class TranslationHead(nn.Module):
    def __init__(self, in_channels, out_channel=3, scale=1, **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(out_channel, int):
            self.translation_head = nn.Linear(in_channels, 3)
        elif isinstance(out_channel, tuple) or isinstance(out_channel, tuple):
            assert len(out_channel) == 2
            self.translation_head = nn.Linear(in_channels, out_channel[0]*out_channel[1])
        else:
            raise TypeError(f'wrong param: {out_channel}')
        self.out_channel = out_channel
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        y = self.translation_head(x)
        if not isinstance(self.out_channel, int):
            y = y.reshape(-1, self.out_channel[0], self.out_channel[1])
        if self.scale != 1:
            y[0:2] *= self.scale
        return y
