from __future__ import print_function
from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseDistillation(nn.Module, metaclass=ABCMeta):
    """Base Distillation Loss Module For All Distillation Modules
    """
    def __init__(self, init_cfg=None):
        """Initialize BaseDistillation, inherited from `torch.nn.Module`
        Args:
            init_cfg (dict, optional): Initialization config dict.
        """
        # NOTE init_cfg can be defined in different levels, but init_cfg
        # in low levels has a higher priority.
        super(BaseDistillation, self).__init__()
        self.init_cfg = init_cfg
    
    @abstractmethod
    def forward(self, student, teacher):
        """Forward function.
        Args:
            student (tensor | tuple[tensor]): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.
            teacher (tensor | tuple[tensor]): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.
        """
        pass

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += f'\ninit_cfg={self.init_cfg}'
        return s