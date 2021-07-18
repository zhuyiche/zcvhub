import os
import random
import sys
import time
from getpass import getuser
from socket import gethostname

import numpy as np
import torch

from src.cvutils.runner import get_dist_info
from src.cvutils.utils import is_str


def get_host_info():
    return f'{getuser()}@{gethostname()}'


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def set_random_seed(seed, deterministic=False, use_rank_shift=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
        rank_shift (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Default: False.
    """
    if use_rank_shift:
        rank, _ = get_dist_info()
        seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False