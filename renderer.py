

import numpy as np
import torch

from torchtyping import TensorDetail, TensorType
from typeguard import typechecked

from nerf import NeRF
from poses import Pose


@typechecked
def render(nerf: NeRF, camera_pose: Pose):
    raise NotImplemented

