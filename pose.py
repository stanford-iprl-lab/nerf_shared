

import numpy as np
import torch

from torchtyping import TensorDetail, TensorType
from typeguard import typechecked


class Pose(torch.nn.Module):
    def __init__(self):
        raise NotImplemented

    @typechecked
    def homogenious_mat(self) -> TensorType["batch", 4, 4]:
        raise NotImplemented

    @typechecked
    def rot_mat(self) -> TensorType["batch", 3, 3]:
        raise NotImplemented

    @typechecked
    def positions(self) -> TensorType["batch", 3]:
        raise NotImplemented

    @typechecked
    def quaternions(self) -> TensorType["batch", 4]:
        raise NotImplemented

    @typechecked
    def euler_vec(self) -> TensorType["batch", 3]:
        raise NotImplemented

