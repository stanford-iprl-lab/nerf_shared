
import numpy as np
import torch

from torchtyping import TensorDetail, TensorType
from typeguard import typechecked


class Nerf(torch.nn.Module):
    def __init__(self, cfg):
        raise NotImplemented

    @typechecked
    def forward(self, points: TensorType["dims":..., 3], view_direction: TensorType["dims":..., 3]):
        raise NotImplemented

    @typechecked
    def density(self, points: TensorType["dims":..., 3]):
        raise NotImplemented

