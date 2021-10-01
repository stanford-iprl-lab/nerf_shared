
import numpy as np
import torch

from torchtyping import TensorDetail, TensorType
from typeguard import typechecked


class Nerf(torch.nn.Module):
    def __init__(self, cfg):
        raise NotImplemented

    @typechecked
    def forward(self, points: TensorType["dims":..., 3], view_direction: TensorType["dims":..., 3]) -> TensorType["dims":...,4]:
        raise NotImplemented

    @typechecked
    def density(self, points: TensorType["dims":..., 3]) -> TensorType["dims":...]:
        raise NotImplemented

class CompositeNerf(Nerf):
    def __init__(self, nerf_list, pose_list, cfg):
        raise NotImplemented

    @typechecked
    def forward(self, points: TensorType["dims":..., 3], view_direction: TensorType["dims":..., 3]) -> TensorType["dims":...,4]:
        raise NotImplemented

    @typechecked
    def density(self, points: TensorType["dims":..., 3]) -> TensorType["dims":...]:
        raise NotImplemented
