
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
        view_dir = torch.ones_like(points)
        output = self.forward(points, view_dir)
        return output[..., -1]

class CompositeNerf(Nerf):
    def __init__(self, nerf_list, pose_list, cfg):
        self.nerf_list = nerf_list
        self.pose_list = nerf_list

        raise NotImplemented

    @typechecked
    def forward(self, points: TensorType["dims":..., 3], view_direction: TensorType["dims":..., 3]) -> TensorType["dims":...,4]:
        raise NotImplemented

        for nerf, pose in zip(self.nerf_list, self.pose_list):
            nerf.forward(points, view_direction)

        # alpha combine

    @typechecked
    def density(self, points: TensorType["dims":..., 3]) -> TensorType["dims":...]:
        raise NotImplemented
