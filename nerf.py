
import numpy as np
import torch

from torchtyping import TensorDetail, TensorType
from typeguard import typechecked

class NerfMLP(torch.nn.Module):
    """ Implements the raw NeRF MLP """
    def __init__(self, cfg):
        raise NotImplemented

    @typechecked
    def forward(self, points: TensorType["dims":..., 3], view_direction: TensorType["dims":..., 3]) -> TensorType["dims":...,4]:
        raise NotImplemented

class Nerf(torch.nn.Module):
    """ Wrapper around a course and fine NeRF network """
    def __init__(self, cfg):
        raise NotImplemented
        self.course_mlp = NerfMLP()
        self.fine_mlp = NerfMLP()

    @typechecked
    def forward(self, points: TensorType["dims":..., 3], view_direction: TensorType["dims":..., 3], fine_network = True) -> TensorType["dims":...,4]:
        raise NotImplemented

    @typechecked
    def density(self, points: TensorType["dims":..., 3], fine_network = True) -> TensorType["dims":...]:
        view_dir = torch.ones_like(points)
        output = self.forward(points, view_dir, fine_network)
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
