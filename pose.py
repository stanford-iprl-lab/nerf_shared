from __future__ import annotations

import numpy as np
import torch

from torchtyping import TensorDetail, TensorType
from typeguard import typechecked


# inspired by https://github.com/princeton-vl/lietorch
# reimplementing is helpful to get full understanding
# and hopefully will be easier to debug

# Michal vs torch internals
class Pose(torch.Tensor):

    from torch._C import _disabled_torch_function_impl
    __torch_function__ = _disabled_torch_function_impl

    def __new__(cls, homogenious_mat = None, requires_grad=True):
        if homogenious_mat == None:
            homogenious_mat = torch.eye(4)

        tangent = torch.zeros(6, requires_grad=True) # always zero, only used for gradients
        new_self = torch.Tensor._make_subclass(cls, tangent, requires_grad)

        new_self._manifold = torch.tensor(homogenious_mat) #actaully stores our position

        new_self.position = True
        new_self.rotation = True
        new_self.scale    = False

        return new_self

    @classmethod
    def from_components(cls, positions = None,
                             rot_mat = None, euler_vec = None, quaternion = None,
                             homogenious_mat = None,
                             scale = None):

        assert sum( x != None for x in [rot_mat, euler_vec, quaternion]) <= 1, "Rotation specified in multiple ways!"
        if homogenious_mat != None:
            assert all( x == None for x in [rot_mat, euler_vec, quaternion, positions] ), "Position/Rotation specified in multiple ways"
        assert scale == None, "Scale not supprted yet!"

        raise NotImplemented

    @typechecked
    def homogenious_mat(self) -> TensorType["batch", 4, 4]:
        return self.tanget_to_manifold(self) @ self._manifold

    @typechecked
    def rot_mat(self) -> TensorType["batch", 3, 3]:
        raise NotImplemented
        # return self._manifold[:3, :3]

    @typechecked
    def positions(self) -> TensorType["batch", 3]:
        raise NotImplemented
        # return self._manifold[:3, -1]

    @typechecked
    def quaternions(self) -> TensorType["batch", 4]:
        raise NotImplemented

    @typechecked
    def euler_vec(self) -> TensorType["batch", 3]:
        raise NotImplemented

    @typechecked
    def inv(self) -> Pose:
        return Pose( homogenious_mat = torch.linalg.inv(self._manifold) )

    @typechecked
    def __matmul__(self, other: Pose) -> Pose:
        return Pose( homogenious_mat = self._manifold @ other._manifold )

    def add_(self, exp, alpha):
        self._manifold = self.tanget_to_manifold(exp * alpha) @ self._manifold

    @staticmethod
    def tanget_to_manifold(exp):
        position  = exp[0:3]
        euler_vec = exp[3:7]
        
        out = torch.zeros((4,4))
        out[:3, :3] = vec_to_rot_matrix(euler_vec)
        out[:3, -1] = position

        out[3,3] = 1
        return out


@typechecked
def vec_to_rot_matrix(rot_vec: TensorType["batch":..., 3]) -> TensorType["batch":..., 3,3]:
    assert not torch.any(torch.isnan(rot_vec))

    angle = torch.norm(rot_vec, dim=-1, keepdim=True)

    axis = rot_vec / (1e-10 + angle)
    S = skew_matrix(axis)
    angle = angle[...,None]
    rot_matrix = (
            torch.eye(3)
            + torch.sin(angle) * S
            + (1 - torch.cos(angle)) * S @ S
            )
    return rot_matrix

@typechecked
def skew_matrix(vec: TensorType["batch":..., 3]) -> TensorType["batch":..., 3,3]:
    batch_dims = vec.shape[:-1]
    S = torch.zeros(*batch_dims, 3, 3)
    S[..., 0, 1] = -vec[..., 2]
    S[..., 0, 2] =  vec[..., 1]
    S[..., 1, 0] =  vec[..., 2]
    S[..., 1, 2] = -vec[..., 0]
    S[..., 2, 0] = -vec[..., 1]
    S[..., 2, 1] =  vec[..., 0]
    return S


