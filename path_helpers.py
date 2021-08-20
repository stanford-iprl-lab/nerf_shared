import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

def main():
    # load nerf from file TODO
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)

    start_pose = None # 4x4 homogenious pytorch tensor
    end_pose = None # 4x4 homogenious pytorch tensor

    # prepare initial path
    path_parameter = initial_path_parameters(nerf, start_pose, end_pose)


    for iteration in range(1000):
        # execute step
        # 2 options:
        #    - use next pose and add randomness to that
        #    - use action, add randomnes and integrate forward

        # add randomness to action
        param = traj.next_non_start_param()

        # relocalize
        corrected_pose = relocalize(nerf, guess_pose, real_pose)

        # replan
        new_trajectory = replan(nerf, corrected_pose, old_trajectroy)

        # log new path

def nerf(points: TensorType["batch":..., 3]) -> TensorType["batch":..., 1]:
    pass

def render_nerf(pose) -> :
    pass

class Trajectory:
    def __init__(self, nerf, start_pose, end_pose):
        self.start_pose = start_pose
        self.end_pose = end_pose

        self.params = self.initial_path_parameters(nerf, start_pose, end_pose, steps)

    def get_homogenious_mats(self) -> TensorType["path_length":..., 4,4]:
        middle = self.path_from_parameters(self.params)
        return torch.concat( [self.start_pose, middle, self.end_pose], dim=0 )

    def next_non_start_param(self) -> TensorType[2,3]:
        return self.params[0,:,:]

    def update_trajectory(self, real_param):
        pass


    @staticmethod
    def initial_path_parameters(nerf, start_pose, end_pose, steps) -> TensorType["path_length":..., 2,3]:
        pass

    @staticmethod
    def path_from_parameters(params : TensorType["path_length":..., 2,3]) -> TensorType["path_length":..., 4,4]:
        """ params[...,0,:] are 3-vectors representing translation
            params[...,1,:] are 3-vectors represention rotation in a axis angle/euler vector representation
            return type is a 4x4 homogenouis transforms """
        pass


def path_cost(nerf, path: TensorType["batch":..., 4,4]):
    pass
