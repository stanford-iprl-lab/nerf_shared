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

        # relocalize
        corrected_pose = relocalize(nerf, guess_pose, rendered_points)

        # replan
        new_trajectory = replan(nerf, corrected_pose, old_trajectroy)

        # log new path


class Trajectory:
    def __init__(self, start_pose, end_pose):
        self.start_pose = start_pose
        self.end_pose = end_pose

        self.params = TODO

    def get_homogenious_mats(self):
        pass


def initial_path_parameters(nerf, start_pose, end_pose) -> TensorType["batch":..., 2,3]:
    pass

def path_from_parameters(params : TensorType["batch":..., 2,3]) -> TensorType["batch":..., 4,4]:
    """ params[...,0,:] are 3-vectors representing translation
        params[...,1,:] are 3-vectors represention rotation in a axis angle/euler vector representation
        return type is a 4x4 homogenouis transforms """
    pass

def path_cost(nerf, path: TensorType["batch":..., 4,4]):
    pass
