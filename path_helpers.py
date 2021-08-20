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

    start_pose = None # 4x4 homogenious pytorch tensor (entered by hand)
    end_pose = None # 4x4 homogenious pytorch tensor (entered by hand)

    # prepare initial path
    traj = Trajectory(nerf, start_pose, end_pose)

    for iteration in range(1000):
        # execute step
        # add randomness to action
        # 2 options:
        #    - use next pose and add randomness to that
        #    - use action, add randomnes and integrate forward
        guess_pose = traj.next_non_start_param()
        real_pose = guess_pose + 0.1*torch.random(2,3) # terrible way to add randomness, more for ilustration

        # relocalize TODO
        corrected_pose = relocalize(nerf, guess_pose, real_pose)

        # replan
        traj.update_trajectory(nerf,corrected_pose)
        traj.log_to_file("log.txt") # will be used to generate visuals for paper

def nerf(points: TensorType["batch":..., 3]) -> TensorType["batch":..., 1]:
    pass

def render_nerf(pose) -> :
    pass

def relocalize(nerf, guess_pose, real_pose):
    pass
    # rough outline, 
    
    opt = torch.optim.Adam(guess_pose, lr=0.0001)

    render_nerf(real_pose)
    render_nerf(corrected_pose)

    for _ in range(100):
        pass

    return guess_pose


class Trajectory:
    def __init__(self, nerf, start_pose: TensorType[4,4], end_pose: TensorType[4,4]):
        self.start_pose = start_pose
        self.end_pose = end_pose

        steps = 100
        self.params = self.initial_path_parameters(nerf, start_pose, end_pose, steps)

    def get_homogenious_mats(self) -> TensorType["path_length":..., 4,4]:
        middle = self.path_from_parameters(self.params)
        return torch.concat( [self.start_pose, middle, self.end_pose], dim=0 )

    def next_non_start_param(self) -> TensorType[2,3]:
        return self.params[0,:,:]

    def update_trajectory(self, nerf, real_param):
        self.start_pose = self.path_from_parameters(real_param)
        self.params = self.params[1:, ...]
        self.optimize_path(nerf)

    def log_to_file(self, file_name):
        pass

    def optimize_path(self, nerf):
        opt = torch.optim.Adam(self.params, lr=0.0001)
        for _ in range(100):
            opt.zero_grad()
            cost = path_cost(nerf, self.get_homogenious_mats())
            cost.backward()
            opt.step()

    @staticmethod
    def initial_path_parameters(nerf, start_pose, end_pose, steps) -> TensorType["path_length":..., 2,3]:
        pass
        # run A* (or initialize as stright line path?)

    @staticmethod
    @typechecked
    def path_from_parameters(params : TensorType["path_length":..., 2,3]) -> TensorType["path_length":..., 4,4]:
        """ params[...,0,:] are 3-vectors representing translation
            params[...,1,:] are 3-vectors represention rotation in a axis angle/euler vector representation
            return type is a 4x4 homogenouis transforms """

        assert not torch.any(torch.isnan(params))
        device = parmas.device

        trans_vec = params[...,0,:]
        rot_vec   = params[...,1,:]

        angle = torch.norm(rot_vec, dim=-1, keepdim=True)
        axis = rot_vec / (1e-5 + angle)

        # S is skew matrix
        batch_dims = rot_vec.shape[:-1]
        S = torch.zeros(*batch_dims, 3, 3, device=device)
        S[..., 0, 1] = -axis[..., 2]
        S[..., 0, 2] =  axis[..., 1]
        S[..., 1, 0] =  axis[..., 2]
        S[..., 1, 2] = -axis[..., 0]
        S[..., 2, 0] = -axis[..., 1]
        S[..., 2, 1] =  axis[..., 0]

        rot_matrix = (
                torch.eye(3, device=device)
                + torch.sin(angle) * S
                + (1 - torch.cos(angle)) * S @ S
                )

        transform = torch.cat([rot_matrix, trans_vec[:, None]], dim=-1)

        last_row = torch.tensor([[0, 0, 0, 1]], device=device).expand(
                *transform.shape[0:-2], 1, 4
                )

        transform = torch.cat([transform, last_row], dim=-2)
        return transform



def path_cost(nerf, path: TensorType["batch":..., 4,4]):
    pass
