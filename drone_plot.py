import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

def main():
    start_state = torch.tensor([0,0,1, 0,0,0, 0,0,0, 0,0,0])
    end_state   = torch.tensor([10,0,1, 0,0,0, 0,0,0, 0,0,0])

    steps = 50

    traj = Trajectory(start_state, end_state, steps)

    # opt = torch.optim.Adam(traj.params, lr=0.01)

    # for it in range(1000):
    #     opt.zero_grad()
    #     loss = traj.total_cost()
    #     print(it, loss)
    #     loss.backward()

    #     opt.step()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    traj.plot(ax)

    plt.show()


class Trajectory:
    def __init__(self, start_state, end_state, steps):
        self.start_state = start_state[None,:]
        self.end_state = end_state[None,:]

        slider = torch.linspace(0, 1, steps)[1:-1, None]

        self.states = slider * start_state + (1-slider) * end_state

        self.actions = torch.zeros(steps-1, 4)

    def plot(self, ax):
        states = self.all_states()
        pos = states[..., :3]
        rot = states[..., 6:9]

        rot_matrix = vec_to_rot_matrix(rot).detach().numpy()
        pos = pos.detach().numpy()

        # ax.axis('equal')
        ax.set_xlim3d(0, 10)
        ax.set_ylim3d(-5, 5)
        ax.set_zlim3d(-5, 5)
        ax.plot( * pos.T )

        size = 0.5

        # create point for origin, plus a right-handed coordinate indicator.
        points = np.array( [[0, 0, 0], [size, 0, 0], [0, size, 0], [0, 0, size]])

        colors = ["r", "g", "b"]

        # P, 3, 4          =  P, 3, 3   @ 3, 4     + P, 3, _
        points_world_frame = rot_matrix @ points.T + pos[..., None]
        # print(points_world_frame.shape)

        for p in range(pos.shape[0]):
            for i in range(1, 4):
                ax.plot(
                        points_world_frame[p, 0, [0,i]],
                        points_world_frame[p, 1, [0,i]],
                        points_world_frame[p, 2, [0,i]],
                    c=colors[i - 1],
                )



    def params(self):
        return [self.actions, self.states]

    def all_states(self):
        return torch.cat( [self.start_state, self.states, self.end_state], dim=0)

    def dynamics_cost(self):
        cost = 0
        states = self.all_states()
        for action, state, n_state in zip(self.actions, states[:-1], states[1:]):
            pred_state = next_state(state,action)
            cost += torch.abs( pred_state - n_state)

        return cost

    def action_cost(self):
        cost = 0
        for action in self.actions:
            cost += action[0]**2
            cost += (action[1:])**2
        return cost

    def total_cost(self):
        return self.dynamics_cost() + self.action_cost()

@typechecked
def rot_matrix_to_vec( R: TensorType["batch":..., 3, 3]) -> TensorType["batch":..., 3]:
    batch_dims = R.shape[:-2]

    trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
    angle = torch.arccos((trace - 1) / 2)[..., None]

    vec = (
        1
        / (2 * torch.sin(angle))
        * torch.stack(
            [
                R[..., 2, 1] - R[..., 1, 2],
                R[..., 0, 2] - R[..., 2, 0],
                R[..., 1, 0] - R[..., 0, 1],
            ],
            dim=-1,
        )
    )

    # needed to overwrite nanes from dividing by zero
    vec[angle[..., 0] == 0] = torch.zeros(3, device=R.device)

    # eg TensorType["batch_size", "views", "max_objects", 3, 1]
    rot_vec = (angle * vec)[...]

    return rot_vec

@typechecked
def vec_to_rot_matrix(rot_vec: TensorType["batch":..., 3]) -> TensorType["batch":..., 3,3]:
    assert not torch.any(torch.isnan(rot_vec))

    angle = torch.norm(rot_vec, dim=-1, keepdim=True)
    axis = rot_vec / (1e-5 + angle)
    S = skew_matrix(axis)
    # print(S.shape)
    # print(angle.shape)
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


def next_state(state, action):
    pos = state[...,0:3]
    v   = state[...,3:6]
    euler_vector = state[...,6:9]
    omega = state[...,9:12]

    fz = action[...,0, None]
    torque = action[...,1:4]

    e3 = torch.tensor([0,0,1])

    mass = 1
    dt = 0.1
    g = 10
    J = torch.eye(3)
    J_inv = torch.linalg.inv(J)

    R = vec_to_rot_matrix(euler_vector)

    dv = g * e3 + R @ (fz * e3) / m
    domega = J_inv @ torque - J_inv @ skew_matrix(omega) @ J @ omega
    dpos = dt * v

    next_v = dv * dt
    next_euler = rot_matrix_to_vec(R @ vec_to_rot_matrix(omega * dt))
    next_omega = domega * dt
    next_pos = dpos * dt

    next_state = torch.cat([next_pos, next_v, next_euler_vector, next_omega], dim=-1)
    return next_state


if __name__ == "__main__":
    main()
