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
    start_state = torch.tensor([0,0, 1.0,  0, 0,0,   0  ,0  ,0, 0.0,0.0,0])
    end_state   = torch.tensor([5,0,1.0,  0,-0,0,  0  ,0  ,0, 0.0,0.0,0])

    steps = 20

    traj = Trajectory(start_state, end_state, steps)

    opt = torch.optim.Adam(traj.params(), lr=0.1)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)
    # opt = torch.optim.Adam([traj.states], lr=0.1)
    # opt = torch.optim.Adam([traj.actions], lr=0.1)

    # act = torch.tensor([ 20.0, -5.0,0,1])
    # state = start_state.clone()
    # states = traj.states.clone().detach()
    # for i in range(steps - 2):
    #     state = next_state(state, act)
    #     states[i] = state.clone()
    # traj.states = states


    for it in range(500):
        opt.zero_grad()
        loss = traj.total_cost()
        print(it, loss)
        loss.backward()
        # print(traj.actions.grad)

        opt.step()
        # scheduler.step()

    fig = plt.figure(figsize=plt.figaspect(2.))
    ax3d = fig.add_subplot(2, 1, 1, projection='3d')
    ax2d = fig.add_subplot(2, 1, 2)
    traj.plot(ax3d)
    traj.plot_action(ax2d) 
    plt.show()


class Trajectory:
    def __init__(self, start_state, end_state, steps):
        self.start_state = start_state[None,:]
        self.end_state = end_state[None,:]

        slider = torch.linspace(0, 1, steps)[1:-1, None]

        states = (1-slider) * start_state + slider * end_state
        # self.states = torch.tensor(states, requires_grad=True)
        self.states = states.clone().detach().requires_grad_(True)

        self.actions = torch.zeros(steps-1, 4, requires_grad=True)
        # self.actions = torch.randn(steps-1, 4, requires_grad=True)

    def plot_action(self, ax):
        actions = self.actions.detach().numpy() 
        ax.plot(actions[...,0], label="fz")
        ax.plot(actions[...,1], label="tx")
        ax.plot(actions[...,2], label="ty")
        ax.plot(actions[...,3], label="tz")

        states = self.states.detach().numpy()
        ax.plot(states[...,0], label="px")
        ax.plot(states[...,4], label="vx")
        ax.plot(states[...,7], label="ey")


        cost = []
        states = self.all_states()
        for action, state, n_state in zip(self.actions, states[:-1], states[1:]):
            pred_state = next_state(state,action)
            cost.append( torch.sum( torch.abs( pred_state - n_state) ).detach() )
        ax.plot(cost, label="cost")

        ax.legend()

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
        steps = states.shape[0]
        # for action, state, n_state in zip(self.actions, states[:-1], states[1:]):
        #     pred_state = next_state(state,action)
        #     cost += torch.sum( torch.abs( pred_state - n_state) )

        # for a1,a2,a3, state, n_state in zip(self.actions[:-2],self.actions[1:-1],self.actions[2:], states[:-3], states[3:]):
        #     pred_state = next_state(next_state(next_state(state,a1),a2),a3)
        #     cost += torch.sum( torch.abs( pred_state - n_state) )

        # parabola = (states[...,0] - 5)**2  - states[...,2] - 3
        # cost += torch.sum(F.relu(parabola))

        depth = 2
        for i in range(steps-depth):
            actions = self.actions[i:i+depth]
            local_states = states[i:i+depth+1]
            state = local_states[0]
            future_states = local_states[1:]

            for action, compare_state in zip(actions, future_states):
                state = next_state(state, action)
                cost += torch.sum( torch.abs(state - compare_state) )

        return cost

    def action_cost(self):
        cost = 0
        for action in self.actions:
            cost += action[0]**2
            cost += torch.sum( (action[1:])**2 )
        return cost

    def total_cost(self):
        return self.dynamics_cost()# + 0.1* self.action_cost()

@typechecked
def rot_matrix_to_vec( R: TensorType["batch":..., 3, 3]) -> TensorType["batch":..., 3]:
    batch_dims = R.shape[:-2]

    trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)

    def acos_safe(x, eps=1e-4):
        """https://github.com/pytorch/pytorch/issues/8069"""
        slope = np.arccos(1-eps) / eps
        # TODO: stop doing this allocation once sparse gradients with NaNs (like in
        # th.where) are handled differently.
        buf = torch.empty_like(x)
        good = abs(x) <= 1-eps
        bad = ~good
        sign = torch.sign(x[bad])
        buf[good] = torch.acos(x[good])
        buf[bad] = torch.acos(sign * (1 - eps)) - slope*sign*(abs(x[bad]) - 1 + eps)
        return buf

    angle = acos_safe((trace - 1) / 2)[..., None]
    # print(trace, angle)

    vec = (
        1
        / (2 * torch.sin(angle + 1e-5))
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
    #TODO batch this
    pos = state[...,0:3]
    v   = state[...,3:6]
    euler_vector = state[...,6:9]
    omega = state[...,9:12]

    fz = action[...,0, None]
    torque = action[...,1:4]

    e3 = torch.tensor([0,0,1])

    mass = 1
    dt = 0.1
    g = -10
    J = torch.eye(3)
    J_inv = torch.linalg.inv(J)

    R = vec_to_rot_matrix(euler_vector)

    dv = g * e3 + R @ (fz * e3) / mass
    domega = J_inv @ torque - J_inv @ skew_matrix(omega) @ J @ omega
    dpos = v

    next_v = dv * dt + v
    next_euler = rot_matrix_to_vec(R @ vec_to_rot_matrix(omega * dt))
    # next_euler = rot_matrix_to_vec(R @ vec_to_rot_matrix(omega * dt) @ vec_to_rot_matrix(domega * dt**2))
    next_omega = domega * dt + omega
    next_pos = dpos * dt + pos# + 0.5 * dv * dt**2

    next_state = torch.cat([next_pos, next_v, next_euler, next_omega], dim=-1)
    return next_state


if __name__ == "__main__":
    main()
