import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

@typechecked
def nerf(points: TensorType["batch":..., 2]) -> TensorType["batch":...]:
    x = points[..., 0]
    y = points[..., 1]

    sharpness = 8
    return torch.sigmoid(sharpness * (y-1 )) * torch.sigmoid(sharpness * (x-1 )) 


def plot_nerf(ax, nerf):
    linspace = torch.linspace(-5,5, 100)

    # 50, 50, 2
    coods = torch.stack( torch.meshgrid( linspace, linspace ), dim=-1)
    density = nerf(coods)
    density = density.detach().numpy()

    ax.pcolormesh(coods[...,0],coods[...,1],  density, cmap = cm.binary, shading='auto')
    # plt.pcolormesh(coods[...,0],coods[...,1],  density, cmap = cm.viridis)


class System:
    def __init__(self, start_state, end_state, steps):
        self.dt = 0.1

        self.start_state = start_state[None,:]
        self.end_state = end_state[None,:]

        slider = torch.linspace(0, 1, steps)[1:-1, None]

        states = (1-slider) * start_state + slider * end_state
        # self.states = torch.tensor(states, requires_grad=True)
        self.states = states.clone().detach().requires_grad_(True)

        body = torch.stack( torch.meshgrid( torch.linspace(-0.5, 0.5, 10), 
                                            torch.linspace(-  1,   1, 10) ), dim=-1)

        self.robot_body = body.reshape(-1, 2)

    def params(self):
        return [self.states]

    def get_states(self):
        return torch.cat( [self.start_state, self.states, self.end_state], dim=0)

    def get_actions(self):
        states = self.get_states()
        prev_state = states[:-1, :]
        next_state = states[1:, :]

        middle_rot = (prev_state[:, 2] + next_state[:,2])/2
        rot_matrix = self.rot_matrix(-middle_rot) # inverse because world -> body

        lin_vel = rot_matrix @ (next_state[:, :2] - prev_state[:, :2])[...,None] / self.dt
        lin_vel = lin_vel[...,0]

        rot_vel = (next_state[:, 2:] - prev_state[:,2:])/self.dt

        return torch.cat( [lin_vel, rot_vel], dim=-1 )

    def get_hitpoints(self) -> TensorType["states", "points", 2]:
        states = self.get_states()
        pos = states[..., :2]
        rot = states[..., 2]

        # S, 2, P      S, 2, 2               2, P                S, 2, _
        body_points = self.rot_matrix(rot) @ self.robot_body.T + pos[..., None]
        return body_points.swapdims(-1,-2)

    def get_cost(self):
        actions = self.get_actions()

        x = actions[:, 0]**4
        y = actions[:, 1]**4
        a = actions[:, 2]**4

        pos = self.get_states()[:-1, :2]

        distance = (x**2 + y**2)**0.5 * self.dt
        density = nerf( self.get_hitpoints()[1:,...] )**2

        colision_prob = torch.sum( density, dim = -1) * distance

        return y*10 + a*0.1 + 0.01*x + colision_prob * 0.1

    def total_cost(self):
        return torch.sum(self.get_cost())


    @staticmethod
    @typechecked
    def rot_matrix(angle: TensorType["batch":...]) -> TensorType["batch":..., 2, 2]:
        rot_matrix = torch.zeros( angle.shape + (2,2) )
        rot_matrix[:, 0,0] =  torch.cos(angle)
        rot_matrix[:, 0,1] = -torch.sin(angle)
        rot_matrix[:, 1,0] =  torch.sin(angle)
        rot_matrix[:, 1,1] =  torch.cos(angle)
        return rot_matrix

    def plot(self, fig = None):
        if fig == None:
            fig = plt.figure(figsize=plt.figaspect(2.))
        ax_map = fig.add_subplot(2, 1, 1)
        ax_graph = fig.add_subplot(2, 1, 2)
        self.plot_map(ax_map)
        plot_nerf(ax_map, nerf)

        self.plot_graph(ax_graph) 
        plt.show()

    def plot_graph(self, ax):
        states = self.get_states().detach().numpy()
        ax.plot(states[...,0], label="x")
        ax.plot(states[...,1], label="y")
        ax.plot(states[...,2], label="a")
        actions = self.get_actions().detach().numpy() 
        ax.plot(actions[...,0], label="dx")
        ax.plot(actions[...,1], label="dy")
        ax.plot(actions[...,2], label="da")

        ax.plot(self.get_cost().detach().numpy(), label="cost")
        ax.legend()

    def plot_map(self, ax):
        ax.set_aspect('equal')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)

        states = self.get_states()
        pos = states[..., :2]
        rot = states[..., 2]

        rot_matrix = self.rot_matrix(rot)

        rot_matrix = rot_matrix.detach().numpy()
        pos        = pos.detach().numpy()

        # PLOT PATH
        ax.plot( * pos.T )

        # PLOTS BODY POINTS
        # S, P, 2
        body_points = self.get_hitpoints().detach().numpy()
        for state_body in body_points:
            ax.plot( *state_body.T, "g.", ms=72./ax.figure.dpi, alpha = 0.5)

        # PLOTS AXIS
        # create point for origin, plus a right-handed coordinate indicator.
        size = 0.5
        points = np.array( [[0, 0], [size, 0], [0, size]])
        colors = ["r", "b"]

        # S, 2, 3          =  S, 2, 2   @ 2, 3     + S, 2, _
        points_world_frame = rot_matrix @ points.T + pos[..., None]
        for p in range(pos.shape[0]):
            for i in range(1, 3):
                ax.plot(points_world_frame[p, 0, [0,i]],
                        points_world_frame[p, 1, [0,i]],
                        c=colors[i - 1],)


def main():
    start_state = torch.tensor([4,0,0])
    # end_state   = torch.tensor([3,3, np.pi/2])
    # end_state   = torch.tensor([3,3, 0])
    end_state   = torch.tensor([0,4, 0.01])

    steps = 20

    traj = System(start_state, end_state, steps)

    opt = torch.optim.Adam(traj.params(), lr=0.05)

    for it in range(1500):
        opt.zero_grad()
        loss = traj.total_cost()
        print(it, loss)
        loss.backward()

        opt.step()

    traj.plot()





if __name__ == "__main__":
    main()
