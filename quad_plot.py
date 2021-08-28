import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import json

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()


from load_nerf import get_nerf

# # hard coded "nerf" for testing. see below to import real nerf
# def get_nerf(*args, **kwargs):
#     @typechecked
#     def nerf(points: TensorType["batch":..., 3]) -> TensorType["batch":...]:
#         x = points[..., 0]
#         y = points[..., 1] - 1

#         return torch.sigmoid( (2 -(x**2 + y**2)) * 8 )
#     return nerf


def plot_nerf(ax_map, nerf):
    # can plot nerf in matplotlib but hard to be interpretable
    pass


class System:
    def __init__(self, nerf, start_state, end_state, start_vel, end_vel, steps, dt):
        self.nerf = nerf
        self.dt = dt

        # create initial and final 3 states to constrain: position, velocity and possibly angle in the future
        self.start_states = start_state[None,:] + torch.tensor([-1,0,1])[:,None] * self.dt * start_vel
        self.end_states   = end_state[None,:]   + torch.tensor([-1,0,1])[:,None] * self.dt * end_vel  

        slider = torch.linspace(0, 1, steps)[1:-1, None]

        states = (1-slider) * self.start_states[-1,:] + slider * self.end_states[0,:]
        self.states = states.clone().detach().requires_grad_(True)

        #PARAM this sets the shape of the robot body point cloud
        body = torch.stack( torch.meshgrid( torch.linspace(-0.05, 0.05, 10),
                                            torch.linspace(-0.05, 0.05, 10),
                                            torch.linspace(-0.02, 0.02,  5)), dim=-1)
        self.robot_body = body.reshape(-1, 3)
        # self.robot_body = torch.zeros(1,3)

        self.epoch = 0

    def params(self):
        return [self.states]

    def get_states(self):
        return torch.cat( [self.start_states, self.states, self.end_states], dim=0)

    def get_actions(self):
        mass = 1

        rot_matrix, z_accel = self.get_rots_and_accel()

        #TODO horrible -> there should be a better way without rotation matricies
        #calculate angular velocities
        ang_vel = rot_matrix_to_vec( rot_matrix[1:, ...] @ rot_matrix[:-1, ...].swapdims(-1,-2) ) / self.dt

        # if not torch.allclose( rot_matrix @ rot_matrix.swapdims(-1,-2), torch.eye(3)):
        #     print( rot_matrix @ rot_matrix.swapdims(-1,-2), torch.eye(3) )
        #     assert False

        #calculate angular acceleration
        angular_accel = (ang_vel[1:,...] - ang_vel[:-1,...])/self.dt

        J = torch.eye(3)
        # S, 3    3,3      S, 3, 1
        torques = (J @ angular_accel[...,None])[...,0]

        return torch.cat([ z_accel*mass, torques ], dim=-1)

    def get_rots_and_accel(self):
        g = torch.tensor([0,0,-10])

        states = self.get_states()
        prev_state = states[:-1, :]
        next_state = states[1:, :]

        diff = (next_state - prev_state)/self.dt
        vel = diff[..., :3]

        prev_vel = vel[:-1, :]
        next_vel = vel[1:, :]

        target_accel = (next_vel - prev_vel)/self.dt - g
        z_accel     = torch.norm(target_accel, dim=-1, keepdim=True)

        # needs to be pointing in direction of acceleration
        z_axis_body = target_accel/z_accel

        #duplicate first and last angle to enforce zero angular velocity constraint
        z_axis_body = torch.cat( [ z_axis_body[:1,:], z_axis_body, z_axis_body[-1:,:]], dim=0)

        z_angle = states[:,3]
        in_plane_heading = torch.stack( [torch.sin(z_angle), -torch.cos(z_angle), torch.zeros_like(z_angle)], dim=-1)

        x_axis_body = torch.cross(z_axis_body, in_plane_heading, dim=-1)
        x_axis_body = x_axis_body/torch.norm(x_axis_body, dim=-1, keepdim=True)
        y_axis_body = torch.cross(z_axis_body, x_axis_body, dim=-1)

        # S, 3, 3 # assembled manually from basis vectors
        rot_matrix = torch.stack( [x_axis_body, y_axis_body, z_axis_body], dim=-1)
        return rot_matrix, z_accel

    @typechecked
    def body_to_world(self, points: TensorType["batch", 3]) -> TensorType["states", "batch", 3]:
        states = self.get_states()
        pos = states[:, :3]
        rot_matrix, _ = self.get_rots_and_accel()

        # S, 3, P    =    S,3,3       3,P       S, 3, _
        world_points =  rot_matrix @ points.T + pos[..., None]
        return world_points.swapdims(-1,-2)

    def get_cost(self):
        actions = self.get_actions()

        fz = actions[:, 0]
        torques = torch.norm(actions[:, 1:], dim=-1)**2

        states = self.get_states()
        prev_state = states[:-1, :]
        next_state = states[1:, :]

        # multiplied by distance to prevent it from just speed tunnelling
        distance = torch.sum( (next_state - prev_state)[...,:3]**2 + 1e-5, dim = -1)**0.5
        density = self.nerf( self.body_to_world(self.robot_body)[1:,...] )**2
        colision_prob = torch.mean( density, dim = -1) * distance
        colision_prob = colision_prob[1:]


        fade_out_epoch = 1000
        sharpness = 10
        if self.epoch < fade_out_epoch:
            t = torch.linspace(0,1, colision_prob.shape[0])
            position = self.epoch/fade_out_epoch
            mask = torch.sigmoid(sharpness * (position - t))
            colision_prob = colision_prob * mask

        #PARAM cost function shaping
        return 1000*fz**2 + 0.01*torques**2 + colision_prob * 1e6, colision_prob*1e6

    def total_cost(self):
        total_cost, colision_loss = self.get_cost()
        return torch.mean(total_cost)


    def plot(self, fig = None):
        if fig == None:
            fig = plt.figure(figsize=(16, 8))

        ax_map = fig.add_subplot(1, 2, 1, projection='3d')
        ax_graph = fig.add_subplot(1, 2, 2)
        self.plot_map(ax_map)
        plot_nerf(ax_map, self.nerf)

        self.plot_graph(ax_graph) 
        plt.tight_layout()
        plt.show()

    def plot_graph(self, ax):
        actions = self.get_actions().detach().numpy() 
        ax.plot(actions[...,0], label="fz")
        ax.plot(actions[...,1], label="tx")
        ax.plot(actions[...,2], label="ty")
        ax.plot(actions[...,3], label="tz")

        # states = self.states.detach().numpy()
        # ax.plot(states[...,0], label="px")
        # ax.plot(states[...,4], label="vx")
        # ax.plot(states[...,7], label="ey")

        ax_right = ax.twinx()

        total_cost, colision_loss = self.get_cost()
        ax_right.plot(total_cost.detach().numpy(), 'black', label="cost")
        ax_right.plot(colision_loss.detach().numpy(), 'cyan', label="colision")
        ax.legend()

    def plot_map(self, ax):
        ax.auto_scale_xyz([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
        ax.set_ylim3d(-1, 1)
        ax.set_xlim3d(-1, 1)
        ax.set_zlim3d( 0, 1)

        # PLOT PATH
        # S, 1, 3
        pos = self.body_to_world( torch.zeros((1,3))).detach().numpy()
        # print(pos.shape)
        ax.plot( pos[:,0,0], pos[:,0,1],   pos[:,0,2],  )

        # PLOTS BODY POINTS
        # S, P, 2
        body_points = self.body_to_world( self.robot_body ).detach().numpy()
        for state_body in body_points:
            ax.plot( *state_body.T, "g.", ms=72./ax.figure.dpi, alpha = 0.5)

        # PLOTS AXIS
        # create point for origin, plus a right-handed coordinate indicator.
        size = 0.05
        points = torch.tensor( [[0, 0, 0], [size, 0, 0], [0, size, 0], [0, 0, size]])
        colors = ["r", "g", "b"]

        # S, 4, 2
        points_world_frame = self.body_to_world(points).detach().numpy()
        for state_axis in points_world_frame:
            for i in range(1, 4):
                ax.plot(state_axis[[0,i], 0],
                        state_axis[[0,i], 1],
                        state_axis[[0,i], 2],
                    c=colors[i - 1],)


    def save_poses(self, filename):
        states = self.get_states()
        rot_mats, _ = self.get_rots_and_accel()

        with open(filename,"w+") as f:
            for pos, rot in zip(states[...,:3], rot_mats):
                pose = np.zeros((4,4))
                pose[:3, :3] = rot.detach().numpy()
                pose[:3, 3]  = pos.detach().numpy()
                pose[3,3] = 1

                json.dump(pose.tolist(), f)
                f.write('\n')


def main():
    #PARAM nerf config
    nerf = get_nerf('configs/playground.txt')

    #PARAM start and end positions for the planner. [x,y,z,yaw]
    # start_state = torch.tensor([0, -0.8, 0.01, 0])
    # end_state   = torch.tensor([0,  0.9, 0.6 , 0])
    
    # start_state = torch.tensor([-0.11, -0.7, 0.7, 0])
    # end_state   = torch.tensor([-0.11, 0.45, 0.7, 0])

    start_state = torch.tensor([ 0.25, -0.47, 0.01, 0])
    end_state   = torch.tensor([-0.25,  0.6,  0.6 , 0])

    #PARAM initial and final velocities
    start_vel = torch.tensor([0, 0, 0, 0])
    end_vel   = torch.tensor([0, 0, 0, 0])

    #PARAM
    steps = 40
    dt = 0.1

    traj = System(nerf, start_state, end_state, start_vel, end_vel, steps, dt)

    #PARAM learning rate for the path optimization
    opt = torch.optim.Adam(traj.params(), lr=0.001)

    try:
        for it in range(2500):
            opt.zero_grad()
            traj.epoch = it
            loss = traj.total_cost()
            print(it, loss)
            loss.backward()
            opt.step()

            save_step = 50
            if it%save_step == 0:
                traj.save_poses("paths/"+str(it//save_step)+"_testing.json")

    except KeyboardInterrupt:
        print("finishing early")

    #PARAM file to save the trajectory
    traj.save_poses("paths/playground_testing.json")
    traj.plot()

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


if __name__ == "__main__":
    main()
