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

torch.manual_seed(0)
np.random.seed(0)

from quad_helpers import Simulator, QuadPlot
from quad_helpers import rot_matrix_to_vec, vec_to_rot_matrix

# # hard coded "nerf" for testing. see below to import real nerf
def get_manual_nerf(name):
    if name =='empty':
        class FakeRenderer:
            @typechecked
            def get_density(self, points: TensorType["batch":..., 3]) -> TensorType["batch":...]:
                return torch.zeros_like( points[...,0] )
        return FakeRenderer()

    if name =='cylinder':
        class FakeRenderer:
            @typechecked
            def get_density(self, points: TensorType["batch":..., 3]) -> TensorType["batch":...]:
                x = points[..., 0]
                y = points[..., 1] - 1

                return torch.sigmoid( (2 -(x**2 + y**2)) * 8 )
        return FakeRenderer()

    raise ValueError



class System:
    def __init__(self, renderer, start_state, end_state, cfg):
        self.nerf = renderer.get_density

        self.T_final            = cfg['T_final']
        self.steps              = cfg['steps']
        self.lr                 = cfg['lr']
        self.epochs_init        = cfg['epochs_init']
        self.epochs_update      = cfg['epochs_update']
        self.fade_out_epoch     = cfg['fade_out_epoch']
        self.fade_out_sharpness = cfg['fade_out_sharpness']

        self.dt = self.T_final / self.steps

        start_vel = torch.cat( [start_state[3:6], torch.zeros(1)], dim =0)
        end_vel = torch.cat( [end_state[3:6], torch.zeros(1)], dim =0)

        start_state= torch.cat( [start_state[:3], torch.zeros(1)], dim =0)
        end_state = torch.cat( [end_state[:3], torch.zeros(1)], dim =0)


        # create initial and final 3 states to constrain: position, velocity and possibly angle in the future
        self.start_states = start_state[None,:] + torch.tensor([-1,0,1])[:,None] * self.dt * start_vel
        self.end_states   = end_state[None,:]   + torch.tensor([-1,0,1])[:,None] * self.dt * end_vel  

        slider = torch.linspace(0, 1, self.steps)[1:-1, None]

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
        J = torch.eye(3)

        rot_matrix, z_accel, _ = self.get_rots_and_accel()

        #TODO horrible -> there should be a better way without rotation matricies
        #calculate angular velocities
        ang_vel = rot_matrix_to_vec( rot_matrix[1:, ...] @ rot_matrix[:-1, ...].swapdims(-1,-2) ) / self.dt

        # if not torch.allclose( rot_matrix @ rot_matrix.swapdims(-1,-2), torch.eye(3)):
        #     print( rot_matrix @ rot_matrix.swapdims(-1,-2), torch.eye(3) )
        #     assert False

        #calculate angular acceleration
        angular_accel = (ang_vel[1:,...] - ang_vel[:-1,...])/self.dt

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

        current_vel = (next_vel + prev_vel)/2

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

        # return pos, current_vel, rot_matrix, angular_rate, 
        return rot_matrix, z_accel, current_vel

    def get_next_action(self) -> TensorType["state_dim"]:
        actions = self.get_actions()
        # fz, tx, ty, tz
        return actions[0, :]

    def get_full_state(self):
        rot_matrix, z_accel, current_vel = self.get_rots_and_accel()

        # pos, vel, rotation matrix
        return states[:, :3], current_vel, rot_matrix


    @typechecked
    def body_to_world(self, points: TensorType["batch", 3]) -> TensorType["states", "batch", 3]:
        states = self.get_states()
        pos = states[:, :3]
        rot_matrix, _, _ = self.get_rots_and_accel()

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

        if self.epoch < self.fade_out_epoch:
            t = torch.linspace(0,1, colision_prob.shape[0])
            position = self.epoch/self.fade_out_epoch
            mask = torch.sigmoid(self.fade_out_sharpness * (position - t))
            colision_prob = colision_prob * mask

        #PARAM cost function shaping
        return 1000*fz**2 + 0.01*torques**2 + colision_prob * 1e6, colision_prob*1e6

    def total_cost(self):
        total_cost, colision_loss = self.get_cost()
        return torch.mean(total_cost)

    def learn_init(self):
        opt = torch.optim.Adam(self.params(), lr=self.lr)

        try:
            for it in range(self.epochs_init):
                opt.zero_grad()
                self.epoch = it
                loss = self.total_cost()
                print(it, loss)
                loss.backward()
                opt.step()

                save_step = 50
                if it%save_step == 0:
                    self.save_poses("paths/"+str(it//save_step)+"_testing.json")

        except KeyboardInterrupt:
            print("finishing early")

    def learn_update(self):
        opt = torch.optim.Adam(self.params(), lr=self.lr)

        for it in range(self.epochs_update):
            opt.zero_grad()
            self.epoch = it
            loss = self.total_cost()
            print(it, loss)
            loss.backward()
            opt.step()

            # save_step = 50
            # if it%save_step == 0:
        # self.save_poses("paths/"+str(it//save_step)+"_testing.json")

    @typechecked
    def update_state(self, measured_state: TensorType["state_dim"]):
        measured_state = measured_state[None, :]
        print(self.start_states.shape)
        print(measured_state.shape)
        self.start_states = torch.cat( [self.start_states, measured_state], dim=0 )
        self.states = self.states[1:, :].detach().requires_grad_(True)

    def plot(self, quadplot):

        quadplot.trajectory( self, "g" )
        ax = quadplot.ax_graph

        actions = self.get_actions().detach().numpy() 
        ax.plot(actions[...,0], label="fz")
        ax.plot(actions[...,1], label="tx")
        ax.plot(actions[...,2], label="ty")
        ax.plot(actions[...,3], label="tz")

        ax_right = quadplot.ax_graph_right

        total_cost, colision_loss = self.get_cost()
        ax_right.plot(total_cost.detach().numpy(), 'black', label="cost")
        ax_right.plot(colision_loss.detach().numpy(), 'cyan', label="colision")
        ax.legend()


    def save_poses(self, filename):
        states = self.get_states()
        rot_mats, _, _ = self.get_rots_and_accel()

        with open(filename,"w+") as f:
            for pos, rot in zip(states[...,:3], rot_mats):
                pose = np.zeros((4,4))
                pose[:3, :3] = rot.detach().numpy()
                pose[:3, 3]  = pos.detach().numpy()
                pose[3,3] = 1

                json.dump(pose.tolist(), f)
                f.write('\n')


def main():

    # renderer = get_nerf('configs/stonehenge.txt')
    # stonehenge - simple
    start_pos = torch.tensor([-0.05,-0.9, 0.2])
    end_pos   = torch.tensor([-1 , 0.7, 0.05])

    start_state = torch.cat( [start_pos, torch.zeros(3), torch.eye(3).reshape(-1), torch.zeros(3)], dim=0 )
    end_state   = torch.cat( [end_pos,   torch.zeros(3), torch.eye(3).reshape(-1), torch.zeros(3)], dim=0 )

    renderer = get_manual_nerf("empty")

    cfg = {"T_final": 2,
            "steps": 20,
            "lr": 0.001,
            "epochs_init": 2500,
            "fade_out_epoch": 500,
            "fade_out_sharpness": 10,
            "epochs_update": 200,
            }

    traj = System(renderer, start_state, end_state, cfg)
    traj.learn_init()


    sim = Simulator(start_state)

    quadplot = QuadPlot()
    traj.plot(quadplot)
    quadplot.show()


    if True:
        for step in range(cfg['steps']):
            # # idealy something like this but we jank it for now
            # action = traj.get_actions()[0 or 1, :]

            # action = traj.get_next_action()
            # action = traj.get_actions()[step,:]

            # current_state = next_state(action)

            # action = traj.get_actions()[0 or 1, :]

            # we jank it

            current_state = traj.states[0, :].detach()
            sim.advance(action)

            # randomness = torch.normal(mean= 0, std=torch.tensor([0.02, 0.02, 0.02, 0.1]) )

            # measured_state = current_state + randomness
            full_state = sim.get_current_state().detach()

            traj.update_state( torch.cat([full_state[:3], torch.zeros(1)]) )


            traj.learn_update()
            # traj.save_poses(???)

            print("sim step", step)

            quadplot = QuadPlot()
            traj.plot(quadplot)
            quadplot.trajectory( sim, "r" )
            quadplot.show()



    #PARAM file to save the trajectory
    # traj.save_poses("paths/playground_testing.json")
    # traj.plot()


if __name__ == "__main__":
    main()
