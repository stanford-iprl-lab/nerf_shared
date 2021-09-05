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
    @typechecked
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

        self.mass = 1
        self.J = torch.eye(3)
        self.g = torch.tensor([0,0,-10])

        self.start_state = start_state
        self.end_state   = end_state

        slider = torch.linspace(0, 1, self.steps)[1:-1, None]

        states = (1-slider) * self.full_to_reduced_state(start_state) + \
                    slider  * self.full_to_reduced_state(end_state)

        self.states = states.clone().detach().requires_grad_(True)

        #PARAM this sets the shape of the robot body point cloud
        body = torch.stack( torch.meshgrid( torch.linspace(-0.05, 0.05, 10),
                                            torch.linspace(-0.05, 0.05, 10),
                                            torch.linspace(-0.02, 0.02,  5)), dim=-1)
        self.robot_body = body.reshape(-1, 3)
        # self.robot_body = torch.zeros(1,3)

        self.epoch = 0

    @typechecked
    def full_to_reduced_state(self, state: TensorType[18]) -> TensorType[4]:
        pos = state[:3]
        R = state[6:15].reshape((3,3))

        x,y,_ = R @ torch.tensor( [1.0, 0, 0 ] )
        angle = torch.atan2(y, x)

        return torch.cat( [pos, torch.tensor([angle]) ], dim = -1).detach()


    def params(self):
        return [self.states]

    # @typechecked
    def calc_everything(self) -> (
            TensorType["states", 3], #pos
            TensorType["states", 3], #vel
            TensorType["states", 3], #accel
            TensorType["states", 3,3], #rot_matrix
            TensorType["states", 3], #omega
            TensorType["states", 3], #angualr_accel
            TensorType["states", 4], #actions
        ):

        start_pos   = self.start_state[None, 0:3]
        start_v     = self.start_state[None, 3:6]
        start_R     = self.start_state[6:15].reshape((1, 3, 3))
        start_omega = self.start_state[None, 15:]

        end_pos   = self.end_state[None, 0:3]
        end_v     = self.end_state[None, 3:6]
        end_R     = self.end_state[6:15].reshape((1, 3, 3))
        end_omega = self.end_state[None, 15:]

        current_pos = torch.cat( [start_pos, self.states[:, :3], end_pos], dim=0)
        prev_pos = current_pos[:-1, :]
        next_pos = current_pos[1: , :]

        midpoint_vel = (next_pos - prev_pos)/self.dt

        # reverse of averaging midpoint to get real value
        prestart_vel = 2*start_v - midpoint_vel[0,None,:]
        postend_vel  = 2*end_v   - midpoint_vel[-1,None,:]

        midpoint_vel = torch.cat( [ prestart_vel, midpoint_vel, postend_vel], dim=0)
        prev_vel = midpoint_vel[:-1, :]
        next_vel = midpoint_vel[1:, :]

        current_vel = (next_vel + prev_vel)/2

        current_accel = (next_vel - prev_vel)/self.dt - self.g

        accel_mag     = torch.norm(current_accel, dim=-1, keepdim=True)

        # needs to be pointing in direction of acceleration
        z_axis_body = current_accel/accel_mag

        # remove first and last state - we already have their rotations constrained
        z_axis_body = z_axis_body[1:-1, :]

        z_angle = self.states[:,3]
        in_plane_heading = torch.stack( [torch.sin(z_angle), -torch.cos(z_angle), torch.zeros_like(z_angle)], dim=-1)

        x_axis_body = torch.cross(z_axis_body, in_plane_heading, dim=-1)
        x_axis_body = x_axis_body/torch.norm(x_axis_body, dim=-1, keepdim=True)
        y_axis_body = torch.cross(z_axis_body, x_axis_body, dim=-1)

        # S, 3, 3 # assembled manually from basis vectors
        rot_matrix = torch.stack( [x_axis_body, y_axis_body, z_axis_body], dim=-1)
        rot_matrix = torch.cat( [start_R, rot_matrix, end_R], dim=0)

        midpoint_omega = rot_matrix_to_vec( rot_matrix[1:, ...] @ rot_matrix[:-1, ...].swapdims(-1,-2) ) / self.dt

        # reverse of averaging midpoint to get real value
        prestart_omega = 2*start_omega - midpoint_omega[0,None,:]
        postend_omega  = 2*end_omega   - midpoint_omega[-1,None,:]

        midpoint_omega = torch.cat( [ prestart_omega, midpoint_omega, postend_omega], dim=0)
        prev_omega = midpoint_omega[:-1, :]
        next_omega = midpoint_omega[1:, :]

        current_omega = (next_omega + prev_omega)/2
        angular_accel = (next_omega - prev_omega)/self.dt

        # S, 3    3,3      S, 3, 1
        torques = (self.J @ angular_accel[...,None])[...,0]
        actions =  torch.cat([ accel_mag*self.mass, torques ], dim=-1)

        return current_pos, current_vel, current_accel, rot_matrix, current_omega, angular_accel, actions

    def get_full_states(self) -> TensorType["states", 18]:
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()
        return torch.cat( [pos, vel, rot_matrix.reshape(-1, 9), omega], dim=-1 )

    def get_actions(self) -> TensorType["states", 4]:
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()
        return actions

    def get_next_action(self) -> TensorType[4]:
        actions = self.get_actions()
        # fz, tx, ty, tz
        return 2*actions[0, :]

    @typechecked
    def body_to_world(self, points: TensorType["batch", 3]) -> TensorType["states", "batch", 3]:
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()

        # S, 3, P    =    S,3,3       3,P       S, 3, _
        world_points =  rot_matrix @ points.T + pos[..., None]
        return world_points.swapdims(-1,-2)

    def get_state_cost(self) -> TensorType["states"]:
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()

        fz = actions[:, 0]
        torques = torch.norm(actions[:, 1:], dim=-1)

        # multiplied by distance to prevent it from just speed tunnelling
        distance = torch.sum( vel**2 + 1e-5, dim = -1)**0.5
        density = self.nerf( self.body_to_world(self.robot_body) )**2
        colision_prob = torch.mean( density, dim = -1) * distance

        if self.epoch < self.fade_out_epoch:
            t = torch.linspace(0,1, colision_prob.shape[0])
            position = self.epoch/self.fade_out_epoch
            mask = torch.sigmoid(self.fade_out_sharpness * (position - t))
            colision_prob = colision_prob * mask

        #dynamics residual loss - make sure acceleration point in body frame z axis
        start_body_frame_accel = rot_matrix[0 ,:,:].T @ accel[0 ,:]
        s_residue_angle = torch.atan2( (start_body_frame_accel[0]**2 + start_body_frame_accel[1]**2)**0.5, start_body_frame_accel[2])



        end_body_frame_accel   = rot_matrix[-1 ,:,:].T @ accel[-1 ,:]
        e_residue_angle = torch.atan2( (end_body_frame_accel[0]**2 + end_body_frame_accel[1]**2)**0.5, end_body_frame_accel[2])

        # dynamics_residual = torch.norm(start_body_frame_accel[:2])**2  + \
        #                     torch.norm(end_body_frame_accel[:2])**2 

        dynamics_residual = s_residue_angle**2 + e_residue_angle**2


        #PARAM cost function shaping
        return 1000*fz**2 + 0.01*torques**4 + colision_prob * 1e6, colision_prob*1e6, 1e6 * dynamics_residual

    def total_cost(self):
        total_cost, colision_loss, dynamics_residual = self.get_state_cost()
        print("dynamics_residual", dynamics_residual)
        return torch.mean(total_cost) + dynamics_residual

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
    def update_state(self, measured_state: TensorType[18]):
        self.start_state = measured_state
        self.states = self.states[1:, :].detach().requires_grad_(True)


    def plot(self, quadplot):
        quadplot.trajectory( self, "g" )
        ax = quadplot.ax_graph

        pos, vel, accel, _, omega, _, actions = self.calc_everything()
        actions = actions.detach().numpy()
        pos = pos.detach().numpy()
        vel = vel.detach().numpy()
        omega = omega.detach().numpy()

        ax.plot(actions[...,0], label="fz")
        ax.plot(actions[...,1], label="tx")
        ax.plot(actions[...,2], label="ty")
        ax.plot(actions[...,3], label="tz")

        ax.plot(pos[...,0], label="px")
        # ax.plot(pos[...,1], label="py")
        # ax.plot(pos[...,2], label="pz")

        ax.plot(vel[...,0], label="vx")
        # ax.plot(vel[...,1], label="vy")
        ax.plot(vel[...,2], label="vz")

        # ax.plot(omega[...,0], label="omx")
        ax.plot(omega[...,1], label="omy")
        # ax.plot(omega[...,2], label="omz")

        ax_right = quadplot.ax_graph_right

        total_cost, colision_loss, dynamics_residual = self.get_state_cost()
        ax_right.plot(total_cost.detach().numpy(), 'black', label="cost")
        ax_right.plot(colision_loss.detach().numpy(), 'cyan', label="colision")
        ax.legend()

    def save_poses(self, filename):
        positions, _, _, rot_matrix, _, _, _ = self.calc_everything()
        with open(filename,"w+") as f:
            for pos, rot in zip(positions, rot_matrix):
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
    end_pos   = torch.tensor([-1 , 0.7, 0.35])
    # start_pos = torch.tensor([-1, 0, 0.2])
    # end_pos   = torch.tensor([ 1, 0, 0.5])

    start_state = torch.cat( [start_pos, torch.tensor([0,0,0]), torch.eye(3).reshape(-1), torch.zeros(3)], dim=0 )
    end_state   = torch.cat( [end_pos,   torch.zeros(3), torch.eye(3).reshape(-1), torch.zeros(3)], dim=0 )

    renderer = get_manual_nerf("empty")

    cfg = {"T_final": 2,
            "steps": 20,
            "lr": 0.001,
            "epochs_init": 2500,
            "fade_out_epoch": 500,
            "fade_out_sharpness": 10,
            "epochs_update": 500,
            }

    traj = System(renderer, start_state, end_state, cfg)
    traj.learn_init()


    sim = Simulator(start_state)
    sim.dt = traj.dt

    save = Simulator(start_state)
    save.copy_states(traj.get_full_states())

    # quadplot = QuadPlot()
    # traj.plot(quadplot)
    # quadplot.show()


    if True:
        for step in range(cfg['steps']):
            action = traj.get_next_action()
            # action = traj.get_actions()[step,:]
            print(action)

            sim.advance(action)
            # sim.advance_smooth(action, 10)

            # randomness = torch.normal(mean= 0, std=torch.tensor([0.02]*18) )
            # measured_state = traj.get_full_states()[1,:].detach()
            # sim.add_state(measured_state)
            # measured_state += randomness

            measured_state = sim.get_current_state().detach()
            traj.update_state(measured_state)


            traj.learn_update()
            # # traj.save_poses(???)

            print("sim step", step)
            if step % 5 !=0 or step == 0:
                continue

            quadplot = QuadPlot()
            traj.plot(quadplot)
            quadplot.trajectory( sim, "r" )
            quadplot.trajectory( save, "b", show_cloud=False )
            quadplot.show()



    #PARAM file to save the trajectory
    # traj.save_poses("paths/playground_testing.json")
    # traj.plot()


if __name__ == "__main__":
    main()
