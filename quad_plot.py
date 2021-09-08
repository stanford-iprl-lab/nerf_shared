import torch
from torch._C import device
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import json
import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

from load_nerf import get_nerf

from quad_helpers import Simulator, QuadPlot
from quad_helpers import rot_matrix_to_vec, vec_to_rot_matrix, next_rotation
from quad_helpers import astar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, renderer, start_state: TensorType[18], end_state: TensorType[18], cfg):
        self.renderer = renderer
        self.nerf = renderer.get_density

        self.cfg                = cfg
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
        self.initial_accel = torch.tensor([10.0,10.0]).requires_grad_(True)

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

    def a_star_init(self):
        side = 100 #PARAM grid size
        linspace = torch.linspace(-1,1, side) #PARAM extends of the thing

        # side, side, side, 3
        coods = torch.stack( torch.meshgrid( linspace, linspace, linspace ), dim=-1)

        kernel_size = 5 # 100/5 = 20. scene size of 2 gives a box size of 2/20 = 0.1 = drone size
        output = self.nerf(coods)
        maxpool = torch.nn.MaxPool3d(kernel_size = kernel_size)
        #PARAM cut off such that neural network outputs zero (pre shifted sigmoid)

        # 20, 20, 20
        occupied = maxpool(output[None,None,...])[0,0,...] > 0.33

        grid_size = side//kernel_size

        #convert to index cooredinates
        start_grid_float = grid_size*(self.start_state[:3] + 1)/2
        end_grid_float   = grid_size*(self.end_state  [:3] + 1)/2
        start = tuple(int(start_grid_float[i]) for i in range(3) )
        end =   tuple(int(end_grid_float[i]  ) for i in range(3) )

        print(start, end)
        path = astar(occupied, start, end)

        # convert from index cooredinates
        squares =  2* (torch.tensor( path, dtype=torch.float)/grid_size) -1

        #adding way
        states = torch.cat( [squares, torch.zeros( (squares.shape[0], 1) ) ], dim=-1)

        #prevents weird zero derivative issues
        randomness = torch.normal(mean= 0, std=0.001*torch.ones(states.shape) )
        states += randomness

        # smooth path (diagram of which states are averaged)
        # 1 2 3 4 5 6 7
        # 1 1 2 3 4 5 6
        # 2 3 4 5 6 7 7
        prev_smooth = torch.cat([states[0,None, :], states[:-1,:]],        dim=0)
        next_smooth = torch.cat([states[1:,:],      states[-1,None, :], ], dim=0)
        states = (prev_smooth + next_smooth + states)/3

        self.states = states.clone().detach().requires_grad_(True)

    def params(self):
        return [self.initial_accel, self.states]

    @typechecked
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

        next_R = next_rotation(start_R, start_omega, self.dt)

        # start, next, decision_states, last, end

        start_accel = start_R @ torch.tensor([0,0,1.0]) * self.initial_accel[0] + self.g
        next_accel = next_R @ torch.tensor([0,0,1.0]) * self.initial_accel[1] + self.g

        next_vel = start_v + start_accel * self.dt
        after_next_vel = next_vel + next_accel * self.dt

        next_pos = start_pos + start_v * self.dt
        after_next_pos = next_pos + next_vel * self.dt
        after2_next_pos = after_next_pos + after_next_vel * self.dt
    
        # position 2 and 3 are unused - but the atached roations are
        current_pos = torch.cat( [start_pos, next_pos, after_next_pos, after2_next_pos, self.states[2:, :3], end_pos], dim=0)

        prev_pos = current_pos[:-1, :]
        next_pos = current_pos[1: , :]

        current_vel = (next_pos - prev_pos)/self.dt
        current_vel = torch.cat( [ current_vel, end_v], dim=0)

        prev_vel = current_vel[:-1, :]
        next_vel = current_vel[1: , :]

        current_accel = (next_vel - prev_vel)/self.dt - self.g

        # duplicate last accceleration - its not actaully used for anything (there is no action at last state)
        current_accel = torch.cat( [ current_accel, current_accel[-1,None,:] ], dim=0)

        accel_mag     = torch.norm(current_accel, dim=-1, keepdim=True)

        # needs to be pointing in direction of acceleration
        z_axis_body = current_accel/accel_mag

        # remove states with rotations already constrained
        z_axis_body = z_axis_body[2:-1, :]

        z_angle = self.states[:,3]

        in_plane_heading = torch.stack( [torch.sin(z_angle), -torch.cos(z_angle), torch.zeros_like(z_angle)], dim=-1)

        x_axis_body = torch.cross(z_axis_body, in_plane_heading, dim=-1)
        x_axis_body = x_axis_body/torch.norm(x_axis_body, dim=-1, keepdim=True)
        y_axis_body = torch.cross(z_axis_body, x_axis_body, dim=-1)

        # S, 3, 3 # assembled manually from basis vectors
        rot_matrix = torch.stack( [x_axis_body, y_axis_body, z_axis_body], dim=-1)

        rot_matrix = torch.cat( [start_R, next_R, rot_matrix, end_R], dim=0)

        current_omega = rot_matrix_to_vec( rot_matrix[1:, ...] @ rot_matrix[:-1, ...].swapdims(-1,-2) ) / self.dt
        current_omega = torch.cat( [ current_omega, end_omega], dim=0)

        prev_omega = current_omega[:-1, :]
        next_omega = current_omega[1:, :]

        angular_accel = (next_omega - prev_omega)/self.dt
        # duplicate last ang_accceleration - its not actaully used for anything (there is no action at last state)
        angular_accel = torch.cat( [ angular_accel, angular_accel[-1,None,:] ], dim=0)

        # S, 3    3,3      S, 3, 1
        torques = (self.J @ angular_accel[...,None])[...,0]
        actions =  torch.cat([ accel_mag*self.mass, torques ], dim=-1)

        return current_pos, current_vel, current_accel, rot_matrix, current_omega, angular_accel, actions

    def get_full_states(self) -> TensorType["states", 18]:
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()
        return torch.cat( [pos, vel, rot_matrix.reshape(-1, 9), omega], dim=-1 )

    def get_actions(self) -> TensorType["states", 4]:
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()

        if not torch.allclose( actions[:2, 0], self.initial_accel ):
            print(actions)
            print(self.initial_accel)
        return actions

    def get_next_action(self) -> TensorType[4]:
        actions = self.get_actions()
        # fz, tx, ty, tz
        return actions[0, :]

    @typechecked
    def body_to_world(self, points: TensorType["batch", 3]) -> TensorType["states", "batch", 3]:
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()

        # S, 3, P    =    S,3,3       3,P       S, 3, _
        world_points =  rot_matrix @ points.T + pos[..., None]
        return world_points.swapdims(-1,-2)

    def get_state_cost(self) -> TensorType["states"]:
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()

        fz = actions[:, 0].to(device)
        torques = torch.norm(actions[:, 1:], dim=-1).to(device)

        # multiplied by distance to prevent it from just speed tunnelling
        distance = (torch.sum( vel**2 + 1e-5, dim = -1)**0.5).to(device)
        density = (self.nerf( self.body_to_world(self.robot_body) )**2).to(device)
        colision_prob = torch.mean( density, dim = -1) * distance

        if self.epoch < self.fade_out_epoch:
            t = torch.linspace(0,1, colision_prob.shape[0])
            position = self.epoch/self.fade_out_epoch
            mask = torch.sigmoid(self.fade_out_sharpness * (position - t)).to(device)
            colision_prob = colision_prob * mask

        ##dynamics residual loss - make sure acceleration point in body frame z axis
        ## S, 3, _     =   S, 3, 3  @ S, 3, _
        #body_frame_accel   = ( rot_matrix.swapdims(-1,-2) @ accel[:,:,None]) [:,:,0]
        ## pick out the ones we want to constrain (the rest are already constrained
        #residue_angle = torch.atan2( torch.norm(body_frame_accel[:,:2], dim =-1 ) , body_frame_accel[:,2])
        #print("residue_angle", residue_angle)


        #PARAM cost function shaping
        return 1000*fz**2 + 0.01*torques**4 + colision_prob * 1e6, colision_prob*1e6

    def total_cost(self):
        total_cost, colision_loss  = self.get_state_cost()
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
                    self.save_poses("paths/"+str(it//save_step)+"_testing.json", loss.clone().cpu().detach().numpy().tolist())

        except KeyboardInterrupt:
            print("finishing early")

    def learn_update(self, iteration):
        opt = torch.optim.Adam(self.params(), lr=self.lr)

        for it in range(self.epochs_update):
            opt.zero_grad()
            self.epoch = it
            loss = self.total_cost()
            print(it, loss)
            loss.backward()
            opt.step()
            # it += 1

            # if (it > self.epochs_update and self.max_residual < 1e-3):
            #     break

            save_step = 50
            if it%save_step == 0:
                self.save_poses("paths/"+str(it//save_step)+f'update{iteration}' + '.json', loss.clone().cpu().detach().numpy().tolist())

    @typechecked
    def update_state(self, measured_state: TensorType[18]):
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()

        self.start_state = measured_state
        self.states = self.states[1:, :].detach().requires_grad_(True)
        self.initial_accel = actions[1:3, 0].detach().requires_grad_(True)
        # print(self.initial_accel.shape)


    def plot(self, quadplot):
        quadplot.trajectory( self, "g" )
        ax = quadplot.ax_graph

        pos, vel, accel, _, omega, _, actions = self.calc_everything()
        actions = actions.cpu().detach().numpy()
        pos = pos.cpu().detach().numpy()
        vel = vel.cpu().detach().numpy()
        omega = omega.cpu().detach().numpy()

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

        total_cost, colision_loss = self.get_state_cost()
        ax_right.plot(total_cost.detach().numpy(), 'black', label="cost")
        ax_right.plot(colision_loss.detach().numpy(), 'cyan', label="colision")
        ax.legend()

    def save_poses(self, filename, loss):
        positions, _, _, rot_matrix, _, _, _ = self.calc_everything()
        poses = []
        pose_dict = {}
        with open(filename,"w+") as f:
            for pos, rot in zip(positions, rot_matrix):
                pose = np.zeros((4,4))
                pose[:3, :3] = rot.cpu().detach().numpy()
                pose[:3, 3]  = pos.cpu().detach().numpy()
                pose[3,3] = 1

                poses.append(pose.tolist())
            pose_dict["poses"] = poses
            pose_dict["loss"] = loss
            json.dump(pose_dict, f)

    def save_progress(self, filename):
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass

        if hasattr(self.renderer, "config_filename"):
            config_filename = self.renderer.config_filename
        else:
            config_filename = None

        to_save = {"cfg": self.cfg,
                    "start_state": self.start_state,
                    "end_state": self.end_state,
                    "states": self.states,
                    "initial_accel":self.initial_accel,
                    "config_filename": config_filename,
                    }
        torch.save(to_save, filename)

    @classmethod
    def load_progress(cls, filename, renderer=None):
        # a note about loading: it won't load the optimiser learned step sizes
        # so the first couple gradient steps can be quite bad

        loaded_dict = torch.load(filename)
        print(loaded_dict)

        if renderer == None:
            assert loaded_dict['config_filename'] is not None
            renderer = load_nerf(loaded_dict['config_filename'])

        obj = cls(renderer, loaded_dict['start_state'], loaded_dict['end_state'], loaded_dict['cfg'])
        obj.states = loaded_dict['states'].requires_grad_(True)
        obj.initial_accel = loaded_dict['initial_accel'].requires_grad_(True)

        return obj

def main():

    # violin - astar
    # renderer = get_nerf('configs/violin.txt')
    # start_state = torch.tensor([0.44, -0.23, 0.2, 0])
    # end_state = torch.tensor([-0.58, 0.66, 0.15, 0])


    #playground
    filename = "playground.plan"
    renderer = get_nerf('configs/playground.txt')

    # 2d across
    # start_pos = torch.tensor([-0.0, -0.45, 0.12])
    # end_pos = torch.tensor([0.02, 0.58, 0.65])

    # under slide
    start_pos = torch.tensor([-0.3, -0.27, 0.06])
    end_pos = torch.tensor([0.02, 0.58, 0.65])

    #stonehenge
    # renderer = get_nerf('configs/stonehenge.txt')
    # start_state = torch.tensor([-0.06, -0.79, 0.2, 0])
    # end_state = torch.tensor([-0.46, 0.55, 0.16, 0])

    # start_pos = torch.tensor([-0.05,-0.9, 0.2])
    # end_pos   = torch.tensor([-1 , 0.7, 0.35])
    # start_pos = torch.tensor([-1, 0, 0.2])
    # end_pos   = torch.tensor([ 1, 0, 0.5])


    start_R = vec_to_rot_matrix( torch.tensor([0.0,0.0,0]))

    start_state = torch.cat( [start_pos, torch.tensor([0,0,0]), start_R.reshape(-1), torch.zeros(3)], dim=0 )
    end_state   = torch.cat( [end_pos,   torch.zeros(3), torch.eye(3).reshape(-1), torch.zeros(3)], dim=0 )

    filename = "line.plan"
    renderer = get_manual_nerf("empty")
    # renderer = get_manual_nerf("cylinder")

    cfg = {"T_final": 2,
            "steps": 20,
            "lr": 0.01,
            "epochs_init": 2500,
            "fade_out_epoch": 0,
            "fade_out_sharpness": 10,
            "epochs_update": 250,
            }

    # traj = System(renderer, start_state, end_state, cfg)
    traj = System.load_progress(filename, renderer)
    traj.epochs_update = 250 #change depending on noise

    # traj.a_star_init()

    # quadplot = QuadPlot()
    # traj.plot(quadplot)
    # quadplot.show()

    # traj.learn_init()

    quadplot = QuadPlot()
    traj.plot(quadplot)
    quadplot.show()

    # traj.save_progress(filename)

    save = Simulator(start_state)
    save.copy_states(traj.get_full_states())

    if True: # for mpc control
        sim = Simulator(start_state)
        sim.dt = traj.dt #Sim time step changes best on number of steps

        for step in range(cfg['steps']):
            action = traj.get_next_action().clone().detach()
            print(action)

            state_noise = torch.normal(mean= 0, std=torch.tensor( [0.01]*3 + [0.01]*3 + [0]*9 + [0.005]*3 ))
            # state_noise[3] += 0.0 #crosswind

            # sim.advance(action) # no noise
            sim.advance(action, state_noise) #add noise
            measured_state = sim.get_current_state().clone().detach()

            measurement_noise = torch.normal(mean= 0, std=torch.tensor( [0.01]*3 + [0.02]*3 + [0]*9 + [0.005]*3 ))
            measured_state += measurement_noise
            traj.update_state(measured_state) 

            traj.learn_update()

            print("sim step", step)
            if step % 5 !=0 or step == 0:
                continue

            quadplot = QuadPlot()
            traj.plot(quadplot)
            quadplot.trajectory( sim, "r" )
            quadplot.trajectory( save, "b", show_cloud=False )
            quadplot.show()

    if False:
        sim = Simulator(start_state)
        sim.dt = traj.dt #Sim time step changes best on number of steps

        for step in range(cfg['steps']):
            # for open loop control
            action = traj.get_actions()[step,:].detach()
            print(action)
            sim.advance(action)

    quadplot = QuadPlot()
    traj.plot(quadplot)
    quadplot.trajectory( sim, "r" )
    quadplot.trajectory( save, "b", show_cloud=False )
    quadplot.show()

if __name__ == "__main__":
    main()