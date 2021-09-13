import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import json
import shutil
import pathlib

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

from load_nerf import get_nerf
import math

torch.manual_seed(0)
np.random.seed(0)

from quad_helpers import Simulator, QuadPlot
from quad_helpers import rot_matrix_to_vec, vec_to_rot_matrix, next_rotation
from quad_helpers import astar

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
    def __init__(self, renderer, start_state: TensorType[3], end_state: TensorType[3], cfg):
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

        self.CHURCH = False

        self.dt = self.T_final / self.steps

        self.mass = 1

        self.start_state = start_state
        self.end_state   = end_state

        slider = torch.linspace(0, 1, self.steps)[1:-1, None]

        states = (1-slider) * start_state + \
                slider  * end_state

        self.states = states.clone().detach().requires_grad_(True)

        #PARAM this sets the shape of the robot body point cloud
        body = torch.stack( torch.meshgrid( torch.linspace(-0.075, 0.075, 30),
                                            torch.linspace(-0.0375, 0.0375, 10),
                                            torch.linspace(-0.02, 0.02,  4)), dim=-1) / 2

        self.z_height = 0.52

        self.robot_body = body.reshape(-1, 3)
        # self.robot_body = torch.zeros(1,3)

        self.epoch = 0


    def a_star_init(self, kernel_size = 5):
        side = 100 #PARAM grid size

        if self.CHURCH:
            x_linspace = torch.linspace(-2,-1, side)
            y_linspace = torch.linspace(-1.2,-0.2, side)
            z_linspace = torch.linspace(0.4,1.4, side)

            coods = torch.stack( torch.meshgrid( x_linspace, y_linspace, z_linspace ), dim=-1)
            # kernel_size = 2 # 100/5 = 20. scene size of 2 gives a box size of 2/20 = 0.1 = drone size
        else:
            linspace = torch.linspace(-1,1, side) #PARAM extends of the thing
            # side, side, side, 3
            coods = torch.stack( torch.meshgrid( linspace, linspace, linspace ), dim=-1)
            # kernel_size = 5 # 100/5 = 20. scene size of 2 gives a box size of 2/20 = 0.1 = drone size
            # kernel_size = 4


        min_value = coods[0,0,0,:]
        side_length = coods[-1,-1,-1,:] - coods[0,0,0,:]
        print(min_value)
        print(side_length)

        output = self.nerf(coods)
        maxpool = torch.nn.MaxPool3d(kernel_size = kernel_size)
        #PARAM cut off such that neural network outputs zero (pre shifted sigmoid)

        # 20, 20, 20
        occupied = maxpool(output[None,None,...])[0,0,...] > 0.33

        grid_size = side//kernel_size


        #convert to index cooredinates
        start_grid_float = grid_size*(self.start_state[:3] - min_value)/side_length
        end_grid_float   = grid_size*(self.end_state  [:3] - min_value)/side_length
        start = tuple(int(start_grid_float[i]) for i in range(3) )
        end =   tuple(int(end_grid_float[i]  ) for i in range(3) )

        print(start, end)
        path = astar(occupied, start, end)
        print(path)

        # convert from index cooredinates
        squares =  side_length * (torch.tensor(path, dtype=torch.float)/grid_size) + min_value
        print(squares)

        #adding yaw
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
        return [self.states]

    @typechecked
    def calc_everything(self) -> (
            TensorType["states", 3], #pos
            TensorType["states", 3], #vel
            None, #accel
            TensorType["states", 3,3], #rot_matrix
            TensorType["states", 3], #omega
            None, #angualr_accel
            TensorType["states", 3], #actions
        ):

        start_pos   = self.start_state[None, :]
        end_pos   = self.end_state[None, :]

        all_states = torch.cat( [start_pos, self.states, end_pos], dim=0)
        current_pos = torch.cat( [ all_states[:,:2], self.z_height * torch.ones( (all_states.shape[0], 1) )], dim=-1 )

        prev_pos = current_pos[:-1, :]
        next_pos = current_pos[1: , :]

        current_vel = (next_pos - prev_pos)/self.dt
        current_vel = torch.cat( [ current_vel, torch.zeros_like(current_vel[-1,None,:])], dim=0)


        z_angle = all_states[:,2]
        z_axis_body = torch.tensor([0,0,1.0]).expand( all_states.shape[0], 3)
        in_plane_heading = torch.stack( [torch.sin(z_angle), -torch.cos(z_angle), torch.zeros_like(z_angle)], dim=-1)

        x_axis_body = torch.cross(z_axis_body, in_plane_heading, dim=-1)
        x_axis_body = x_axis_body/torch.norm(x_axis_body, dim=-1, keepdim=True)
        y_axis_body = torch.cross(z_axis_body, x_axis_body, dim=-1)

        rot_matrix = torch.stack( [x_axis_body, y_axis_body, z_axis_body], dim=-1)

        current_omega = rot_matrix_to_vec( rot_matrix[1:, ...] @ rot_matrix[:-1, ...].swapdims(-1,-2) ) / self.dt
        current_omega = torch.cat( [ current_omega, torch.zeros_like(current_omega[-1,None,:])], dim=0)


        # print(rot_matrix[:,:2,:2].swapdims(-1,-2).shape)
        # print(current_vel[:,:,None].shape)
                    #S, 2, 2                     S, 2, _
        lin_vel = (rot_matrix[:,:2,:2].swapdims(-1,-2)  @ (current_vel[:,:2,None] ))[:,:,0]

        actions =  torch.cat([ lin_vel, current_omega[:,2,None] ], dim=-1)

        return current_pos, current_vel, None, rot_matrix, current_omega, None, actions

    def get_full_states(self) -> TensorType["states", 18]:
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()
        return torch.cat( [pos, vel, rot_matrix.reshape(-1, 9), omega], dim=-1 )

    def get_actions(self) -> TensorType["states", 3]:
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()
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

        x = torch.exp(actions[:, 0] ** 4)
        y = torch.exp(actions[:, 1] ** 4)
        a = actions[:, 2] ** 2

        # S, B, 3  =  S, _, 3 +      _, B, 3   X    S, _,  3
        B_body, B_omega = torch.broadcast_tensors(self.robot_body, omega[:,None,:])
        point_vels = vel[:,None,:] + torch.cross(B_body, B_omega, dim=-1)
        # S, B
        distance = torch.sum( vel**2 + 1e-5, dim = -1)**0.5
        # S, B
        density = self.nerf( self.body_to_world(self.robot_body) )**2

        #weight points closer to body higher
        point_weight = 1/(torch.norm(self.robot_body, dim=-1) + 0.1)

        # multiplied by distance to prevent it from just speed tunnelling
        # S =   S,B * S,_
        colision_prob = torch.mean(point_weight * density * distance[:,None], dim = -1) 

        if self.epoch < self.fade_out_epoch:
            t = torch.linspace(0,1, colision_prob.shape[0])
            position = self.epoch/self.fade_out_epoch
            mask = torch.sigmoid(self.fade_out_sharpness * (position - t))
            colision_prob = colision_prob * mask

        colision_prob = colision_prob **2

        #PARAM cost function shaping
        return 100*y + 1*a + 100*x +  colision_prob * 1e3, colision_prob*1e3

    def total_cost(self):
        total_cost, colision_loss  = self.get_state_cost()
        return torch.mean(total_cost)

    def learn_init(self):
        opt = torch.optim.Adam(self.params(), lr=self.lr)
        # opt = torch.optim.SGD(self.params(), lr=0.002, momentum = 0.999)

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
                    if hasattr(self, "basefolder"):
                        # self.save_poses(self.basefolder / "train_poses" / (str(it//save_step)+".json"))
                        self.save_data(self.basefolder / "train" / (str(it//save_step)+".json"))
                    else:
                        print("WANRING: data not saved!")


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
            # it += 1

            # if (it > self.epochs_update and self.max_residual < 1e-3):
            #     break

            # save_step = 50
            # if it%save_step == 0:
        # self.save_poses("paths/"+str(it//save_step)+"_testing.json")

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
        actions = actions.detach().numpy()
        pos = pos.detach().numpy()
        vel = vel.detach().numpy()
        omega = omega.detach().numpy()

        ax.plot(actions[...,0], label="fz")
        ax.plot(actions[...,1], label="tx")
        ax.plot(actions[...,2], label="ty")
        # ax.plot(actions[...,3], label="tz")

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

    def save_data(self, filename):
        positions, vel, _, rot_matrix, omega, _, actions = self.calc_everything()
        total_cost, colision_loss  = self.get_state_cost()

        poses = torch.zeros((positions.shape[0], 4,4))
        poses[:, :3, :3] = rot_matrix
        poses[:, :3, 3]  = positions
        poses[:, 3,3] = 1

        full_states = self.get_full_states()

        output = {"colision_loss": colision_loss.detach().numpy().tolist(),
                  "poses": poses.detach().numpy().tolist(),
                  "actions": actions.detach().numpy().tolist(),
                  "total_cost": total_cost.detach().numpy().tolist(),
                  "full_states": full_states.detach().numpy().tolist(),
                  }

        with open(filename,"w+") as f:
            json.dump( output,  f)

    def save_progress(self, filename):
        if hasattr(self.renderer, "config_filename"):
            config_filename = self.renderer.config_filename
        else:
            config_filename = None

        to_save = {"cfg": self.cfg,
                    "start_state": self.start_state,
                    "end_state": self.end_state,
                    "states": self.states,
                    # "initial_accel":self.initial_accel,
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
        # obj.initial_accel = loaded_dict['initial_accel'].requires_grad_(True)

        return obj

def main():

    church = False


    renderer = get_nerf('configs/church.txt')
    # experiment_name = "church_omni2"
    # start_state = torch.tensor([-1.1, -0.55,  -math.pi/4 + 10e-2 ])
    # end_state = torch.tensor([-1.53, -0.83, -math.pi/4 + 10e-2])

    experiment_name = "church_omni4"
    start_state = torch.tensor([-1.1, -0.47,  -math.pi/4 + 10e-2 ])
    end_state = torch.tensor([-1.53, -0.83, -math.pi/4 + 10e-2])

    church = True

    cfg = {"T_final": 2,
            "steps": 20,
            "lr": 0.005,
            "epochs_init": 2500,
            "fade_out_epoch": 0,
            "fade_out_sharpness": 10,
            "epochs_update": 250,
            }


    # experiment_name = "test" 
    # filename = "line.plan"
    # renderer = get_manual_nerf("empty")
    # renderer = get_manual_nerf("cylinder")

    # start_state = torch.cat( [start_pos, torch.tensor([0,0,0]), start_R.reshape(-1), torch.zeros(3)], dim=0 )
    # end_state   = torch.cat( [end_pos,   torch.zeros(3), end_R.reshape(-1), torch.zeros(3)], dim=0 )

    LOAD = False

    basefolder = "experiments" / pathlib.Path(experiment_name)

    if not LOAD:
        if basefolder.exists():
            print(basefolder, "already exists!")
            if input("Clear it before continuing? [y/N]:").lower() == "y":
                shutil.rmtree(basefolder)
        basefolder.mkdir()
        (basefolder / "train").mkdir()

    print("created", basefolder)


    if LOAD:
        traj = System.load_progress(basefolder / "trajectory.pt", renderer); traj.epochs_update = cfg['epochs_update'] #change depending on noise
    else:
        traj = System(renderer, start_state, end_state, cfg)

    traj.basefolder = basefolder
    traj.CHURCH = church

    # quadplot = QuadPlot()
    # traj.plot(quadplot)
    # quadplot.show()

    if not LOAD:
        traj.learn_init()
        traj.save_progress(basefolder / "trajectory.pt")

    quadplot = QuadPlot()
    traj.plot(quadplot)
    quadplot.show()


    # save = Simulator(start_state)
    # save.copy_states(traj.get_full_states())


    # quadplot = QuadPlot()
    # traj.plot(quadplot)
    # quadplot.trajectory( sim, "r" )
    # quadplot.trajectory( save, "b", show_cloud=False )
    # quadplot.show()



if __name__ == "__main__":
    main()
