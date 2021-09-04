import torch
import torch.nn as nn
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
import numpy as np
import math

import json

import time

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

torch.manual_seed(0)
np.random.seed(0)

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


def vec2ss_matrix(vector):  # vector to skewsym. matrix

    ss_matrix = torch.zeros((3,3))
    ss_matrix[0, 1] = -vector[2]
    ss_matrix[0, 2] = vector[1]
    ss_matrix[1, 0] = vector[2]
    ss_matrix[1, 2] = -vector[0]
    ss_matrix[2, 0] = -vector[1]
    ss_matrix[2, 1] = vector[0]

    return ss_matrix

class path(nn.Module):
    def __init__(self, init_actions, steps, mass, g, I, dt):
        super(path, self).__init__()
        self.steps = steps
        self.actions = nn.Parameter(torch.tensor(init_actions))
        self.mass = mass
        self.g = g
        self.I = I
        self.invI = torch.inverse(self.I)
        self.dt = dt

    def dynamics(self, state_now, action):
        #State is 18 dimensional [pos(3), vel(3), R (9), omega(3)] where pos, vel are in the world frame, R is the rotation from points in the body frame to world frame
        # and omega are angular rates in the body frame

        #Actions are [total thrust, torque x, torque y, torque z]
        fz = action[0]
        tau = action[1:]

        state = state_now.clone()

        #Define state vector
        pos = state[0:3]
        v   = state[3:6]
        R_flat = state[6:15]
        R = R_flat.reshape((3, 3))
        omega = state[15:]

        # The acceleration
        sum_action = torch.zeros(3)
        sum_action[2] = fz

        grav = torch.tensor([0,0,-self.mass*self.g])

        dv = (grav + R @ sum_action)/self.mass

        # The angular accelerations
        domega = self.invI @ (tau - torch.cross(omega, self.I @ omega))

        # Propagate rotation matrix using exponential map of the angle displacements
        angle = omega*self.dt
        theta = torch.norm(angle, p=2)
        if theta == 0:
            exp_i = torch.eye(3)
        else:
            angle_norm = angle/theta
            K = vec2ss_matrix(angle_norm)

            exp_i = torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.matmul(K, K)

        next_R = R @ exp_i

        next_pos = pos + v * self.dt
        next_vel = v + dv * self.dt

        next_rot = next_R.reshape(-1)

        next_omega = omega + domega * self.dt

        return torch.hstack((next_pos, next_vel, next_rot, next_omega))

    def forward(self, x):
        states = torch.zeros((self.steps, 18))

        for iter in range(self.steps):
            if iter == 0:
                states[0] = self.dynamics(x, self.actions[0])
            else:
                states[iter] = self.dynamics(states[iter-1], self.actions[iter])

        actions = self.actions
        return states, actions

class Planner:
    def __init__(self, renderer, start_pose, end_pose,
                        cfg):
        self.nerf = renderer.get_density

        self.T_final            = cfg['T_final']
        self.steps              = cfg['steps']
        self.lr                 = cfg['lr']
        self.epochs_init        = cfg['epochs_init']
        self.epochs_update      = cfg['epochs_update']
        self.fade_out_epoch     = cfg['fade_out_epoch']
        self.fade_out_sharpness = cfg['fade_out_sharpness']

        #Dimensions of body
        self.x_length = cfg['x_length']
        self.y_length =cfg['y_length']
        self.z_length = cfg['z_length']
        self.cloud_density = cfg['cloud_density']
        self.cloud_density_per_axis = math.floor(self.cloud_density**(1/3))

        self.dt = self.T_final / self.steps

        self.start_pose = start_pose
        self.end_pose = end_pose

        #TODO: CHANGE PENALTY TERMS TO BE IN CONFIG

        self.end_state_penalty = 1000
        self.thrust_penalty = 1
        self.torque_penalty = .1
        self.density_penalty = 1e6

        #PARAM this sets the shape of the robot body point cloud
        body = torch.stack( torch.meshgrid( torch.linspace(-self.x_length/2, self.x_length/2, self.cloud_density_per_axis),
                                            torch.linspace(-self.y_length/2, self.y_length/2, self.cloud_density_per_axis),
                                            torch.linspace(-self.z_length/2, self.z_length/2, self.cloud_density_per_axis)), dim=-1)
        self.robot_body = body.reshape(-1, 3)

        print('Body', self.robot_body.shape, self.robot_body)

    def dynamics(self, state_now, action):
            #State is 18 dimensional [pos(3), vel(3), R (9), omega(3)] where pos, vel are in the world frame, R is the rotation from points in the body frame to world frame
            # and omega are angular rates in the body frame

            #Actions are [total thrust, torque x, torque y, torque z]
            fz = action[0]
            tau = action[1:]

            state = state_now.clone()

            #Define state vector
            pos = state[0:3]
            v   = state[3:6]
            R_flat = state[6:15]
            R = R_flat.reshape((3, 3))
            omega = state[15:]

            # The acceleration
            sum_action = torch.zeros(3)
            sum_action[2] = fz

            grav = torch.tensor([0,0,-self.mass*self.g])

            # The angular accelerations
            domega = self.invI @ (tau - torch.cross(omega, self.I @ omega))

            # Propagate rotation matrix using exponential map of the angle displacements
            angle = omega*self.dt
            theta = torch.norm(angle, p=2)
            if theta == 0:
                exp_i = torch.eye(3)
            else:
                angle_norm = angle/theta
                K = vec2ss_matrix(angle_norm)

                exp_i = torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.matmul(K, K)

            next_R = R @ exp_i

            dv = (grav + next_R @ sum_action)/self.mass

            next_pos = pos + v * self.dt
            next_vel = v + dv * self.dt

            next_rot = next_R.reshape(-1)

            next_omega = omega + domega * self.dt

            return torch.hstack((next_pos, next_vel, next_rot, next_omega))

    def plan_traj(self, state_estimate, init_actions):

        #Initialize path module
        starting_pose = torch.Tensor(state_estimate)
        path_propagation = path(init_actions, self.steps, 1., 10., torch.eye(3), self.dt)
        optimizer = torch.optim.Adam(params=path_propagation.parameters(), lr=self.lr, betas=(0.9, 0.999))

        for it in range(self.epochs_init):
            optimizer.zero_grad()
            t1 = time.time()
            projected_states, actions = path_propagation(starting_pose)

            loss = self.get_loss(projected_states, actions)
            t2 = time.time()
            #print('Propagation', t2 - t1)

            loss.backward()
            optimizer.step()
            t3 = time.time()

            if it % 20 == 0:
                print('Iteration', it)
                print(projected_states[-1])
                print('Loss', loss)

            new_lrate = self.lr * (0.8 ** ((it + 1) / self.epochs_init))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
            #print('Backprop', t3-t2)

        return projected_states, actions

    def get_loss(self, states, actions):
        density_loss = self.get_density_loss(states)
        state_loss = self.get_state_loss(states)
        action_loss = self.get_action_loss(actions)

        loss = density_loss + state_loss + action_loss

        return loss

    def get_density_loss(self, states):

        next_states = torch.vstack((states, self.end_pose))
        prev_states = torch.vstack((self.start_pose, states))

        #Get points of trajectory first

        #points = states[:, :3]
        #densities = self.nerf(points)
        #density_loss = torch.sum(densities)

        points_in_body_frame = self.robot_body

        #Convert to homogenous coordinates
        points_in_body_frame = torch.cat((points_in_body_frame, torch.ones((points_in_body_frame.shape[0], 1))), -1)

        #TODO: Check to see if this gives intended result
        body2worldrot = states[:, 6:15].reshape(-1, 3, 3)
        body2worldtrans = states[:, :3].reshape(-1, 3, 1)
        body2worldpose = torch.cat((body2worldrot, body2worldtrans), -1)

        points_in_world_frame = body2worldpose @ points_in_body_frame.T
        points_in_world_frame = points_in_world_frame.permute(0, 2, 1)

        densities = (self.nerf(points_in_world_frame))**2

        distances = torch.norm(next_states[1:,:] - prev_states[:-1,:], dim=1, p=2)

        colision_prob =  densities * distances[..., None].expand(*densities.shape)
        colision_prob = torch.mean(colision_prob, dim=1)

        '''
                if self.epoch < self.fade_out_epoch:
            t = torch.linspace(0,1, colision_prob.shape[0]).to(device)
            position = self.epoch/self.fade_out_epoch
            mask = torch.sigmoid(self.fade_out_sharpness * (position - t))
            colision_prob = colision_prob * mask
        '''

        density_loss = self.density_penalty*torch.sum(colision_prob)

        return density_loss

    def get_state_loss(self, states):
        #offsets = states - self.end_pose.expand(self.steps, -1)

        #state_loss = torch.sum(torch.norm(offsets, dim=1))

        state_loss = torch.mean((torch.norm(states[1:, ...] - states[:-1, ...], dim=1, p=2))**2)

        state_loss = state_loss +  self.end_state_penalty *  (torch.norm(states[-1,...] - self.end_pose, p=2))**2

        #state_loss = self.end_state_penalty*torch.norm(states[-1] - self.end_pose, p=2)

        return state_loss

    def get_action_loss(self, actions):
        #action_loss = torch.mean(torch.norm(actions, dim=1, p=2))
        action_loss = torch.mean(self.thrust_penalty*actions[:, 0]**2 + self.torque_penalty*(torch.norm(actions[:, 1:], dim=1, p=2))**2)

        return action_loss

renderer = get_manual_nerf("empty")
cfg = {"T_final": 2,
        "steps": 20,
        "lr": 0.1,
        "epochs_init": 500,
        "fade_out_epoch": 500,
        "fade_out_sharpness": 10,
        "epochs_update": 500,
        'x_length': 0.1,
        'y_length': 0.1,
        'z_length': 0.05,
        'cloud_density': 1000
        }

if __name__ == "__main__":
    start_pose = torch.zeros(18)
    end_pose = torch.zeros(18)
    #end_pose[:3] = torch.ones(3)

    start_pose[:3] = torch.tensor([0., -0.8, 0.01])
    end_pose[:3]   = torch.tensor([0.5,  0.9, 0.6])

    start_pose[6:15] = torch.eye(3).reshape(-1)
    end_pose[6:15] = torch.eye(3).reshape(-1)

    planner = Planner(renderer, start_pose, end_pose, cfg)

    with open('actions.json', 'r') as fp:
        meta = json.load(fp)
        actions = meta["actions"]

    init_actions = torch.tensor(actions[:20])

    #print(init_actions)

    proj_states, action_planner = planner.plan_traj(start_pose, init_actions)

    print('Projected states', proj_states)
    print('Actions Planner', action_planner)
    #print('Initialized actions', init_actions)