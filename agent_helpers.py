import os
import quaternion
import numpy as np
from simulator import Simulation
import torch
import json

#Helper functions

vec_to_rot_matrix = lambda x: quaternion.as_rotation_matrix(quaternion.from_rotation_vector(x))

rot_matrix_to_vec = lambda y: quaternion.as_rotation_vector(quaternion.from_rotation_matrix(y))

def convert_blender_to_sim_pose(pose):
    #Incoming pose converts body canonical frame to world canonical frame. We want a pose conversion from body
    #sim frame to world sim frame.
    world2sim = np.array([[1., 0., 0.],
                        [0., 0., 1.],
                        [0., -1., 0.]])
    body2cam = world2sim
    rot = pose[:3, :3]          #Rotation from body to world canonical
    trans = pose[:3, 3]

    rot_c2s = world2sim @ rot @ body2cam.T
    trans_sim = world2sim @ trans

    print('Trans', trans)
    print('Trans sim', trans_sim)

    c2w = np.zeros((4, 4))
    c2w[:3, :3] = rot_c2s
    c2w[:3, 3] = trans_sim
    c2w[3, 3] = 1.

    return c2w

def skew_matrix(vector):  # vector to skewsym. matrix

    ss_matrix = np.zeros((3,3))
    ss_matrix[0, 1] = -vector[2]
    ss_matrix[0, 2] = vector[1]
    ss_matrix[1, 0] = vector[2]
    ss_matrix[1, 2] = -vector[0]
    ss_matrix[2, 0] = -vector[1]
    ss_matrix[2, 1] = vector[0]

    return ss_matrix

def skew_matrix_torch(vector):  # vector to skewsym. matrix

    ss_matrix = torch.zeros((3,3))
    ss_matrix[0, 1] = -vector[2]
    ss_matrix[0, 2] = vector[1]
    ss_matrix[1, 0] = vector[2]
    ss_matrix[1, 2] = -vector[0]
    ss_matrix[2, 0] = -vector[1]
    ss_matrix[2, 1] = vector[0]

    return ss_matrix


def rotation_matrix(angles):
    ct = torch.cos(angles[0])
    cp = torch.cos(angles[1])
    cg = torch.cos(angles[2])
    st = torch.sin(angles[0])
    sp = torch.sin(angles[1])
    sg = torch.sin(angles[2])

    #Create rotation matrices
    R_x = torch.zeros((3, 3))
    R_x[0, 0] = 1.
    R_x[1, 1] = ct
    R_x[1, 2] = -st
    R_x[2, 1] = st
    R_x[2, 2] = ct

    R_y = torch.zeros((3, 3))
    R_y[0, 0] = cp
    R_y[0, 2] = sp
    R_y[1, 1] = 1.
    R_y[2, 0] = -sp
    R_y[2, 2] = cp

    R_z = torch.zeros((3, 3))
    R_z[0, 0] = cg
    R_z[0, 1] = -sg
    R_z[1, 0] = sg
    R_z[1, 1] = cg
    R_z[2, 2] = 1.

    #R_x = torch.tensor([[1,0,0],[0,ct,-st],[0,st,ct]])
    #R_y = torch.tensor([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    #R_z = torch.tensor([[cg,-sg,0],[sg,cg,0],[0,0,1]])
    R = R_z @ R_y @ R_x
    return R

def wrap_angle(val):
    pi = torch.tensor(np.pi)
    return torch.remainder(val + pi, (2 * pi)) - pi

def add_noise_to_state(state, noise):
    state = state.numpy()
    rot = state[6:15]
    vec = rot_matrix_to_vec(rot.reshape(3, 3))

    condensed_state = np.concatenate((state[:6], vec, state[15:])) + noise

    rot_noised = vec_to_rot_matrix(condensed_state[6:9])

    return torch.tensor(np.concatenate((condensed_state[:6], rot_noised.reshape(-1), condensed_state[9:])), dtype=torch.float32)

class Agent():
    def __init__(self, x0, sim_cfg, cfg, agent_type=None) -> None:

        #Initialize simulator
        self.sim = Simulation(sim_cfg)

        self.agent_type = agent_type

        #Initialized pose
        self.x0 = x0
        self.x = x0

        self.dt = cfg['dt']
        self.g = cfg['g']
        self.mass = cfg['mass']
        self.I = cfg['I']
        self.invI = torch.inverse(self.I)

        self.states_history = [self.x.clone().cpu().detach().numpy().tolist()]

    def reset(self):
        self.x = self.x0
        return

    def step(self, action, noise=None):
        #DYANMICS FUNCTION

        action = action.reshape(-1)

        newstate = self.drone_dynamics(self.x, action)

        if noise is not None:
            newstate_noise = add_noise_to_state(newstate.cpu().clone().detach(), noise)
        else:
            newstate_noise = newstate

        self.x = newstate_noise

        new_state = newstate_noise.clone().cpu().detach().numpy()

        new_pose = np.zeros((4, 4))
        new_pose[:3, :3] = (new_state[6:15]).reshape((3, 3))
        new_pose[:3, 3] = new_state[:3]
        new_pose[3, 3] = 1.

        new_pose = convert_blender_to_sim_pose(new_pose)

        img = self.sim.get_image(new_pose)

        self.states_history.append(self.x.clone().cpu().detach().numpy().tolist())

        return new_pose, new_state, img[...,:3]


    def drone_dynamics(self, state, action):
        #State is 18 dimensional [pos(3), vel(3), R (9), omega(3)] where pos, vel are in the world frame, R is the rotation from points in the body frame to world frame
        # and omega are angular rates in the body frame
        next_state = torch.zeros(18)

        #Actions are [total thrust, torque x, torque y, torque z]
        fz = action[0]
        tau = action[1:]

        #Define state vector
        pos = state[0:3]
        v   = state[3:6]
        R_flat = state[6:15]
        R = R_flat.reshape((3, 3))
        omega = state[15:]

        # The acceleration
        sum_action = torch.zeros(3)
        sum_action[2] = fz

        dv = (torch.tensor([0,0,-self.mass*self.g]) + R @ sum_action)/self.mass

        # The angular accelerations
        domega = self.invI @ (tau - torch.cross(omega, self.I @ omega))

        # Propagate rotation matrix using exponential map of the angle displacements
        angle = omega*self.dt
        theta = torch.norm(angle, p=2)
        if theta == 0:
            exp_i = torch.eye(3)
        else:
            exp_i = torch.eye(3)
            angle_norm = angle / theta
            K = skew_matrix_torch(angle_norm)

            exp_i = torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.matmul(K, K)

        next_R = R @ exp_i

        next_state[0:3] = pos + v * self.dt
        next_state[3:6] = v + dv * self.dt

        next_state[6:15] = next_R.reshape(-1)

        next_state[15:] = omega + domega * self.dt

        return next_state

    def save_data(self, filename):
        true_states = {}
        true_states['true_states'] = self.states_history
        with open(filename,"w+") as f:
            json.dump(true_states, f)
        return