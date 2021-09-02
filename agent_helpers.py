import os
import quaternion
import numpy as np
from simulator import Simulation
import torch

#Helper functions

vec_to_rot_matrix = lambda x: quaternion.as_rotation_matrix(quaternion.from_rotation_vector(x))

rot_matrix_to_vec = lambda y: quaternion.as_rotation_vector(quaternion.from_rotation_matrix(y))

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

class Agent():
    def __init__(self, x0, sim_cfg, cfg, agent_type=None) -> None:

        #Initialize simulator
        self.sim = Simulation(sim_cfg)

        self.agent_type = agent_type

        #Initialized pose
        self.x = x0

        self.dt = cfg['dt']
        self.g = cfg['g']
        self.mass = cfg['mass']
        self.I = cfg['I']
        self.invI = torch.inverse(self.I)

    def pose2state(self):
        if self.agent_type == 'planar':
            pass

        elif self.agent_type == 'quad':
            pass

        else:
            print('System not identified')

    def state2pose(self):
        if self.agent_type == 'planar':
            pass

        elif self.agent_type == 'quad':
            pass

        else:
            print('System not identified')

    def step(self, action):
        #DYANMICS FUNCTION
        '''

        if self.agent_type is None:
            #Naively add noise to the pose and treat that as ground truth.
            std = 1*1e-2

            rot = quaternion.as_rotation_vector(quaternion.from_rotation_matrix(pose[:3, :3]))
            rot = rot + np.random.normal(0, std, size=(3, ))
            trans = pose[:3, 3].reshape(3, ) + np.random.normal(0, std, size=(3, ))

            new_rot = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(rot))

            new_pose = np.eye(4)
            new_pose[:3, :3] = new_rot
            new_pose[:3, 3] = trans
        elif self.agent_type == 'planar':
            pass

        elif self.agent_type == 'quad':
            state = self.pose2state(pose)
            
            if action is None:
                next_state = state

            else:
                pass

        else:
            print('System not identified')
        '''
        action = action.reshape(-1)

        newstate = self.drone_dynamics(self.x, action)
        new_state = newstate.cpu().detach().numpy()

        new_pose = np.zeros((4, 4))
        new_pose[:3, :3] = (new_state[6:15]).reshape((3, 3))
        new_pose[:3, 3] = new_state[:3]
        new_pose[3, 3] = 1.

        img = self.sim.get_image(new_pose)

        self.x = newstate

        return new_pose, newstate.detach(), img[...,:3]

        '''
        new_state = self.drone_dynamics_test(self.x, action)
        self.x = new_state

        return new_state
        '''

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
            angle_norm = angle/theta
            K = skew_matrix_torch(angle_norm)

            exp_i = torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.matmul(K, K)

        next_R = R @ exp_i

        next_state[0:3] = pos + v * self.dt
        next_state[3:6] = v + dv * self.dt

        next_state[6:15] = next_R.reshape(-1)

        next_state[15:] = omega + domega * self.dt

        return next_state

    def drone_dynamics_test(self, state, action):
        #TODO batch this
        state = state.cpu().detach().numpy()
        pos = state[...,0:3]
        v   = state[...,3:6]
        euler_vector = state[...,6:9]
        omega = state[...,9:12]

        action = action.cpu().detach().numpy()

        fz = action[...,0, None]
        torque = action[...,1:4]

        e3 = np.array([0,0,1])

        mass = 1
        dt = 0.1
        g = -10
        J = np.eye(3)
        J_inv = np.linalg.inv(J)

        R = vec_to_rot_matrix(euler_vector)

        dv = g * e3 + R @ (fz * e3) / mass
        domega = J_inv @ torque - J_inv @ skew_matrix(omega) @ J @ omega
        dpos = v

        next_v = torch.tensor(dv * dt + v)
        next_euler = torch.tensor(rot_matrix_to_vec(R @ vec_to_rot_matrix(omega * dt)))
        # next_euler = rot_matrix_to_vec(R @ vec_to_rot_matrix(omega * dt) @ vec_to_rot_matrix(domega * dt**2))
        next_omega = torch.tensor(domega * dt + omega)
        next_pos = torch.tensor(dpos * dt + pos) # + 0.5 * dv * dt**2

        next_state = torch.cat([next_pos, next_v, next_euler, next_omega], dim=-1)
        return next_state


