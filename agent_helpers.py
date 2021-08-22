import os
import quaternion
import numpy as np
from simulator import Simulation

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

class Agent():
    def __init__(self, P0, scene_dir, hwf, agent_type=None) -> None:

        #Initialize simulator
        self.sim = Simulation(scene_dir, hwf)

        self.agent_type = agent_type

        #Initialized pose
        self.pose = P0

        if agent_type == 'quad':
            self.mass = 1
            self.dt = 0.1
            self.g = -10
            self.J = np.eye(3)
            self.J_inv = np.linalg.inv(self.J)

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

    def step(self, pose, action=None):
        #DYANMICS FUNCTION

        if self.agent_type is None:
            #Naively add noise to the pose and treat that as ground truth.
            std = 1e-2

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
                pos = state[...,0:3]
                v   = state[...,3:6]
                euler_vector = state[...,6:9]
                omega = state[...,9:12]

                fz = action[...,0, None]
                torque = action[...,1:4]

                e3 = np.array([0,0,1])

                #R = vec_to_rot_matrix(euler_vector)
                R = vec_to_rot_matrix(euler_vector)

                dv = self.g * e3 + R @ (fz * e3) / self.mass
                domega = self.J_inv @ torque - self.J_inv @ skew_matrix(omega) @ self.J @ omega
                dpos = v

                next_v = v + dv * self.dt
                next_euler = rot_matrix_to_vec(R @ vec_to_rot_matrix(omega * self.dt))
                # next_euler = rot_matrix_to_vec(R @ vec_to_rot_matrix(omega * dt) @ vec_to_rot_matrix(domega * dt**2))
                next_omega = omega + domega * self.dt
                next_pos = pos + dpos * self.dt# + 0.5 * dv * dt**2

                next_state = np.concatenate([next_pos, next_v, next_euler, next_omega], dim=-1)

        else:
            print('System not identified')

        img = self.sim.get_image(new_pose)

        return new_pose, img[:, :, :3]