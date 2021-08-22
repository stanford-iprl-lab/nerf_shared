import os
import quaternion
import numpy as np
from simulator import Simulation

class Agent():
    def __init__(self, P0, scene_dir, hwf, agent_type=None) -> None:

        #Initialize simulator
        sim = Simulation(scene_dir, hwf)

        self.agent_type = agent_type

        #Initialized pose
        self.pose = P0

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

        def step(self, pose):
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
                pass
            
            else:
                print('System not identified')

            img = sim.get_image(new_pose)

            return new_pose, img[:, :, :3]