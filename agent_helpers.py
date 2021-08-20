import quaternion
import numpy as np

class Agent():
    def __init__(self, P0, scene_dir, agent_type) -> None:

        #Initialize simulator
        sim = simulation(scene_dir)

        self.agent_type = agent_type

        self.pose = P0

        def pose2state(self):
            if self.agent_type == 'planar':
                pass
            pass

        def state2pose(self):
            pass

        def step(self, pose):

            #Naively add noise to the pose and treat that as ground truth.
            std = 1e-2

            rot = quaternion.as_rotation_vector(quaternion.from_rotation_matrix(pose[:3, :3]))
            rot = rot + np.random.normal(0, std, size=(3, ))
            trans = pose[:3, 3].reshape(3, ) + np.random.normal(0, std, size=(3, ))

            new_rot = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(rot))

            new_pose = np.eye(4)
            new_pose[:3, :3] = new_rot
            new_pose[:3, 3] = trans

            img = sim.get_image(new_pose)

            return new_pose, img[:, :, :3]