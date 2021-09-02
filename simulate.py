import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
from skimage.transform import resize
from scipy.spatial.transform import Rotation as R

from torchtyping import TensorDetail, TensorType
from typeguard import typechecked

from tqdm import tqdm, trange

import matplotlib.pyplot as plt

# Import Helper Classes
from render_functions import Renderer
from visual_helpers import visualize
from estimator_helpers import Estimator
from agent_helpers import Agent
from quad_plot import System
from quad_plot import get_manual_nerf

DEBUG = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Helper Functions
def convert_planner_init_to_agent_init(start_state, start_vel):
    #Start state is size 4, start vel is size 3. Outputs 18 dimensional state. Assumes rotation is identity and angular rates are 0.
    state = torch.zeros(18)
    pos = start_state[:3]
    yaw = start_state[-1]

    #Construct state
    rot = torch.eye(3).reshape(-1)
    state[:3] = pos
    state[3:6] = start_vel[:3]
    state[6:15] = rot
    state[15:] = torch.zeros(3)

    return state

def convert_pose_to_planner_state(pose):
    #Start state is size 4, start vel is size 3. Outputs 18 dimensional state. Assumes rotation is identity and angular rates are 0.
    state = torch.zeros(4)

    r = R.from_matrix(np.array(pose[:3, :3]))

    yaw, pitch, roll = r.as_euler('zyx')

    #Construct state
    state[:3] = torch.tensor(pose[:3, 3])
    state[-1] = torch.tensor(yaw)

    return state

####################### MAIN LOOP ##########################################
def main_loop(P0: TensorType[4, 4], PT: TensorType[4, 4], T: int, N: int, N_iter: int, savedir: str, render_args: dict, render_kwargs_train: dict, scene_dir: str) -> None:
    '''We've assumed that by calling this function, the NeRF model has already been created (i.e. create_nerf has been called) such that
    such that calling render() returns a valid RGB, etc tensor.

    How trajectory planning works:

    A good initialization for the sequence of poses is returned by running A*. This is only run once! A trajectory loss is computed, consisting of a collision loss
    (querying densities from the NeRF from x,y,z points) and a trust region loss. The outputs are a sequence of future rollout poses (where the planner wants the agent to be)
    and a control action(s) to update the agent. This algorithm is run MPC style, with the intent that A* yields a good initialization for the trajectory, and subsequent optimizations can just be done by
    performing gradient descent on the trajectory loss whilst having good performance. 

    How state estimation works: 

    Given an image, gradient descent is performed on the NeRF reconstruction loss, optimizing on the estimated pose in SE(3). The exponential map was used to create SE(3) from se(3) in R6 such that 
    the transformation is differentiable. Two sampling schemes exist: (1) random sampling of pixels from the full image H x W, or (2) random sampling from a mask around features detected by ORB/SIFT on the
    observed image (termed interest region sampling by iNeRF). 

    How the whole pipeline works:

    The objective is to path plan from pose P0 at time t = 0 to PT at time t = T. At time t, the agent runs the trajectory planning algorithm, yielding a control action(s) and future desired poses P{t+1:T}.
    The agent takes the control action and also receives an image corresponding to the "real" pose at time t + 1. The state estimator uses P{t+1} as the anchor of the tangential plane and returns P_hat_{t+1} = P @ P{t+1},
    where P in SE(3) are the parameters optimized by the state estimator. P_hat_{t+1} is passed to the trajectory planner as the pose estimate. 

    Args:
        

    '''
    ###TODO: MOVE THESE CONFIGS OUTSIDE
    hwf = render_args['hwf']
    chunk = render_args['chunk']
    K = render_args['K']

    sim_cfg = {'hwf': hwf,
                'scene_dir': scene_dir,
                'hfov': 40}

    if DEBUG == False:
        renderer = Renderer(hwf, K, chunk, render_kwargs_train)

        ####TODO: SHIFT INITIALIZATIONS TO BE PASSED IN AS A CONFIG FILE/DICTIONARY

        #PARAM initial and final velocities
        start_vel = torch.tensor([0, 0, 0, 0]).to(device)
        end_vel   = torch.tensor([0, 0, 0, 0]).to(device)

        # stonehenge - simple
        start_state = torch.tensor([-0.05,-0.9, 0.2, 0]).to(device)
        end_state   = torch.tensor([-0.2 , 0.7, 0.15 , 0]).to(device)

        # stonehenge - tricky
        # start_state = torch.tensor([ 0.4 ,-0.9, 0.2, 0])
        # end_state   = torch.tensor([-0.2 , 0.7, 0.15 , 0])

        # stonehenge - very simple
        # start_state = torch.tensor([-0.43, -0.75, 0.2, 0])
        # end_state = torch.tensor([-0.26, 0.48, 0.15, 0])

        ###TODO: SHIFT cfg TO BE PASSED IN AS CONFIG/DICTIONARY
        cfg = {"T_final": 2,
                "steps": 20,
                "lr": 0.001,
                "epochs_init": 500,
                "fade_out_epoch": 500,
                "fade_out_sharpness": 10,
                "epochs_update": 500,
                }

        #Initialize Planner and Estimator:
        #Planner should initialize with A*
        #Arguments: Initial Pose P0, final pose PT, Number of Time Steps T, Discretization of A* N

        traj = System(get_manual_nerf("empty"), start_state, end_state, start_vel, end_vel, cfg)
        traj.learn_init()
        traj.plot()

        #Convert planner poses into states
        x0 = convert_planner_init_to_agent_init(start_state, start_vel)

        #Arguments: Number of grad. descent iterations N_iter
        estimator = Estimator(N_iter, 512, 'interest_regions', renderer, dil_iter=3, kernel_size=5, lrate=.01, noise=None, sigma=0.01, amount=0.8, delta_brightness=0.)

        #Arguments: Starting pose P0. Within Agent class, the simulator class is initialized. P0 initial pose must be array.
        #TODO: MOVE AGENT CONFIG TO OUTSIDE
        agent_cfg = {'dt': cfg["T_final"]/cfg["steps"],
                    'mass': 1.,
                    'g': 10,
                    'I': torch.eye(3)}

        ###TODO: MAKE SURE YOU ALSO PASS IN SIMULATOR CONFIGS
        ###TODO: CHANGE INITIAL POSE TO BE AN 18 DIMENSIONAL STATE THAT ALIGNS WITH THE TRAJECTORY PLANNER INITIALIZATIONS
        ###CAUTION: states in agent dynamics and planner are in coordinates [right, forward, up], but simulator works in
        ### [right, up, back]
        agent = Agent(x0, sim_cfg, agent_cfg, agent_type=None)

        true_states = [x0]
        pose_estimates = []

        #actions = traj.get_actions()
        
        #for action in actions:
        #    true_pose = agent.step(action)
        #    print(action, true_pose)

        for iter in trange(N):

            print(f'Iteration {iter}')

            action = traj.get_next_action()

            #action = torch.tensor([12, 10, 10, 10])

            print('Action', action)

            #Step based on recommended action. Action should be array, not tensor! Output true_pose and gt_img are arrays.
            true_pose, true_state, gt_img = agent.step(action)
            true_states.append(true_state)

            plt.imsave('paths/traj_image.png', gt_img)

            print('True state', true_state)

            #Estimate pose from ground truth image initialized from above. Estimate_pose will print MSE loss and rotational & translational errors.
            #Assume inputs to estimate_pose are arrays.

            #pose_estimate = estimator.estimate_pose(pose_init, gt_img, true_pose)
            #pose_estimates.append(pose_estimate)

            # # idealy something like this but we jank it for now
            # action = traj.get_actions()[0 or 1, :]
            # current_state = next_state(action)

            # we jank it
            #current_state = traj.states[0, :].detach()
            #randomness = torch.normal(mean= 0, std=torch.tensor([0.02, 0.02, 0.02, 0.1]) )

            #measured_state = current_state + randomness
            #Convert 4x4 pose matrix into [x,y,z, yaw] 
            measured_state = convert_pose_to_planner_state(true_pose)           #No noise
            print('Measured state', measured_state)

            traj.update_state( measured_state )
            traj.learn_update()
            traj.save_poses('paths/Step' + f'{iter} poses.json')
            traj.plot()

        #Visualizes the trajectory
        with torch.no_grad():
            pass
            #visualize(background_pose, true_poses, pose_estimates, savedir, render_args, render_kwargs_train)


    else:
        ####################################### DEBUGING ENVIRONMENT ####################################################3
        renderer = Renderer(hwf, K, chunk, render_kwargs_train)

        estimator = Estimator(N_iter, 512, 'interest_regions', renderer, dil_iter=3, kernel_size=5, lrate=.01, noise=None, sigma=0., amount=0., delta_brightness=0.)

        agent = Agent(P0, scene_dir, hwf, agent_type=None)

        true_pose, gt_img = agent.step(P0.cpu().detach().numpy())  

        #plt.figure()
        #plt.imshow(gt_img)
        #plt.show()

        pose_estimate = estimator.estimate_pose(P0, gt_img, true_pose)

        #print(pose_estimate, true_pose, P0)     
    
        #density = renderer.get_density_from_pt(torch.tensor([[[0., 1., 0.], [0., 0.5, 0.]]]))
        #print('Density', density, density.shape)
    return

####################### END OF MAIN LOOP ##########################################