from nerf_core import render
import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
from skimage.transform import resize
from scipy.spatial.transform import Rotation
import math

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
from planner import Planner
from quad_helpers import Simulator, QuadPlot
from quad_helpers import rot_matrix_to_vec, vec_to_rot_matrix, next_rotation

DEBUG = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nerf_filter = True

#Helper Functions

def convert_sim_to_blender_pose(pose):
    #Incoming pose converts body canonical frame to world canonical frame. We want a pose conversion from body
    #sim frame to world sim frame.
    world2sim = np.array([[1., 0., 0.],
                        [0., 0., 1.],
                        [0., -1., 0.]])
    body2cam = world2sim
    rot = pose[:3, :3]          #Rotation from body to world canonical
    trans = pose[:3, 3]

    rot_c2s = world2sim.T @ rot @ body2cam
    trans_sim = world2sim.T @ trans

    print('Trans', trans)
    print('Trans sim', trans_sim)

    c2w = np.zeros((4, 4))
    c2w[:3, :3] = rot_c2s
    c2w[:3, 3] = trans_sim
    c2w[3, 3] = 1.

    return c2w

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

def convert_full_state2pose(state):
    state = state.numpy()
    pose = np.zeros((4, 4))
    pose[:3, :3] = state[6:15].reshape((3, 3))
    pose[:3, 3] = state[:3]
    pose[3, 3] = 1.

    return pose

def plot_trajectory(states, sim_states):
    x = states[:, 0].cpu().detach().numpy()
    y = states[:, 1].cpu().detach().numpy()
    z = states[:, 2].cpu().detach().numpy()

    xsim = sim_states[:, 0]
    ysim = sim_states[:, 1]
    zsim = sim_states[:, 2]

    ax = plt.axes(projection='3d')

    ax.scatter3D(x, y, z, c='b')

    ax.scatter3D(xsim, ysim, zsim, c='r')

    plt.show()
    plt.close()

    return

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

        # renderer = get_nerf('configs/stonehenge.txt')
        # stonehenge - simple
        start_pos = torch.tensor([-0.05,-0.9, -0.1])
        end_pos   = torch.tensor([-0.32,0.6, 0.37])
        # start_pos = torch.tensor([-1, 0, 0.2])
        # end_pos   = torch.tensor([ 1, 0, 0.5])

        start_R = vec_to_rot_matrix( torch.tensor([0.2,0.3,0]))
        end_R = vec_to_rot_matrix(-torch.tensor([0.2,0.3,0]))

        start_state = torch.cat( [start_pos, torch.tensor([0,1,0]), start_R.reshape(-1), torch.zeros(3)], dim=0 )
        end_state   = torch.cat( [end_pos,   torch.zeros(3), end_R.reshape(-1), torch.zeros(3)], dim=0 )


        cfg = {"T_final": 2,
                "steps": 20,
                "lr": 0.01,
                "epochs_init": 2500,
                "fade_out_epoch": 0,
                "fade_out_sharpness": 10,
                "epochs_update": 250,
                }

        traj = System(renderer, start_state, end_state, cfg)
        traj.learn_init()

        #Arguments: Starting pose P0. Within Agent class, the simulator class is initialized. P0 initial pose must be array.
        #TODO: MOVE AGENT CONFIG TO OUTSIDE
        agent_cfg = {'dt': cfg["T_final"]/cfg["steps"],
                    'mass': 1.,
                    'g': 10.,
                    'I': torch.eye(3)}

        ###TODO: MAKE SURE YOU ALSO PASS IN SIMULATOR CONFIGS
        ###TODO: CHANGE INITIAL POSE TO BE AN 18 DIMENSIONAL STATE THAT ALIGNS WITH THE TRAJECTORY PLANNER INITIALIZATIONS
        ###CAUTION: states in agent dynamics and planner are in coordinates [right, forward, up], but simulator works in
        ### [right, up, back]
        agent = Agent(start_state, sim_cfg, agent_cfg, agent_type=None)

        if nerf_filter == True:
            sig = .1 * torch.eye(start_state.shape[0])

            Q = 1e-2 * torch.eye(start_state.shape[0])

            estimator = Estimator(N_iter, 512, 'interest_regions', renderer, agent, start_state, sig, Q, dil_iter=3, kernel_size=5, lrate=.01, noise=None, sigma=0.0, amount=0., delta_brightness=0.)
        else:
            #Arguments: Number of grad. descent iterations N_iter
            estimator = Estimator(N_iter, 512, 'interest_regions', renderer, dil_iter=3, kernel_size=5, lrate=.01, noise=None, sigma=0.0, amount=0., delta_brightness=0.)

        true_states = start_state.cpu().detach().numpy()

        measured_states = []
        
        noise = np.random.normal(0., [1e-2, 1e-2, 1e-2, 0., 0., 0., 1e-1, 1e-1, 1e-1, 0., 0., 0.])

        state_estimate = start_state

        for iter in trange(cfg['steps']):
            if iter < cfg['steps'] - 4:
                action = traj.get_next_action().clone().detach()
            else:
                action = traj.get_actions()[iter - cfg['steps'] + 3, :]

            true_pose, true_state, gt_img = agent.step(action, noise)
            true_states = np.vstack((true_states, true_state))

            plt.figure()
            plt.imsave('./paths/true/'+ f'{iter}_gt_img.png', gt_img)
            plt.close()

            if nerf_filter == True:
                state_est = estimator.estimate_state(gt_img, true_pose, action)
                measured_state = state_est
            else:
                #Propagate state estimate
                next_estimate = agent.drone_dynamics(state_estimate, action)

                est_pose = convert_full_state2pose(next_estimate.cpu().detach())

                est_pose = convert_blender_to_sim_pose(est_pose)

                pose_estimate = estimator.estimate_pose(est_pose, gt_img, true_pose)

                pose_estimate = convert_sim_to_blender_pose(pose_estimate)

                measured_state = next_estimate.cpu().clone().detach().numpy()
                measured_state[:3] = pose_estimate[:3, 3]
                measured_state[6:15] = pose_estimate[:3, :3].reshape(-1)

            measured_state = torch.tensor(measured_state)
            measured_states.append(measured_state)

            state_estimate = measured_state

            if iter < cfg['steps'] - 4:
                traj.update_state( measured_state )
                traj.learn_update(iter)

            #plot_trajectory(traj.get_full_states(), true_states)

        print(true_states)
        print(measured_states)
        print(traj.get_full_states())

        #plot_trajectory(traj.get_full_states(), true_states)
        estimator.save_data('./paths/estimator_data.json')
        agent.save_data('./paths/agent_data.json')

    else:
        ####################################### DEBUGING ENVIRONMENT ####################################################3
        pass

    return

####################### END OF MAIN LOOP ##########################################