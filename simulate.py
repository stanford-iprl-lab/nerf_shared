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

    rot = pose[6:15].reshape((3, 3))

    r = Rotation.from_matrix(np.array(rot))

    roll, pitch, yaw = r.as_euler('xyz')

    #Construct state
    state[:3] = torch.tensor(pose[:3])
    state[3] = torch.tensor(yaw)

    return state

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
        #start_state = torch.tensor([-0.05,-0.9, 0.2, 0]).to(device)
        #end_state   = torch.tensor([-0.2 , 0.7, 0.15 , 0]).to(device)

        #Playground
        start_state = torch.tensor([0.0, -0.8, 0.01, 0])
        end_state   = torch.tensor([0.0,  3, 0.6 , 0])
        #end_state   = torch.tensor([-0.5,  1., 0.6 , 0])

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
                "epochs_init": 1,
                "fade_out_epoch": 500,
                "fade_out_sharpness": 10,
                "epochs_update": 500,
                }

        #Initialize Planner and Estimator:
        #Planner should initialize with A*
        #Arguments: Initial Pose P0, final pose PT, Number of Time Steps T, Discretization of A* N

        traj = System(renderer, start_state, end_state, start_vel, end_vel, cfg)
        traj.learn_init()
        traj.plot()

        actions = traj.get_actions()

        act = {}

        act["actions"] = actions.cpu().detach().numpy().tolist()
        with open('actions.json', 'w') as outfile:
            json.dump(act, outfile)

        plan_cfg = {"T_final": 2,
                "steps": 20,
                "lr": 0.1,
                "epochs_init": 2500,
                "fade_out_epoch": 500,
                "fade_out_sharpness": 10,
                "epochs_update": 500,
                'x_length': 0.1,
                'y_length': 0.1,
                'z_length': 0.05,
                'cloud_density': 1000,
                'mass': 1.,
                'g': 10.,
                'I': torch.eye(3)
                }

        start_pose = torch.zeros(18)
        end_pose = torch.zeros(18)

        start_pose[:3] = start_state[:3]
        end_pose[:3]   = end_state[:3]

        start_pose[6:15] = torch.eye(3).reshape(-1)
        end_pose[6:15] = torch.eye(3).reshape(-1)

        drone_planner = Planner(renderer, start_pose, end_pose, plan_cfg)

        with open('actions.json', 'r') as fp:
            meta = json.load(fp)
            actions = meta["actions"]

        init_actions = torch.tensor(actions[:20])

        #print(init_actions)

        proj_states, action_planner = drone_planner.plan_traj(start_pose, init_actions)

        print('Projected states', proj_states)
        print('Actions Planner', action_planner)

        '''

        #Convert planner poses into states
        x0 = convert_planner_init_to_agent_init(start_state, start_vel)

        #Arguments: Number of grad. descent iterations N_iter
        estimator = Estimator(N_iter, 512, 'interest_regions', renderer, dil_iter=3, kernel_size=5, lrate=.01, noise=None, sigma=0.01, amount=0.8, delta_brightness=0.)

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
        agent = Agent(x0, sim_cfg, agent_cfg, agent_type=None)

        true_states = [x0]
        pose_estimates = []

        measured_states = []

        actions = traj.get_actions()

        act = {}

        act["actions"] = actions.cpu().detach().numpy().tolist()
        with open('actions.json', 'w') as outfile:
            json.dump(act, outfile)

        for iter in trange(1):
            actions = traj.get_actions()

            states = []
            agent.reset()
            for act in actions:
                true_pose, true_state, gt_img = agent.step(act)
                states.append(true_state)

            true_state = states[iter + 2]

            if iter == 0:
                true_pose, true_state, gt_img = agent.step(actions[0])
                true_pose, true_state, gt_img = agent.step(actions[1])
                true_pose, true_state, gt_img = agent.step(actions[2])
            else:
                #true_pose, true_state, gt_img = agent.step((actions[iter + 2] + actions[iter + 3])/2)
                true_pose, true_state, gt_img = agent.step(actions[iter + 2])

            print('Action shape', actions.shape)

            true_state = convert_pose_to_planner_state(true_state)
            true_states.append(true_state)

            measured_state = traj.states[0, :].detach()
            measured_states.append(measured_state)

            #print('Propagated next state', measured_state)
            #print('Expected next state', traj.states[0])

            traj.update_state( measured_state )
            traj.learn_update()
            traj.save_poses('paths/Step' + f'{iter} poses.json')
            traj.plot()

        print(true_states)
        print(measured_states)
        print(traj.get_states(), traj.get_states().shape)

        for iter in trange(cfg["steps"]):

            print(f'Iteration {iter}')

            actions = traj.get_actions()
            act = actions[iter]

            #Step based on recommended action. Action should be array, not tensor! Output true_pose and gt_img are arrays.
            true_pose, true_state, gt_img = agent.step(act)
            true_states.append(true_state)

            measured_state = convert_pose_to_planner_state(true_state)

            traj.update_state( measured_state )
            traj.learn_update()
            traj.save_poses('paths/Step' + f'{iter} poses.json')
            traj.plot()

            #plt.imsave('paths/traj_image.png', gt_img)

            #print('True state', true_state)

            #Estimate pose from ground truth image initialized from above. Estimate_pose will print MSE loss and rotational & translational errors.
            #Assume inputs to estimate_pose are arrays.

            #pose_estimates.append(pose_estimate)

            # Use planner dynamics as real dynamics + noise. The current state is once we've already taken an action.
            #current_state = traj.states[0, :].detach()
            #randomness = torch.normal(mean= 0, std=torch.tensor([0.02, 0.02, 0.02, 0.1]) )

            #INERF WITH DYNAMICS

            states, vel, rot = traj.get_full_state()

            pose_init = np.zeros((4,4))
            pose_init[:3, :3] = rot[iter + 3]
            pose_init[:3, 3] = states[iter + 3, :3]
            pose_init[3, 3] = 1.

            gt_img, nerf_pose = agent.step_planner_dynamics(pose_init)

            plt.figure()
            plt.imsave('./paths/gt_img.png', gt_img)

            pose_estimate = estimator.estimate_pose(nerf_pose, gt_img, nerf_pose)

            pose_estimate = convert_sim_to_blender_pose(pose_estimate)

            # True state
            #measured_state = current_state + randomness     

            #print('Measured state', measured_state)

            measured_state = convert_pose_to_planner_state(pose_estimate)

            print('Difference Pose', pose_init, pose_estimate)

            print('Difference yaw', states[iter + 1, 3], measured_state[3])

            traj.update_state( measured_state )
            traj.learn_update()
            traj.save_poses('paths/Step' + f'{iter} poses.json')
            traj.plot()
            '''

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