import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch

from torchtyping import TensorDetail, TensorType
from typeguard import typechecked

from tqdm import tqdm, trange

import matplotlib.pyplot as plt

# Import Helper Classes
from render_functions import Renderer
from visual_helpers import visualize
from estimator_helpers import Estimator
from agent_helpers import Agent


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
    hwf = render_args['hwf']
    chunk = render_args['chunk']
    K = render_args['K']

    renderer = Renderer(hwf, K, chunk, render_kwargs_train)

    #Initialize Planner and Estimator:
    #Planner should initialize with A*
    #Arguments: Initial Pose P0, final pose PT, Number of Time Steps T, Discretization of A* N
    planner = Planner(P0, PT, T, N)

    #Arguments: Number of grad. descent iterations N_iter
    estimator = Estimator(N_iter, 512, 'interest_regions', renderer, dil_iter=3, kernel_size=5, lrate=.01, noise=None, sigma=0.01, amount=0.8, delta_brightness=0.)

    #Arguments: Starting pose P0
    agent = Agent(P0)

    #Initialize planner with perfect knowledge of initial pose
    pose_estimate = P0

    true_poses = [P0]
    pose_estimates = []
    for iter in trange(N):
        print(f'Iteration {iter}')
        #Plan based on estimate of pose at current time step
        future_poses = planner.plan(pose_estimate)

        #Step based on recommended action
        true_pose, gt_img = agent.step()
        true_poses.append(true_pose)

        #Initialize estimator based on the planner's rollout pose at next time step
        pose_init = future_poses[0]

        #Estimate pose from ground truth image initialized from above. Estimate_pose will print MSE loss and rotational & translational errors.
        pose_estimate = estimator.estimate_pose(pose_init, gt_img, true_pose)
        pose_estimates.append(pose_estimate)

    #Visualizes the trajectory
    visualize(background_pose, true_poses, pose_estimates, savedir, render_args, render_kwargs_train)
    
    return

####################### END OF MAIN LOOP ##########################################