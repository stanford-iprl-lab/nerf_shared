import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np






def main():

    # load nerf from file
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)

    start_pose = None
    end_pose = None

    # prepare initial path


    for iteration in range(1000):

        # execute step

        # add randomness to action

        # relocalize

        # replan
        # log new path



def generate_initial_path():
    pass

def path_from_parameters():
    pass

def path_cost():
    pass
