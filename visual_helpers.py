import os, sys
import torch
from torchtyping import TensorDetail, TensorType
from typeguard import typechecked

from nerf_core import *
from run_nerf_helpers import *

import matplotlib.pyplot as plt
import numpy as np

def draw_pyramid(points):
    base_pts = points[:-1, :]
    origin = points[-1, :]

    for tt in range(4):
        trianglex = base_pts[[tt,(tt+1)%4], 0]
        np.append(trianglex, origin[0])

        triangley = base_pts[[tt,(tt+1)%4], 1]
        np.append(triangley, origin[1])

        plt.fill(trianglex, triangley, facecolor='lightsalmon', edgecolor='orangered', alpha=0.1)
        plt.quiver(origin[0], origin[1], (base_pts[tt, 0] - origin[0]), (base_pts[tt, 1] - origin[1]), headaxislength=0, headlength=0)

    plt.fill(base_pts[:, 0], base_pts[:, 1], facecolor='lightsalmon', edgecolor='orangered', alpha=0.1)
    #plt.scatter(points[:, 0], points[:, 1])

    return

def scale(points, scale_factor):
    base_pts = points[:-1, :]
    origin = points[-1, :]

    for tt in range(4):
        points[tt, :] = scale_factor*(base_pts[tt, :] - origin) + origin

    return points

def visualize(background_pose: TensorType[4, 4], true_poses: TensorType["T", 3, 5], pose_estimates: TensorType["T", 4, 4], savedir: str, render_args, render_kwargs_train) -> None:

    #Plot background image
    scale_factor = 0.1

    H, W, K, chunk = render_args['H'], render_args['W'], render_args['K'], render_args['chunk']
    rgb, _, _, _ = render(H, W, K, chunk=chunk, c2w=background_pose[:3, :4], **render_kwargs_train)

    fig = plt.figure(1)
    plt.imshow(rgb.cpu().detach().numpy())

    background_pose = background_pose.cpu().detach().numpy()

    def pix2world(vertices, rot, dt, origin):
        dirs = np.zeros((vertices.shape[0], 3))
        for i, row in enumerate(vertices):
            dir = np.array([(row[0] - K[0][2])/K[0][0], -(row[1] - K[1][2])/K[1][1], -dt])
            dir = (rot@(dir.reshape(-1, 1)) + origin.reshape(-1, 1)).reshape(-1)
            print(dir)
            dirs[i, :] = dir

        return dirs

    def world2pix(pts, rot, origin):
        pixs = np.zeros((pts.shape[0], 2))
        for i, row in enumerate(pts):
            pix = rot.T @ (row.reshape(-1, 1) - origin.reshape(-1, 1))
            pix = [pix[0]*K[0][0] + K[0][2], -pix[1]*K[1][1] + K[1][2]]
            pixs[i, :] = pix

        return pixs

    def pose2pixel(pose, K):
        _, rays_d = get_rays(H, W, K, torch.Tensor(pose))

        K_full = np.hstack([K, np.zeros((3, 1))])

        dt = 1.

        #Get pose origin in world frame
        pose = pose.cpu().detach().numpy()
        origin = pose[:3, 3]

        print('Origin', origin)

        rot = pose[:3, :3]

        print('Rotation', rot)

        vertices = np.array([[0, 0], [0, H-1], [W-1, H-1], [W-1, 0]])

        pts = pix2world(vertices, rot, dt, origin)

        pixs = world2pix(pts, background_pose[:3, :3], background_pose[:3, 3])

        pix_origin = world2pix(np.array([origin]), background_pose[:3, :3], background_pose[:3, 3])

        return np.vstack([pixs, pix_origin])

    for t in trange(true_poses.shape[0]):

        pixels = pose2pixel(true_poses[t], K)
        pixels = scale(pixels, scale_factor)
        draw_pyramid(pixels)

        filename = os.path.join(savedir, '{:03d}.png'.format(t))
        plt.savefig(filename)
    

    print('Done Plotting')

    return