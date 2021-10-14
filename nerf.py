
import numpy as np
import torch

from torchtyping import TensorDetail, TensorType
from typeguard import typechecked
from load_blender import load_blender_data
import pdb


class Nerf(torch.nn.Module):
    def __init__(self, cfg):
        super(Nerf, self).__init__()
        
        if cfg.dataset_type == "blender":
            images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
            print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
            i_train, i_val, i_test = i_split

            near = 0.
            far = 7.

            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]

        pdb.set_trace()

        raise NotImplemented


    # def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
    #     """ 
    #     """
    #     super(NeRF, self).__init__()
    #     self.D = D
    #     self.W = W
    #     self.input_ch = input_ch
    #     self.input_ch_views = input_ch_views
    #     self.skips = skips
    #     self.use_viewdirs = use_viewdirs
        
    #     self.pts_linears = nn.ModuleList(
    #         [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
    #     ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
    #     self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

    #     ### Implementation according to the paper
    #     # self.views_linears = nn.ModuleList(
    #     #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
    #     if use_viewdirs:
    #         self.feature_linear = nn.Linear(W, W)
    #         self.alpha_linear = nn.Linear(W, 1)
    #         self.rgb_linear = nn.Linear(W//2, 3)
    #     else:
    #         self.output_linear = nn.Linear(W, output_ch)

    @typechecked
    def forward(self, points: TensorType["dims":..., 3], view_direction: TensorType["dims":..., 3]) -> TensorType["dims":...,4]:
        raise NotImplemented

    @typechecked
    def density(self, points: TensorType["dims":..., 3]) -> TensorType["dims":...]:
        view_dir = torch.ones_like(points)
        output = self.forward(points, view_dir)
        return output[..., -1]

class CompositeNerf(Nerf):
    def __init__(self, nerf_list, pose_list, cfg):
        self.nerf_list = nerf_list
        self.pose_list = nerf_list

        raise NotImplemented

    @typechecked
    def forward(self, points: TensorType["dims":..., 3], view_direction: TensorType["dims":..., 3]) -> TensorType["dims":...,4]:
        raise NotImplemented

        for nerf, pose in zip(self.nerf_list, self.pose_list):
            nerf.forward(points, view_direction)

        # alpha combine

    @typechecked
    def density(self, points: TensorType["dims":..., 3]) -> TensorType["dims":...]:
        raise NotImplemented
