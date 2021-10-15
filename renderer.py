

import numpy as np
import torch

from torchtyping import TensorDetail, TensorType
from typeguard import typechecked

from original.nerf_core import render
from original.run_nerf_helpers import get_rays

from nerf import NeRF
from poses import Pose

class Renderer:
    def __init__(self, nerf: NeRF, H = 400, W = 400):
        raise NotImplemented
        self.nerf = nerf
        self.H = H
        self.W = W
        self.chunk = ? #TODO move chunk into nerf cfg

        self.training = False

    def train(self, train):
        self.training = train
    
    def eval(self):
        self.training = False

    def get_render_kwargs(self, nerf: NeRF):

        network_query_fn = lambda inputs, viewdirs, network_fn : network_fn(input, viewdirs)
        nerf.forward(input, viewdirs, fine_network = True)

        render_kwargs = {
            'network_query_fn' : network_query_fn,
            'perturb' : args.perturb if self.training else False,
            'N_importance' : args.N_importance,

            'network_fn' : model,
            'network_fine' : model_fine,

            'N_samples' : args.N_samples,
            'use_viewdirs' : args.use_viewdirs,
            'white_bkgd' : args.white_bkgd,
            'raw_noise_std' : args.raw_noise_std if self.training else 0.,
        }

        return render_kwargs

    def get_subsambled_img(self, pix: TensorType['NumPixels', 2], camera_pose: Pose, HW=True) -> TensorType['NumPixels', 3]:
        "Returns colors of subsampled image at the pixels specified by the input ([[x1, y1], [x2, y2], ...])"
        raise NotImplemented

        rays_o, rays_d = get_rays(self.H, self.W, self.K, pose)  # (H, W, 3), (H, W, 3)

        if HW == True:
            rays_o = rays_o[pix[:, 0], pix[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[pix[:, 0], pix[:, 1]]
        else:
            rays_o = rays_o[pix[:, 1], pix[:, 0]]  # (N_rand, 3)
            rays_d = rays_d[pix[:, 1], pix[:, 0]]

        batch_rays = torch.stack([rays_o, rays_d], 0)
        rgb, disp_map, acc_map, extras = render(self.H, self.W, self.K, chunk=self.chunk, rays=batch_rays, **self.get_render_kwargs())

        return rgb

    def get_full_img(self, camera_pose: Pose) -> TensorType['H', 'W', 3]:
        "Returns a full images rendered from a pose"
        raise NotImplemented

        rgb, disp_map, acc_map, extras = render(self.H, self.W, self.K, chunk=self.chunk, c2w=pose[:3, :4], **self.get_render_kwargs())

        return rgb

