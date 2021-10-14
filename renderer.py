

import numpy as np
import torch

from torchtyping import TensorDetail, TensorType
from typeguard import typechecked

from nerf import NeRF
from poses import Pose

class Renderer:
    def __init__(self, nerf: NeRF):
        raise NotImplemented
        self.nerf = nerf
        self.H = ?
        self.W = ?
        self.chunk = ?

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
        rgb, _, _, _ = render(self.H, self.W, self.K, chunk=self.chunk, rays=batch_rays, **self.render_kwargs)

        return rgb

    def get_full_img(self, camera_pose: Pose) -> TensorType['H', 'W', 3]:
        "Returns a full images rendered from a pose"
        raise NotImplemented

        rgb, _, _, _ = render(self.H, self.W, self.K, chunk=self.chunk, c2w=pose[:3, :4], **self.render_kwargs)

        return rgb

