import os
from nerf_core import *

from torchtyping import TensorDetail, TensorType
from typeguard import typechecked

class renderer():
    def __init__(self, hwf, chunk, render_kwargs) -> None:
        self.hwf = hwf
        self.chunk = chunk
        self.render_kwargs = render_kwargs

        self.network_fn = render_kwargs["network_fn"]
        self.network_fine = render_kwargs["network_fine"]
        self.network_query_fn = render_kwargs["network_query_fn"]

    def get_img_from_pix(self, pix: TensorType["Number Pixels", 2], pose: TensorType[4, 4]) -> TensorType["Number Pixels", 3]:
        "Returns colors of subsampled image at the pixels specified by the input ([[x1, y1], [x2, y2], ...])"
        H, W, K = self.hwf

        rays_o, rays_d = get_rays(H, W, K,pose)  # (H, W, 3), (H, W, 3)

        rays_o = rays_o[pix[:, 0], pix[:, 1]]  # (N_rand, 3)
        rays_d = rays_d[pix[:, 0], pix[:, 1]]
        batch_rays = torch.stack([rays_o, rays_d], 0)

        rgb, _, _, _ = render(H, W, K, chunk=self.chunk, rays=batch_rays, **self.render_kwargs)

        return rgb

    def get_density_from_pt(self, pt, viewdirs):
        "[N_rays, N_samples, 3] input for pt ([1, 1, 3]) in this case. Can provide None as view_dir argument to network_query_fn ???. Returns output ([1, 1, 4]: [R, G, B, density])"

        run_fn = self.network_fn if self.network_fine is None else self.network_fine
#       raw = run_network(pts, fn=run_fn)
        raw = self.network_query_fn(pt, None, run_fn)

        return raw

