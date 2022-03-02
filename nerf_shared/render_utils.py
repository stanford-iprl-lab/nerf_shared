import imageio
import numpy as np
import os
import torch

import torch.nn.functional as F

from nerf_shared import nerf
from nerf_shared import utils

DEBUG = False

class Renderer(torch.nn.Module):
    def __init__(self, perturb=True, N_importance=128, N_samples=64, use_viewdirs=True,
                 white_bkgd=True, raw_noise_std=0.0, ndc=False, lindisp=False,
                 near=0.0, far=1.0):
        """
        Stores values as class data.
        """
        super(Renderer, self).__init__()
        self.perturb = perturb
        self.N_importance = N_importance
        self.N_samples = N_samples
        self.use_viewdirs = use_viewdirs
        self.white_bkgd = white_bkgd
        self.raw_noise_std = raw_noise_std
        self.ndc = ndc
        self.lindisp = lindisp
        self.near = near
        self.far = far

        print(self.__dict__)

    def render_from_pose(self, H, W, K, chunk, c2w, coarse_model,
                         fine_model, retraw=True):
        rgb, disp, acc, extras = self.render(
            H, W, K, coarse_model, fine_model, chunk=chunk, c2w=c2w, retraw=retraw)

        return rgb, disp, acc, extras

    def render_from_rays(self, H, W, K, chunk, rays, coarse_model,
                         fine_model, retraw=True):
        rgb, disp, acc, extras = self.render(H, W, K, coarse_model, fine_model,
                                             chunk=chunk, rays=rays, retraw=retraw)

        return rgb, disp, acc, extras

    def render_path(self):
        pass

    def render_batch(self, coarse_model, fine_model, rays_flat,
                     chunk=1024*32, retraw=False):
        """Render rays in smaller minibatches to avoid OOM.
        """
        all_ret = {}
        for i in range(0, rays_flat.shape[0], chunk):
            ret = self.render_rays(rays_flat[i:i+chunk], coarse_model,
                                   fine_model, retraw)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret

    def render_rays(self,
                    ray_batch,
                    coarse_model,
                    fine_model,
                    retraw=False,
                    retweights=False,
                    verbose=False,
                    pytest=False):
        """Volumetric rendering.
        Args:
          ray_batch: array of shape [batch_size, ...]. All information necessary
            for sampling along a ray, including: ray origin, ray direction, min
            dist, max dist, and unit-magnitude viewing direction.
          coarse model: self-explanatory
          fine model: self-explanatory
          retraw: bool. If True, include model's raw, unprocessed predictions.
          retweights: bool. If True, include the points and associated weights.
          network_fine: "fine" network with same spec as network_fn.
          verbose: bool. If True, print more debugging info.
        Returns:
          rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
          disp_map: [num_rays]. Disparity map. 1 / depth.
          acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
          raw: [num_rays, num_samples, 4]. Raw predictions from model.
          rgb0: See rgb_map. Output for coarse model.
          disp0: See disp_map. Output for coarse model.
          acc0: See acc_map. Output for coarse model.
          z_std: [num_rays]. Standard deviation of distances along ray for each
            sample.
        """
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
        #viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 6 else None
        #print(ray_batch.shape)
        viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
        near, far = bounds[...,0], bounds[...,1] # [-1,1]

        t_vals = torch.linspace(0., 1., steps=self.N_samples)
        if not self.lindisp:
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

        z_vals = z_vals.expand([N_rays, self.N_samples])

        #perturb_points = self.perturb and self.training

        if self.perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.Tensor(t_rand)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

        raw = coarse_model(pts, viewdirs)
        rgb_map, disp_map, acc_map, weights, depth_map = self.raw2outputs(
            raw, z_vals, rays_d, pytest=pytest)

        if self.N_importance > 0:

            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = utils.sample_pdf(z_vals_mid, weights[...,1:-1],
                                         self.N_importance,
                                         det=(self.perturb==0.), pytest=pytest)
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

            if fine_model is None:
                raw = coarse_model(pts, viewdirs)
            else:
                raw = fine_model(pts, viewdirs)

            rgb_map, disp_map, acc_map, weights, depth_map = self.raw2outputs(
                raw, z_vals, rays_d, pytest=pytest)

        ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
        if retraw:
            ret['raw'] = raw
        if retweights:
            ret['weights'] = weights
            ret['z_vals'] = z_vals
        if self.N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        for k in ret:
            if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
                print(f"! [Numerical Error] {k} contains nan or inf.")

        return ret

    def render(self, H, W, K, coarse_model, fine_model, chunk=1024*32,
               rays=None, retraw=True,
               c2w=None, c2w_staticcam=None):
        """Render rays
        Args:
          H: int. Height of image in pixels.
          W: int. Width of image in pixels.
          focal: float. Focal length of pinhole camera.
          chunk: int. Maximum number of rays to process simultaneously. Used to
            control maximum memory usage. Does not affect final results.
          rays: array of shape [2, batch_size, 3]. Ray origin and direction for
            each example in batch.
          retraw: bool. If True, include model's raw, unprocessed predictions.
          c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
          c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
           camera while using other c2w argument for viewing directions.
        Returns:
          rgb_map: [batch_size, 3]. Predicted RGB values for rays.
          disp_map: [batch_size]. Disparity map. Inverse of depth.
          acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
          extras: dict with everything returned by render_rays().
        """
        if c2w is not None:
            # special case to render full image
            rays_o, rays_d = utils.get_rays(H, W, K, c2w)
        else:
            # use provided ray batch
            rays_o, rays_d = rays

        if self.use_viewdirs:
            # provide ray directions as input
            viewdirs = rays_d
            if c2w_staticcam is not None:
                # special case to visualize effect of viewdirs
                rays_o, rays_d = utils.get_rays(H, W, K, c2w_staticcam)
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1,3]).float()

        sh = rays_d.shape # [..., 3]
        if self.ndc:
            # for forward facing scenes
            rays_o, rays_d = utils.ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1,3]).float()
        rays_d = torch.reshape(rays_d, [-1,3]).float()

        near, far = self.near * torch.ones_like(rays_d[...,:1]), self.far * torch.ones_like(rays_d[...,:1])
        rays = torch.cat([rays_o, rays_d, near, far], -1)
        if self.use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)

        # Render and reshape
        #print(rays.shape)
        all_ret = self.render_batch(coarse_model, fine_model, rays, chunk, retraw)
        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        k_extract = ['rgb_map', 'disp_map', 'acc_map']
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
        return ret_list + [ret_dict]


    def raw2outputs(self, raw, z_vals, rays_d, pytest=False):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        noise = 0.
        if self.raw_noise_std > 0.:
            noise = torch.randn(raw[...,3].shape) * self.raw_noise_std

            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[...,3].shape)) * self.raw_noise_std
                noise = torch.Tensor(noise)

        alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]

        #print('Alpha', alpha.shape)

        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]

        #print('Weights', weights.shape)

        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)

        if self.white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[...,None])

        return rgb_map, disp_map, acc_map, weights, depth_map


    def render_from_batch_poses(self, H, W, K, chunk, batch_c2w, coarse_model, fine_model,
                                retraw, save_directory, b_combine_as_video=False,
                                tb_writer=None):
        '''
        take in a set of poses, render them as images, and save them or log them
        with tensorboard.
        '''
        os.makedirs(save_directory, exist_ok=True)
        rgbs = []
        with torch.no_grad():
            for i, c2w in enumerate(batch_c2w):
                rgb, _, _, _ = self.render_from_pose(H,
                                                     W,
                                                     K,
                                                     chunk=chunk,
                                                     c2w=c2w,
                                                     coarse_model=coarse_model,
                                                     fine_model=fine_model)
                rgbs.append(rgb.cpu().detach().numpy())
                rgb8 = utils.to8b(rgbs[-1])
                filename = os.path.join(save_directory, '{:03d}.png'.format(i))
                imageio.imwrite(filename, rgb8)
            if b_combine_as_video:
                imageio.mimwrite(os.path.join(save_directory, 'video.mp4'), utils.to8b(rgbs), fps=30, quality=8)
            if tb_writer is not None:
                rgb_tensor = torch.tensor(utils.to8b(rgbs))
                tb_writer.add_images('Test/Images', rgb_tensor, dataformats="NHWC")

# def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

#     H, W, focal = hwf

#     print('Render factor', render_factor)

#     if render_factor!=0:
#         # Render downsampled for speed
#         H = H//render_factor
#         W = W//render_factor
#         focal = focal/render_factor

#     rgbs = []
#     disps = []

#     t = time.time()
#     for i, c2w in enumerate(tqdm(render_poses)):
#         print(i, time.time() - t)
#         t = time.time()
#         print('C2W',c2w)
#         rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
#         rgbs.append(rgb.cpu().numpy())
#         disps.append(disp.cpu().numpy())
#         if i==0:
#             print(rgb.shape, disp.shape)

#         """
#         if gt_imgs is not None and render_factor==0:
#             p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
#             print(p)
#         """

#         if savedir is not None:
#             rgb8 = to8b(rgbs[-1])
#             filename = os.path.join(savedir, '{:03d}.png'.format(i))
#             imageio.imwrite(filename, rgb8)

#     rgbs = np.stack(rgbs, 0)
#     disps = np.stack(disps, 0)

#     return rgbs, disps
