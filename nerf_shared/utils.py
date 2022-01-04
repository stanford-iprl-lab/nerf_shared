import imageio
import numpy as np
import os
import time
import torch
import tqdm

from nerf_shared import load_blender
from nerf_shared import load_deepvoxels
from nerf_shared import load_LINEMOD
from nerf_shared import load_llff
from nerf_shared import nerf
from nerf_shared import render_utils

import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False

### MISC FUNCTIONS ####

#Mean-squared error of rendered image pixel values - target image pixel values
img2mse = lambda x, y : torch.mean((x - y) ** 2)

#Converts mean-squared error to Peak-Signal-Noise-Ratio (PSNR)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

#Converts an image's pixels values from [0, 1] to [0, 255]
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

#### Ray-tracing helpers ###
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

# Numpy version of get_rays function
def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

def create_nerf_models(args):
    """Instantiate NeRF.
    """

    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    coarse_model = nerf.NeRF(D=args.netdepth, W=args.netwidth,
                 output_ch=output_ch, skips=skips,
                 use_viewdirs=args.use_viewdirs,
                 multires=args.multires, multires_views=args.multires_views,
                 i_embed=args.i_embed).to(device)

    fine_model = None
    if args.N_importance > 0:
        fine_model = nerf.NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          output_ch=output_ch, skips=skips,
                          use_viewdirs=args.use_viewdirs,
                          multires=args.multires, multires_views=args.multires_views,
                          i_embed=args.i_embed).to(device)

    return coarse_model, fine_model

def get_renderer(args, bds_dict):

    render_kwargs = {
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'N_samples' : args.N_samples,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs['ndc'] = False
        render_kwargs['lindisp'] = args.lindisp

    render_kwargs.update(bds_dict)

    return render_utils.Renderer(**render_kwargs)

def get_optimizer(coarse_model, fine_model, args):
    grad_vars = list(coarse_model.parameters())

    if fine_model is not None:
        grad_vars += list(fine_model.parameters())

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    return optimizer

def load_checkpoint(coarse_model, fine_model, optimizer, args, b_load_ckpnt_as_trainable=False):
    """
    b_load_ckpnt_as_trainable - controls if we load file w/ grad set to true or false. If model
        will continue to be trained this must be True, otherwise set to False to save memory
    """
    start = 0
    basedir = args.basedir
    expname = args.expname

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(
            os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)

    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        coarse_model.load_state_dict(ckpt['coarse_model_state_dict'], strict=False)
        for param in coarse_model.parameters():
            param.requires_grad = b_load_ckpnt_as_trainable

        if fine_model is not None:
            fine_model.load_state_dict(ckpt['fine_model_state_dict'])
            for param in fine_model.parameters():
                param.requires_grad = b_load_ckpnt_as_trainable

    return start

def load_datasets(args):
    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff.load_llff_data(
            args.datadir, args.factor, recenter=True,
            bd_factor=.75, spherify=args.spherify)

        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender.load_blender_data(
            args.datadir, args.half_res, args.testskip)

        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 0.
        far = 4.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        linemod_data = load_LINEMOD.load_LINEMOD_data(args.datadir,
                                                      args.half_res, args.testskip)

        images, poses, render_poses, hwf, K, i_split, near, far = linemod_data

        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_deepvoxels.load_dv_data(
            scene=args.shape, basedir=args.datadir, testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    bds_dict = {
    'near' : near,
    'far' : far,
    }

    i_split = i_train, i_val, i_test

    if args.render_test:
        render_poses = np.array(poses[i_test])

    return images, poses, render_poses, hwf, i_split, K, bds_dict

def copy_log_dir(args):
    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

def render_path(args, render_poses, images, hwf, K, i_split, start, render_kwargs_test):
    i_train, i_val, i_test = i_split
    basedir = args.basedir
    expname = args.expname

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')

        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

def batch_training_data(args, poses, hwf, K, images, i_train):
    H, W, _ = hwf

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)
    else:
        rays_rgb = torch.Tensor([]).to(device)
        i_batch = None

    return images, poses, rays_rgb, use_batching, N_rand, i_batch

def sample_random_ray_batch(args, images, poses, rays_rgb, N_rand, use_batching, i_batch, i_train, hwf, K, start, i):
    H, W, _ = hwf

    # Sample random ray batch
    if use_batching:
        # Random over all images
        batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
        batch = torch.transpose(batch, 0, 1)
        batch_rays, target_s = batch[:2], batch[2]

        i_batch += N_rand
        if i_batch >= rays_rgb.shape[0]:
            print("Shuffle data after an epoch!")
            rand_idx = torch.randperm(rays_rgb.shape[0])
            rays_rgb = rays_rgb[rand_idx]
            i_batch = 0

    else:
        # Random from one image
        img_i = np.random.choice(i_train)
        target = images[img_i]
        target = torch.Tensor(target).to(device)
        pose = poses[img_i, :3,:4]

        if N_rand is not None:
            rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

            if i < args.precrop_iters:
                dH = int(H//2 * args.precrop_frac)
                dW = int(W//2 * args.precrop_frac)
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                        torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                    ), -1)
                if i == start:
                    print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
            else:
                coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

            coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
            select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            batch_rays = torch.stack([rays_o, rays_d], 0)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

    return batch_rays, target_s, rays_rgb, i_batch

def save_checkpoints(args, coarse_model, fine_model, optimizer, global_step, i):
    basedir = args.basedir
    expname = args.expname

    # Logs chckpoints
    path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
    torch.save({
        'global_step': global_step,
        'coarse_model_state_dict': coarse_model.state_dict(),
        'fine_model_state_dict': fine_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print('Saved checkpoints at', path)

def render_training_video(args, render_poses, hwf, K, render_kwargs_test, i):
        basedir = args.basedir
        expname = args.expname
        # Turn on testing mode
        with torch.no_grad():
            rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
        print('Done, saving', rgbs.shape, disps.shape)
        moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
        imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        # if self.args.use_viewdirs:
        #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
        #     with torch.no_grad():
        #         rgbs_still, _ = render_path(render_poses, hwf, self.args.chunk, render_kwargs_test)
        #     render_kwargs_test['c2w_staticcam'] = None
        #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

def render_test_poses(args, images, poses, hwf, K, render_kwargs_test, i_split, i):
    basedir = args.basedir
    expname = args.expname
    i_train, i_val, i_test = i_split

    testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
    os.makedirs(testsavedir, exist_ok=True)
    print('test poses shape', poses[i_test].shape)
    with torch.no_grad():
        render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
    print('Saved test set')

def print_statistics(args, loss, psnr, i, tb_writer=None):
    tqdm.tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

    # Add scalars to tensorboard if using
    if tb_writer is not None:
        tb_writer.add_scalar("Test/Loss", loss, i)
        tb_writer.add_scalar("Test/PSNR", psnr, i)
    """
        print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
        print('iter time {:.05f}'.format(dt))

        with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
            tf.contrib.summary.scalar('loss', loss)
            tf.contrib.summary.scalar('psnr', psnr)
            tf.contrib.summary.histogram('tran', trans)
            if args.N_importance > 0:
                tf.contrib.summary.scalar('psnr0', psnr0)

        if i%args.i_img==0:

            # Log a rendered validation view to Tensorboard
            img_i=np.random.choice(i_val)
            target = images[img_i]
            pose = poses[img_i, :3,:4]
            with torch.no_grad():
                rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                    **render_kwargs_test)

            psnr = mse2psnr(img2mse(rgb, target))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                tf.contrib.summary.scalar('psnr_holdout', psnr)
                tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])

            if args.N_importance > 0:

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                    tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                    tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
    """
