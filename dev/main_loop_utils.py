#TODO: Import only what's necessary
from render_utils import *
from nerf_struct import *
from utils import *
from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data

import torch
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """

    '''
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    '''

    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 output_ch=output_ch, skips=skips,
                 use_viewdirs=args.use_viewdirs,
                 multires=args.multires, multires_views=args.multires_views,
                 i_embed=args.i_embed).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          output_ch=output_ch, skips=skips,
                          use_viewdirs=args.use_viewdirs,
                          multires=args.multires, multires_views=args.multires_views,
                          i_embed=args.i_embed).to(device)
        grad_vars += list(model_fine.parameters())

    '''
    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)
    '''

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'], strict=False)
        for param in model.parameters():
            param.requires_grad = False
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
            for param in model_fine.parameters():
                param.requires_grad = False

    ##########################

    render_kwargs_train = {
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'N_samples' : args.N_samples,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'coarse_model' : model,
        'fine_model' : model_fine
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

class Train():
    def __init__(self, args):
        self.args = args

    def load_datasets(self):
        # Load data
        K = None
        if self.args.dataset_type == 'llff':
            images, poses, bds, render_poses, i_test = load_llff_data(self.args.datadir, self.args.factor,
                                                                    recenter=True, bd_factor=.75,
                                                                    spherify=self.args.spherify)
            hwf = poses[0,:3,-1]
            poses = poses[:,:3,:4]
            print('Loaded llff', images.shape, render_poses.shape, hwf, self.args.datadir)
            if not isinstance(i_test, list):
                i_test = [i_test]

            if self.args.llffhold > 0:
                print('Auto LLFF holdout,', self.args.llffhold)
                i_test = np.arange(images.shape[0])[::self.args.llffhold]

            i_val = i_test
            i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

            print('DEFINING BOUNDS')
            if self.args.no_ndc:
                near = np.ndarray.min(bds) * .9
                far = np.ndarray.max(bds) * 1.
                
            else:
                near = 0.
                far = 1.
            print('NEAR FAR', near, far)

        elif self.args.dataset_type == 'blender':
            images, poses, render_poses, hwf, i_split = load_blender_data(self.args.datadir, self.args.half_res, self.args.testskip)
            print('Loaded blender', images.shape, render_poses.shape, hwf, self.args.datadir)
            i_train, i_val, i_test = i_split

            near = 0.
            far = 7.

            if self.args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]

        elif self.args.dataset_type == 'LINEMOD':
            images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(self.args.datadir, self.args.half_res, self.args.testskip)
            print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
            print(f'[CHECK HERE] near: {near}, far: {far}.')
            i_train, i_val, i_test = i_split

            if self.args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]

        elif self.args.dataset_type == 'deepvoxels':

            images, poses, render_poses, hwf, i_split = load_dv_data(scene=self.args.shape,
                                                                    basedir=self.args.datadir,
                                                                    testskip=self.args.testskip)

            print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, self.args.datadir)
            i_train, i_val, i_test = i_split

            hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
            near = hemi_R-1.
            far = hemi_R+1.

        else:
            print('Unknown dataset type', self.args.dataset_type, 'exiting')
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
        
        self.bds_dict = {
        'near' : near,
        'far' : far,
        }

        i_split = i_train, i_val, i_test

        if self.args.render_test:
            render_poses = np.array(poses[i_test])

        return images, poses, render_poses, hwf, i_split, K, self.bds_dict

    def copy_log_dir(self):
        # Create log dir and copy the config file
        basedir = self.args.basedir
        expname = self.args.expname
        os.makedirs(os.path.join(basedir, expname), exist_ok=True)
        f = os.path.join(basedir, expname, 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(self.args)):
                attr = getattr(self.args, arg)
                file.write('{} = {}\n'.format(arg, attr))
        if self.args.config is not None:
            f = os.path.join(basedir, expname, 'config.txt')
            with open(f, 'w') as file:
                file.write(open(self.args.config, 'r').read())

    def create_nerf_models(self):
        # Create nerf model
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(self.args)
        global_step = start

        render_kwargs_train.update(self.bds_dict)
        render_kwargs_test.update(self.bds_dict)

        return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, global_step

    '''
    def render_path(self):
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
    '''

    def batch_training_data(self, poses, hwf, K, images, i_train):
        H, W, _ = hwf

        # Prepare raybatch tensor if batching random rays
        N_rand = self.args.N_rand
        use_batching = not self.args.no_batching
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

        return images, poses, rays_rgb, use_batching, N_rand, i_batch

    def sample_random_ray_batch(self, images, poses, rays_rgb, N_rand, use_batching, i_batch, i_train, hwf, K, start, i):
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

                if i < self.args.precrop_iters:
                    dH = int(H//2 * self.args.precrop_frac)
                    dW = int(W//2 * self.args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {self.args.precrop_iters}")                
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

    def save_checkpoints(self, render_kwargs_train, optimizer, global_step, i):
        basedir = self.args.basedir
        expname = self.args.expname

        # Logs chckpoints
        path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
        torch.save({
            'global_step': global_step,
            'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
            'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        print('Saved checkpoints at', path)

    def render_training_video(self, render_poses, hwf, K, render_kwargs_test, i):
            basedir = self.args.basedir
            expname = self.args.expname
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, self.args.chunk, render_kwargs_test)
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

    def render_test_poses(self, images, poses, hwf, K, render_kwargs_test, i_split, i):
        basedir = self.args.basedir
        expname = self.args.expname
        i_train, i_val, i_test = i_split

        testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', poses[i_test].shape)
        with torch.no_grad():
            render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, self.args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
        print('Saved test set')

    def print_statistics(self, loss, psnr, i):
        tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
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