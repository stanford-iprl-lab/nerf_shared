import numpy as np
import time
import torch

from torchtyping import TensorDetail, TensorType
from typeguard import typechecked

from tqdm import tqdm, trange

from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

from load_nerf import config_parser

def run():
    parser = config_parser()
    args = parser.parse_args()

    train_utils = Train(args)

    if args.custom is False:
        images, poses, render_poses, hwf, i_split, K, bds_dict = train_utils.load_datasets()

        i_train, i_val, i_test = i_split

        H, W, focal = hwf

        train_utils.copy_log_dir()

        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, global_step = train_utils.create_nerf_models()

        # In render_kwargs_train and render_kwargs_test contains the coarse and fine models which take in pos, view direction and
        # outputs rgb-density
        # coarse_nerf = render_kwargs_test['coarse_model']
        # fine_nerf = render_kwargs_test['fine_model']

        # Move testing data to GPU
        render_poses = torch.Tensor(render_poses).to(device)

        images, poses, rays_rgb, use_batching, N_rand, i_batch = train_utils.batch_training_data(poses, hwf, K, images, i_train)
        
        N_iters = 200000 + 1
        print('Begin')
        print('TRAIN views are', i_train)
        print('TEST views are', i_test)
        print('VAL views are', i_val)

        start = start + 1
        for i in trange(start, N_iters):
            time0 = time.time()

            batch_rays, target_s, rays_rgb, i_batch = train_utils.sample_random_ray_batch(images, poses, 
                                                    rays_rgb, N_rand, use_batching, i_batch, i_train, 
                                                    hwf, K, start, i)

            #####  Core optimization loop  #####
            rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                    verbose=i < 10, retraw=True,
                                                    **render_kwargs_train)

            optimizer.zero_grad()
            img_loss = img2mse(rgb, target_s)
            trans = extras['raw'][...,-1]
            loss = img_loss
            psnr = mse2psnr(img_loss)

            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss = loss + img_loss0
                psnr0 = mse2psnr(img_loss0)

            loss.backward()
            optimizer.step()

            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
            ################################

            # Logging
            if i%args.i_weights==0:
                train_utils.save_checkpoint(render_kwargs_train, optimizer, global_step)

            if i%args.i_video==0 and i > 0:
                train_utils.render_training_video(render_poses, hwf, K, render_kwargs_test, i)

            if i%args.i_testset==0 and i > 0:
                train_utils.render_test_poses(images, poses, hwf, K, render_kwargs_test, i_split, i)

            if i%args.i_print==0:
                train_utils.print_statistics(loss, psnr, i)

            global_step += 1
    else:
        ### Define Custom Functionality Here
        pass

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.empty_cache()

    run()
