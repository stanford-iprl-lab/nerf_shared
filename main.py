import time
import numpy as np
import torch
from tqdm import tqdm, trange

# TODO(pculbert): Refactor to import just module.
from utils import *
import utils
from config_parser import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

def run():
    parser = config_parser()
    args = parser.parse_args()

    if args.training is True:
        #Loads dataset info like images and Ground Truth poses and camera intrinsics
        images, poses, render_poses, hwf, i_split, K, bds_dict = load_datasets(args)

        # Train, val, test split
        i_train, i_val, i_test = i_split

        # Resolution (H, W) and focal length
        H, W, focal = hwf

        # Copy config file to log file
        copy_log_dir(args)

        # Create coarse/fine NeRF models.
        coarse_model, fine_model = utils.create_nerf_models(args)

        # Create optimizer for trainable params.
        optimizer = utils.get_optimizer(coarse_model, fine_model, args)

        # Load any available checkpoints.
        start = utils.load_checkpoint(coarse_model, fine_model, optimizer, args)

        renderer_train, renderer_test = get_renderers(coarse_model, fine_model,
                                                      args, bds_dict)

        global_step = start

        # Move testing data to GPU
        render_poses = torch.Tensor(render_poses).to(device)

        # Batch the training data
        images, poses, rays_rgb, use_batching, N_rand, i_batch = batch_training_data(args, poses, hwf, K, images, i_train)

        N_iters = 200000 + 1
        print('Begin')
        print('TRAIN views are', i_train)
        print('TEST views are', i_test)
        print('VAL views are', i_val)

        start = start + 1
        for i in trange(start, N_iters):
            time0 = time.time()

            # Randomly select a batch of rays across images, or randomly sample from a single image per iteration
            # determined by boolean use_batching
            batch_rays, target_s, rays_rgb, i_batch = sample_random_ray_batch(args, images, poses,
                                                    rays_rgb, N_rand, use_batching, i_batch, i_train,
                                                    hwf, K, start, i)

            #####  Core optimization loop  #####
            # Calls method of training renderer
            rgb, _, _, extras = renderer_train.render_from_rays(H,
                                                                W,
                                                                K,
                                                                chunk=args.chunk,
                                                                rays=batch_rays,
                                                                coarse_model=coarse_model,
                                                                fine_model=fine_model,
                                                                retraw=True)

            optimizer.zero_grad()

            #Mean squared error between rendered ray RGB vs. Ground Truth RGB using the fine model
            img_loss = img2mse(rgb, target_s)
            trans = extras['raw'][...,-1]
            loss = img_loss
            psnr = mse2psnr(img_loss)

            # If using both the coarse and fine model,
            if 'rgb0' in extras:
                # MSE loss between rendered coarse model and GT RGB
                img_loss0 = img2mse(extras['rgb0'], target_s)

                #Add the coarse and fine reconstruction loss together
                loss = loss + img_loss0
                psnr0 = mse2psnr(img_loss0)

            # TODO(pculbert, chengine): Debug optimization; performance does not match
            # original implementation.
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
            # Periodically saves weights
            if i%args.i_weights==0:
                save_checkpoints(args, coarse_model, fine_model, optimizer, global_step, i)
            '''
            # Constructs a panoramic video of a camera within the NeRF scene
            if i%args.i_video==0 and i > 0:
                render_training_video(args, render_poses, hwf, K, render_kwargs_test, i)

            # Renders out the test poses to visually evaluate NeRF quality
            if i%args.i_testset==0 and i > 0:
                render_test_poses(args, images, poses, hwf, K, render_kwargs_test, i_split, i)
            '''
            #Displays loss and PSNR (Peak signal to noise ratio) of the fine reconstruction loss
            if i%args.i_print==0:
                print_statistics(args, loss, psnr, i)

            global_step += 1

    else:
        ### Define Custom Functionality Here
        pass

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.empty_cache()

    run()
