import numpy as np
import time
import torch
from torchtyping import TensorDetail, TensorType
from typeguard import typechecked
from tqdm import tqdm, trange

import cv2
import configargparse
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import *
from render_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)


"""
example usage:
    - install any missing requirements from https://github.com/salykovaa/inerf/blob/main/requirements.txt
    - cd to folder containing demo_est_rel_pose.py
    - python demo_est_rel_pose.py --config relative_pose_estimation_configs/teddy_bear.txt
"""

def estimate_relative_pose(coarse_model, fine_model, renderer, sensor_image, start_pose, K, general_args, extra_arg_dict, obs_img_pose=None, obs_img=None):
    b_print_comparison_metrics = obs_img_pose is not None
    b_generate_overlaid_images = b_print_comparison_metrics and obs_img is not None

    obs_img_noised = sensor_image
    W_obs = sensor_image.shape[0]
    H_obs = sensor_image.shape[1]

     # find points of interest of the observed image
    POI = find_POI(obs_img_noised, DEBUG)  # xy pixel coordinates of points of interest (N x 2)
    obs_img_noised = (np.array(obs_img_noised) / 255.).astype(np.float32)

    # create meshgrid from the observed image
    coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, W_obs - 1, W_obs), np.linspace(0, H_obs - 1, H_obs)), -1), dtype=int)

    # create sampling mask for interest region sampling strategy
    interest_regions = np.zeros((H_obs, W_obs, ), dtype=np.uint8)
    interest_regions[POI[:,1], POI[:,0]] = 1
    I = extra_arg_dict['dil_iter']
    interest_regions = cv2.dilate(interest_regions, np.ones((extra_arg_dict['kernel_size'], extra_arg_dict['kernel_size']), np.uint8), iterations=I)
    interest_regions = np.array(interest_regions, dtype=bool)
    interest_regions = coords[interest_regions]

    # not_POI contains all points except of POI
    coords = coords.reshape(H_obs * W_obs, 2)
    not_POI = set(tuple(point) for point in coords) - set(tuple(point) for point in POI)
    not_POI = np.array([list(point) for point in not_POI]).astype(int)


    # Create pose transformation model
    start_pose = torch.Tensor(start_pose).to(device)
    cam_transf = camera_transf().to(device)
    optimizer = torch.optim.Adam(params=cam_transf.parameters(), lr=extra_arg_dict['lrate'], betas=(0.9, 0.999))

    # calculate angles and translation of the observed image's pose
    if b_print_comparison_metrics:
        phi_ref = np.arctan2(obs_img_pose[1,0], obs_img_pose[0,0])*180/np.pi
        theta_ref = np.arctan2(-obs_img_pose[2, 0], np.sqrt(obs_img_pose[2, 1]**2 + obs_img_pose[2, 2]**2))*180/np.pi
        psi_ref = np.arctan2(obs_img_pose[2, 1], obs_img_pose[2, 2])*180/np.pi
        translation_ref = np.sqrt(obs_img_pose[0,3]**2 + obs_img_pose[1,3]**2 + obs_img_pose[2,3]**2)

    testsavedir = os.path.join(extra_arg_dict['output_dir'], extra_arg_dict['model_name'])
    os.makedirs(testsavedir, exist_ok=True)

    # imgs - array with images are used to create a video of optimization process
    if b_generate_overlaid_images:
        imgs = []

    for k in range(300):
        rand_inds = np.random.choice(interest_regions.shape[0], size=extra_arg_dict['batch_size'], replace=False)
        batch = interest_regions[rand_inds]

        target_s = obs_img_noised[batch[:, 1], batch[:, 0]]
        target_s = torch.Tensor(target_s).to(device)
        pose = cam_transf(start_pose)

        rays_o, rays_d = get_rays(H_obs, W_obs, K, pose)  # (H, W, 3), (H, W, 3)
        rays_o = rays_o[batch[:, 1], batch[:, 0]]  # (N_rand, 3)
        rays_d = rays_d[batch[:, 1], batch[:, 0]]
        batch_rays = torch.stack([rays_o, rays_d], 0)

        rgb, _, _, _ = renderer.render_from_rays(H_obs,
                                                 W_obs,
                                                 K,
                                                 chunk=general_args.chunk,
                                                 rays=batch_rays,
                                                 coarse_model=coarse_model,
                                                 fine_model=fine_model,
                                                 retraw=True)
        optimizer.zero_grad()
        loss = img2mse(rgb, target_s)
        loss.backward()
        optimizer.step()

        new_lrate = extra_arg_dict['lrate'] * (0.8 ** ((k + 1) / 100))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        # print results periodically
        if b_print_comparison_metrics and ((k + 1) % 20 == 0 or k == 0):
            print('Step: ', k)
            print('Loss: ', loss)

            with torch.no_grad():
                pose_dummy = pose.cpu().detach().numpy()
                # calculate angles and translation of the optimized pose
                phi = np.arctan2(pose_dummy[1, 0], pose_dummy[0, 0]) * 180 / np.pi
                theta = np.arctan2(-pose_dummy[2, 0], np.sqrt(pose_dummy[2, 1] ** 2 + pose_dummy[2, 2] ** 2)) * 180 / np.pi
                psi = np.arctan2(pose_dummy[2, 1], pose_dummy[2, 2]) * 180 / np.pi
                translation = np.sqrt(pose_dummy[0,3]**2 + pose_dummy[1,3]**2 + pose_dummy[2,3]**2)
                #translation = pose_dummy[2, 3]
                # calculate error between optimized and observed pose
                phi_error = abs(phi_ref - phi) if abs(phi_ref - phi)<300 else abs(abs(phi_ref - phi)-360)
                theta_error = abs(theta_ref - theta) if abs(theta_ref - theta)<300 else abs(abs(theta_ref - theta)-360)
                psi_error = abs(psi_ref - psi) if abs(psi_ref - psi)<300 else abs(abs(psi_ref - psi)-360)
                rot_error = phi_error + theta_error + psi_error
                translation_error = abs(translation_ref - translation)
                print('Rotation error: ', rot_error)
                print('Translation error: ', translation_error)
                print('-----------------------------------')
        '''
            if b_generate_overlaid_images:
                with torch.no_grad():
                    rgb, _, _, _ = renderer.render_from_pose(H_obs,
                                                             W_obs,
                                                             K,
                                                             chunk=general_args.chunk,
                                                             c2w=pose[:3, :4],
                                                             coarse_model=coarse_model,
                                                             fine_model=fine_model,
                                                             retraw=True)
                    rgb = rgb.cpu().detach().numpy()
                    rgb8 = to8b(rgb)
                    ref = to8b(obs_img)
                    filename = os.path.join(testsavedir, str(k)+'.png')
                    dst = cv2.addWeighted(rgb8, 0.7, ref, 0.3, 0)
                    imageio.imwrite(filename, dst)
                    imgs.append(dst)

    if b_generate_overlaid_images:
        imageio.mimwrite(os.path.join(testsavedir, 'video.gif'), imgs, fps=8) #quality = 8 for mp4 format
    '''
    print("Done with main relative_pose_estimation loop")


def find_POI(img_rgb, DEBUG=False): # img - RGB image in range 0...255
    img = np.copy(img_rgb)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints = sift.detect(img_gray, None)
    if DEBUG:
        img = cv2.drawKeypoints(img_gray, keypoints, img)
        show_img("Detected points", img)
    xy = [keypoint.pt for keypoint in keypoints]
    xy = np.array(xy).astype(int)
    # Remove duplicate points
    xy_set = set(tuple(point) for point in xy)
    xy = np.array([list(point) for point in xy_set]).astype(int)
    return xy # pixel coordinates

rot_psi = lambda phi: np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1]])

rot_theta = lambda th: np.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1]])

rot_phi = lambda psi: np.array([
        [np.cos(psi), -np.sin(psi), 0, 0],
        [np.sin(psi), np.cos(psi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

trans_t = lambda t: np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1]])

def vec2ss_matrix(vector):  # vector to skewsym. matrix

    ss_matrix = torch.zeros((3,3))
    ss_matrix[0, 1] = -vector[2]
    ss_matrix[0, 2] = vector[1]
    ss_matrix[1, 0] = vector[2]
    ss_matrix[1, 2] = -vector[0]
    ss_matrix[2, 0] = -vector[1]
    ss_matrix[2, 1] = vector[0]

    return ss_matrix


class camera_transf(nn.Module):
    def __init__(self):
        super(camera_transf, self).__init__()
        self.w = nn.Parameter(torch.normal(0., 1e-6, size=(3,)))
        self.v = nn.Parameter(torch.normal(0., 1e-6, size=(3,)))
        self.theta = nn.Parameter(torch.normal(0., 1e-6, size=()))

    def forward(self, x):
        exp_i = torch.zeros((4,4))
        w_skewsym = vec2ss_matrix(self.w)
        v_skewsym = vec2ss_matrix(self.v)
        exp_i[:3, :3] = torch.eye(3) + torch.sin(self.theta) * w_skewsym + (1 - torch.cos(self.theta)) * torch.matmul(w_skewsym, w_skewsym)
        exp_i[:3, 3] = torch.matmul(torch.eye(3) * self.theta + (1 - torch.cos(self.theta)) * w_skewsym + (self.theta - torch.sin(self.theta)) * torch.matmul(w_skewsym, w_skewsym), self.v)
        exp_i[3, 3] = 1.
        T_i = torch.matmul(exp_i, x)
        return T_i


def extra_config_parser():
    '''
    Take in the minimum amount required to load in a blender model and run relative_pose_estimation
    '''
    parser = configargparse.ArgumentParser()

    # Path arguments
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--output_dir", type=str, default='./output/',
                        help='where to store output images/videos')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/nerf_synthetic/',
                        help='path to folder with synthetic or llff data')

    # relative_pose_estimation arguments
    parser.add_argument("--dil_iter", type=int, default=3,
                        help='Number of iterations of dilation process')
    parser.add_argument("--kernel_size", type=int, default=5,
                        help='Kernel size for dilation')
    parser.add_argument("--batch_size", type=int, default=512,
                        help='Number of sampled rays per gradient step')
    parser.add_argument("--lrate_relative_pose_estimation", type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument("--sampling_strategy", type=str, default='random',
                        help='options: random / interest_point / interest_region')
    # parameters to define initial pose
    parser.add_argument("--delta_psi", type=float, default=0.0,
                        help='Rotate camera around x axis')
    parser.add_argument("--delta_phi", type=float, default=0.0,
                        help='Rotate camera around z axis')
    parser.add_argument("--delta_theta", type=float, default=0.0,
                        help='Rotate camera around y axis')
    parser.add_argument("--delta_t", type=float, default=0.0,
                        help='translation of camera (negative = zoom in)')
    # apply noise to observed image
    parser.add_argument("--noise", type=str, default='None',
                        help='options: gauss / salt / pepper / sp / poisson')
    parser.add_argument("--sigma", type=float, default=0.01,
                        help='var = sigma^2 of applied noise (variance = std)')
    parser.add_argument("--amount", type=float, default=0.05,
                        help='proportion of image pixels to replace with noise (used in ‘salt’, ‘pepper’, and ‘s&p)')
    parser.add_argument("--delta_brightness", type=float, default=0.0,
                        help='reduce/increase brightness of the observed image, value is in [-1...1]')


    # NeRF model args (needed to load in the nerf)
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", default='False', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", default='False', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='blender', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')


    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')
    return parser



if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.empty_cache()
    parser = extra_config_parser()
    args = parser.parse_args()

    ##################### LOAD IN NERF #####################
    #Loads dataset info like images and Ground Truth poses and camera intrinsics
    images, poses, _, hwf, i_split, K, bds_dict = utils.load_datasets(args)

    # Train, val, test split
    _, _, i_test = i_split

    # Resolution (H, W) and focal length
    _, _, focal = hwf

    # Create coarse/fine NeRF models.
    coarse_model, fine_model = utils.create_nerf_models(args)

    # Create optimizer for trainable params.
    optimizer = utils.get_optimizer(coarse_model, fine_model, args)

    # Load any available checkpoints.
    start = utils.load_checkpoint(coarse_model, fine_model, optimizer, args, b_load_ckpnt_as_trainable=False)

    renderer = utils.get_renderer(args, bds_dict)
    ###############################################################

    relative_pose_image_idx = 0  # Take first image as one to use
    obs_img = images[i_test[relative_pose_image_idx]]
    sensor_image = np.asarray(obs_img*255, dtype=np.uint8)
    gt_pose = poses[i_test[relative_pose_image_idx]]

    start_pose = trans_t(args.delta_t) @ rot_phi(args.delta_phi/180.*np.pi) @ rot_theta(args.delta_theta/180.*np.pi) @ rot_psi(args.delta_psi/180.*np.pi) @ gt_pose

    extra_arg_dict = {
        'model_name': args.expname,
        'output_dir': os.path.dirname(os.path.abspath(__file__)) + "/relative_pose_estimation_output/" + args.expname + "/",
        'dil_iter': args.dil_iter,
        'batch_size': args.batch_size,
        'kernel_size': args.kernel_size,
        'lrate': args.lrate_relative_pose_estimation,
    }
    t = time.time()
    estimate_relative_pose(coarse_model=coarse_model, 
                           fine_model=fine_model, 
                           renderer=renderer, 
                           sensor_image=sensor_image, 
                           start_pose=start_pose, 
                           K=K, 
                           general_args=args, 
                           extra_arg_dict=extra_arg_dict,
                           obs_img_pose=gt_pose,
                           obs_img=obs_img)
    print('Elapsed', time.time() - t)

