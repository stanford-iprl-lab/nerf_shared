import os, sys, pdb
import numpy as np
import imageio
import json
import random
import time
import torch

from torchtyping import TensorDetail, TensorType
from typeguard import typechecked

from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *
from nerf_core import *
from nerf import *

from load_blender import load_blender_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

from load_nerf import config_parser


def adam_tmp():
    parser = config_parser()
    args = parser.parse_args()

    my_nerf = Nerf(args)
    pdb.set_trace()

    s = "train"
    with open(os.path.join(args.datadir, 'transforms_{}.json'.format(s)), 'r') as fp:
        meta = json.load(fp)

    frame = meta['frames'][0]
    fname = os.path.join(args.datadir, frame['file_path'] + '.png')
    img = imageio.imread(fname)

    img = (np.array(img) / 255.).astype(np.float32) # keep all 4 channels (RGBA)

    H, W = img.shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    # if args.half_res:
    #     H = H//2
    #     W = W//2
    #     focal = focal/2.

    #     imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
    #     for i, img in enumerate(imgs):
    #         imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    #     imgs = imgs_half_res
    #     # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    # Cast intrinsics to right types
    H, W = int(H), int(W)
    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])
    pdb.set_trace()



if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.empty_cache()

    adam_tmp()
