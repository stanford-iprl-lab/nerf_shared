

import torch

import matplotlib.pyplot as plt

from nerf_core import create_nerf
from render_functions import Renderer

import argparse

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

def get_args():
    settings = {
            "N_importance":128,
            "N_rand":1024,
            "N_samples":64,
            "basedir":'./logs',
            "chunk":32768,
            "config":'configs/playground.txt',
            "datadir":'./data/nerf_synthetic/playground',
            "dataset_type":'blender',
            "expname":'playground_test',
            "factor":8,
            "ft_path":None,
            "half_res":False,
            "i_embed":0,
            "i_img":500,
            "i_print":100,
            "i_testset":50000,
            "i_video":50000,
            "i_weights":10000,
            "lindisp":False,
            "llffhold":8,
            "lrate":0.0005,
            "lrate_decay":500,
            "multires":10,
            "multires_views":4,
            "netchunk":65536,
            "netdepth":8,
            "netdepth_fine":8,
            "netwidth":256,
            "netwidth_fine":256,
            "no_batching":True,
            "no_ndc":False,
            "no_reload":False,
            "perturb":1.0,
            "precrop_frac":0.5,
            "precrop_iters":500,
            "raw_noise_std":0.0,
            "render_factor":0,
            "render_only":False,
            "render_test":False,
            "shape":'greek',
            "spherify":False,
            "testskip":8,
            "use_viewdirs":True,
            "white_bkgd":True
            }

    args = argparse.Namespace(**settings)
    return args

def get_nerf():
    args = get_args()

    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)

    chunk = args.chunk
    hwf = None, None, None
    K = None
    renderer = Renderer(hwf, K, chunk, render_kwargs_train)

    @typechecked
    def nerf(points: TensorType["batch":..., 3]) -> TensorType["batch":...]:
        out_shape = points.shape[:-1]
        points = points.reshape(1, -1, 3)

        mapping = torch.tensor([[1, 0, 0],
                                [0, 0, 1],
                                [0,-1, 0]], dtype=torch.float)

        points = points @ mapping.T

        output = renderer.get_density_from_pt(points)
        return output.reshape(*out_shape)

    return nerf

def main():

    nerf = get_nerf()

    side = 10
    linspace = torch.linspace(-1,1, side)

    # side, side, side, 3
    coods = torch.stack( torch.meshgrid( linspace, linspace, linspace ), dim=-1)
                
    output = nerf(coods)

    im = torch.mean( output, dim=1)
    x_image = torch.mean( coods, dim=1)[...,0]
    y_image = torch.mean( coods, dim=1)[...,2]

    # +z in nerf is -y in blender
    # +y in nerf is +z in blender
    # +x in nerf is +x in blender

    print("happy")
    # exit()

    plt.pcolormesh(x_image, y_image, im)
    plt.show()


if __name__ == "__main__":
    main()



