
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torchtyping import TensorDetail, TensorType
from typeguard import typechecked
import pdb

class NerfMLP(nn.Module):
    """ Implements the raw NeRF MLP """
    def __init__(self, cfg, D=8, W=256, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NerfMLP, self).__init__()
        self.use_viewdirs = use_viewdirs
        self.D = D
        self.W = W
        self.skips = skips

        self.embedding_layer_pos = MyEmbeddingLayer(num_freqs=cfg.multires, max_freq_log2=cfg.multires-1)

        self.input_ch = self.embedding_layer_pos.dim_out

        # tmp = self.embedding_layer.forward(torch.tensor([[1,2,3], [1,1,1]]))
        if self.use_viewdirs:
            self.embedding_layer_view = MyEmbeddingLayer(num_freqs=cfg.multires_views, max_freq_log2=cfg.multires_views-1)
            self.input_ch_views = self.embedding_layer_view.dim_out


        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(self.input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)


    @typechecked
    def forward(self, points: TensorType["dims":..., 3], view_direction: TensorType["dims":..., 3]) -> TensorType["dims":...,4]:
        embedded_position_input = self.embedding_layer_pos.forward(points)
        h = embedded_position_input
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([embedded_position_input, h], -1)

        if self.use_viewdirs:
            embedded_view_dir_input = self.embedding_layer_view.forward(view_direction)
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, embedded_view_dir_input], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

class Nerf(nn.Module):
    """ Wrapper around a coarse and fine NeRF network """
    def __init__(self, cfg):
        super(Nerf, self).__init__()
        self.cfg = cfg  # store the config vars in the nerf class
        output_ch = 5 if cfg.N_importance > 0 else 4
        skips = [4]
        self.coarse_mlp = NerfMLP(cfg=cfg, D=cfg.netdepth, W=cfg.netwidth, output_ch=output_ch, 
                                  skips=skips, use_viewdirs=cfg.use_viewdirs)
        self.fine_mlp = NerfMLP(cfg=cfg, D=cfg.netdepth_fine, W=cfg.netwidth_fine, output_ch=output_ch, 
                                skips=skips, use_viewdirs=cfg.use_viewdirs)

    @typechecked
    def forward(self, points: TensorType["dims":..., 3], view_direction: TensorType["dims":..., 3], fine_network = True) -> TensorType["dims":...,4]:
        raise NotImplemented

    @typechecked
    def density(self, points: TensorType["dims":..., 3], fine_network = True) -> TensorType["dims":...]:
        view_dir = torch.ones_like(points)
        output = self.forward(points, view_dir, fine_network)
        return output[..., -1]

class CompositeNerf(Nerf):
    def __init__(self, nerf_list, pose_list, cfg):
        self.nerf_list = nerf_list
        self.pose_list = nerf_list

        raise NotImplemented

    @typechecked
    def forward(self, points: TensorType["dims":..., 3], view_direction: TensorType["dims":..., 3], fine_network = True) -> TensorType["dims":...,4]:
        raise NotImplemented

        for nerf, pose in zip(self.nerf_list, self.pose_list):
            nerf.forward(points, view_direction)
        # alpha combine

    @typechecked
    def density(self, points: TensorType["dims":..., 3]) -> TensorType["dims":...]:
        raise NotImplemented


class MyEmbeddingLayer(nn.Module):
    """ Custom fourier embedding layer """
    def __init__(self, num_freqs, max_freq_log2, dim_in=3, b_optimize_freqs=False):
        super().__init__()
        self.dim_in = dim_in
        self.num_weights = num_freqs  # note: no x2 because cos & sin for each element will use same freq for each element of the dim_in
        self.dim_out = dim_in + 2*self.num_weights*dim_in # first 3 outputs are the input. Then for each element of the input, a cos & sin are evaluated at the given frequency
        self.weights = nn.Parameter(2*np.pi * 2.**torch.linspace(0., max_freq_log2, steps=num_freqs), requires_grad=b_optimize_freqs)

    def forward(self, x):
        """
        this assumes x is a batch_size x 3 vector (the 3 dim is for either postion or unit direction vector)
        """
        batch_dim, dim_in = x.shape
        assert(dim_in == self.dim_in)  # make sure whats being passed in is expected length

        x_weighted = x.reshape(-1, 3, 1) * self.weights.reshape(1, 1, -1)
        return torch.cat( (x.reshape(batch_dim, dim_in, 1), torch.sin(x_weighted), torch.cos(x_weighted) ), dim=-1).reshape(batch_dim, -1)
        
