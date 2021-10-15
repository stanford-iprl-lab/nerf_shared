import numpy as np
import time
import torch

from torchtyping import TensorDetail, TensorType
from typeguard import typechecked

from tqdm import tqdm, trange



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

from load_nerf import config_parser

def run():
    parser = config_parser()
    args = parser.parse_args()

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.empty_cache()

    run()
