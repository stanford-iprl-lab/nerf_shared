import numpy as np
import torch
from load_nerf import config_parser
from nerf import *
import pdb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

from load_nerf import config_parser


def run():
    """
    Currently this function loads in parameters as specified from the commandline and creates a nerf object, then quits
    e.g. python3 main_demo.py --config configs/<MY_CONFIG_FILE_NAME_HERE>.txt
    """
    parser = config_parser()
    cfg = parser.parse_args()
    my_nerf = Nerf(cfg)

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.empty_cache()

    run()
    print("Done with Nerf Demo")
