# MSL-IPRL NeRF Stable Repository V1


[NeRF](http://www.matthewtancik.com/nerf) (Neural Radiance Fields) is a method that achieves state-of-the-art results for synthesizing novel views of complex scenes. The purpose of this repository is to create a more object-oriented, intelligible, and minimal codebase for general-purpose use (i.e., training NeRFs, performing pose estimation, etc.). For future works built on top of this project, please create a feature branch. 

<details>
  <summary> Info </summary>
  
  ## Table of Contents
  - Installation
  - Training
  - Examples
  - Creating Blender Datasets
  - Present and Future Extensions of NeRFs

## Installation

It is recommended to install this within a virtual environment. For Conda environments, you can
install the dependencies as follows:

```
git clone https://github.com/stanford-iprl-lab/nerf-shared.git
cd nerf-shared
conda env create -n nerf-shared -f environment.yml
```

If you run into dependency issues, try just doing a ```pip install```. For packages like lietorch or torchsearchsorted (this dependency will come in Stable V2), please go to [LieTorch](https://github.com/princeton-vl/lietorch) and [searchsorted](https://github.com/aliutkus/torchsearchsorted) and follow their installiation instructions.

Here is a full list of dependencies (WIP, not up to date):
<details>
  <summary> Dependencies (click to expand) </summary>
  
  ## Dependencies
  - PyTorch 1.4
  - matplotlib
  - numpy
  - imageio
  - imageio-ffmpeg
  - configargparse

The LLFF data loader requires ImageMagick.

You will also need the [LLFF code](http://github.com/fyusion/llff) (and COLMAP) set up to compute poses if you want to run on your own real data.

Typically, we've just used Blender datasets for ground-truth images and poses. Please see the section below on how to create a Blender dataset that NeRFs can train on.

</details>

## Training

### Quick Start

To get started immediately, download data for two example datasets: `lego` and `fern`. We will eventually be providing our own datasets.
```
bash download_example_data.sh
```

To train a `lego` NeRF:
```
python main.py --config configs/lego.txt
```
Every 10k iterations, the log files will be updated to include test renders (`logs/lego_test/testset_X`) and zipped network weights at `logs/lego_test/X.tar` where `X` denotes the iteration number.

### More Datasets
To play with other scenes presented in the paper, download the data [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1). Place the downloaded dataset according to the following directory structure:
```
├── configs                                                                                                       
│   ├── ...                                                                                     
│                                                                                               
├── data                                                                                                                                                                                                       
│   ├── nerf_llff_data                                                                                                  
│   │   └── fern                                                                                                                             
│   │   └── flower  # downloaded llff dataset                                                                                  
│   │   └── horns   # downloaded llff dataset
|   |   └── ...
|   ├── nerf_synthetic
|   |   └── lego
|   |   └── ship    # downloaded synthetic dataset
|   |   └── ...
```

### File Structure
It is best to understand a bit more about how the data and outputs are organized. In the root directory, there are 5 Python files associated with NeRFs, and 4 Python files associated with loading in datasets (Blender, LINEMOD, LLFF, DeepVoxels).

```
-main.py
-config_parser.py
-nerf.py
-render_utils.py
-utils.py
```

`main.py` contains a minimal training script that calls functions in `config_parser.py` and `utils.py`.

`config_parser.py` contains just the configuration file parser. It is its own separate folder for users to succintly see which arguments are being passed in and what their default values are. It is highly recommended to look over this file and your data's config file to understand which parameters matter to you.

`nerf.py` contains the NeRF class, primarily the neural network portion of NeRFs. In it is an Embedder class for the embedding layer, and the larger NeRF class for combining the MLP with the embedding layer, as well as handling batching of points.

`render_utils.py` contains the Renderer class. To make this class separate from the NeRF class, the methods of Renderer require the user to pass in the NeRF models. Other than that, it performs the say ray-tracing method using quadrature as in the original NeRF implementation. It also automatically handles batching of rays.

`utils.py` contains stuff like getting the optimizer and renderer, ray-generation, sampling, loading checkpoints, batching training data, and logging. It is recommended to understand how the optimizer, renderer, and create_nerf functions in this file work, as well as how they are called in `main.py`

The rest are dataset specific. NOTE: If you are using Blender as your dataset, make sure you change the near and far bounds accordingly in `load_blender.py`! In the future, the near and far bounds will be incorporated either into the data itself, or in the config file.

### Logs
In `logs` folder, a folder will automatically be generated storing your rendered test images and neural network weights, along with some text files indicating the config used to train the model (Very important when sharing models with others!).

### Configs
In `configs` folder contains the config file used to train a particular NeRF. It is highly recommended to take a look at the example config files in order to understand how your model will behave during training. Some parameters that are particularly important if you decide to copy and paste the example config files are `expname, datadir, dataset_type, white_bkgd, half_res` which determine the experiment's name and corresponding name of the log file in `logs`, the directory in which you stored the training data, where you got your dataset from (e.g., Blender), whether or not the NeRF should be trained on images with white backgrounds, and whether you want your model to train on training images at half resolution. 

NOTE: `white_bkgd` primarily applies to Blender datasets that have transparent images so that setting `white_bkgd=True` will allow the NeRF to render properly. If your images have solid background colors, set this parameter to False.

NOTE: Setting `half_res` to True will also cause the NeRF model to render at half resolution.

### Data
The `data` folder is separated into real-life training data `nerf_llff_data` and synthetic (e.g., Blender) data in `nerf_synthetic`. However, the structure within both is the same. Within each scenes folder, there MUST HAVE 3 folders `test`, `train`, and `val` containing the corresponding images, and their respective ground truth poses under `transforms_....json`. It is recommended to look at the `.json` file to see camera instrinsic parameters that the file should provide beside poses.

## Examples and Other Functionality
In the `examples` folder contain example scripts to perform functionality beyond training, such as pose estimation. Within those folders will be anothe README containing a more in-depth how-to.

## Blender Specific
We will eventually provide a script where you can generate these three folders and pose files after loading a scene or object into Blender.

In the meantime, the `.json` file is structured as a dictionary:
```
{
  "Far": ...,   #Far Bound
  "Near": ...,  #Near Bound
  "camera_angle_x: ..., #Horizontal FOV
  "frames": ...
}
```
where `"frames"` is a list of dictionaries (one for each image) containing the file path to the image and its corresponding ground-truth pose as follows:

```
{
  "transform_matrix": ...,   #Pose in SE3
  "file_path": "./{test,train,val}/img_name"  #File path
}
```

### Misc
To train NeRF on different datasets: 

```
python main.py --config configs/{DATASET}.txt
```

replace `{DATASET}` with your experiment name.

### Pre-trained Models
We intend to provide some pre-trained models in the future. Stay tuned!


## Future Direction
Contained in the feature branches are the following extensions:
- Navigation (Planning, Estimation, and Control) within NeRFs
- Distributed NeRF training 
- Speed ups to NeRF to make it real-time for robotics applications
- Manipulation and NeRFs

## Citation
Please cite the following works if you use anything from this repository:
```
@misc{mildenhall2020nerf,
    title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
    author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
    year={2020},
    eprint={2003.08934},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

The Stable V1 repository was built off of this PyTorch implementation of NeRF, so please cite:
```
@misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}
```

Eventually Stable V2 and beyond will be built off of this faster implementation, so then please cite:
```
@misc{placeholder,
  title={placeholder},
  author={placeholder},
  howpublished={placeholder},
  year={placeholder}
}