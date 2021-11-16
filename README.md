# MSL-IPRL NeRF Stable Repository V1


[NeRF](http://www.matthewtancik.com/nerf) (Neural Radiance Fields) is a method that achieves state-of-the-art results for synthesizing novel views of complex scenes. The purpose of this repository is to create a more object-oriented, intelligible, and minimal codebase for general-purpose use (i.e., training NeRFs, performing pose estimation, etc.). For future works built on top of this project, please create a feature branch. 

## Installation

It is recommended to install this within a virtual environment. For Conda environments, you can
install the dependencies as follows:

```
git clone https://github.com/stanford-iprl-lab/nerf-shared.git
cd nerf-shared
conda env create -n nerf-shared -f environment.yml
```

If you run into dependency issues, try just doing a ```pip install```. For packages like lietorch or torchsearchsorted (this dependency will come in Stable V2), please go to [LieTorch](https://github.com/princeton-vl/lietorch) and [searchsorted](https://github.com/aliutkus/torchsearchsorted) and follow their installiation instructions.

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
  
</details>

## How To Run?

### Quick Start

Download data for two example datasets: `lego` and `fern`
```
bash download_example_data.sh
```

To train a low-res `lego` NeRF:
```
python run_nerf.py --config configs/lego.txt
```
After training for 100k iterations (~4 hours on a single 2080 Ti), you can find the following video at `logs/lego_test/lego_test_spiral_100000_rgb.mp4`.

![](https://user-images.githubusercontent.com/7057863/78473103-9353b300-7770-11ea-98ed-6ba2d877b62c.gif)

---

To train a low-res `fern` NeRF:
```
python run_nerf.py --config configs/fern.txt
```
After training for 200k iterations (~8 hours on a single 2080 Ti), you can find the following video at `logs/fern_test/fern_test_spiral_200000_rgb.mp4` and `logs/fern_test/fern_test_spiral_200000_disp.mp4`

![](https://user-images.githubusercontent.com/7057863/78473081-58ea1600-7770-11ea-92ce-2bbf6a3f9add.gif)

---

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

---

To train NeRF on different datasets: 

```
python run_nerf.py --config configs/{DATASET}.txt
```

replace `{DATASET}` with `trex` | `horns` | `flower` | `fortress` | `lego` | etc.

---

To test NeRF trained on different datasets: 

```
python run_nerf.py --config configs/{DATASET}.txt --render_only
```

replace `{DATASET}` with `trex` | `horns` | `flower` | `fortress` | `lego` | etc.


### Pre-trained Models

You can download the pre-trained models [here](https://drive.google.com/drive/folders/1uq0OSpyCuSIOBbT12L3pLiEnMoKIwMDo). Place the downloaded directory in `./logs` in order to test it later. See the following directory structure for an example:

```
├── logs 
│   ├── fern_test
│   ├── flower_test  # downloaded logs
│   ├── trex_test    # downloaded logs
```

### Reproducibility 

Tests that ensure the results of all functions and training loop match the official implentation are contained in a different branch `reproduce`. One can check it out and run the tests:
```
git checkout reproduce
py.test
```

## Method

[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](http://tancik.com/nerf)  
 [Ben Mildenhall](https://people.eecs.berkeley.edu/~bmild/)\*<sup>1</sup>,
 [Pratul P. Srinivasan](https://people.eecs.berkeley.edu/~pratul/)\*<sup>1</sup>,
 [Matthew Tancik](http://tancik.com/)\*<sup>1</sup>,
 [Jonathan T. Barron](http://jonbarron.info/)<sup>2</sup>,
 [Ravi Ramamoorthi](http://cseweb.ucsd.edu/~ravir/)<sup>3</sup>,
 [Ren Ng](https://www2.eecs.berkeley.edu/Faculty/Homepages/yirenng.html)<sup>1</sup> <br>
 <sup>1</sup>UC Berkeley, <sup>2</sup>Google Research, <sup>3</sup>UC San Diego  
  \*denotes equal contribution  
  
<img src='imgs/pipeline.jpg'/>

> A neural radiance field is a simple fully connected network (weights are ~5MB) trained to reproduce input views of a single scene using a rendering loss. The network directly maps from spatial location and viewing direction (5D input) to color and opacity (4D output), acting as the "volume" so we can use volume rendering to differentiably render new views


## Citation
Kudos to the authors for their amazing results:
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

However, if you find this implementation or pre-trained models helpful, please consider to cite:
```
@misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}
```
