<p align="center">
    <!-- project badges -->
    <a href="https://research.zenseact.com/publications/splatad/"><img src="https://img.shields.io/badge/Project-Page-ffa"/></a>
    <!-- paper badges -->
    <a href="https://arxiv.org/abs/2411.16816">
        <img src='https://img.shields.io/badge/arXiv-Page-aff'>
    </a>
</p>

<div align="center">
<h3 style="font-size:2.0em;">SplatAD</h3>
<h4 style="font-size:1.5em;">
Real-Time Lidar and Camera Rendering with 3D Gaussian Splatting for Autonomous Driving
</h4>
</div>
<div align="center">

<div align="center">
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/imgs/front-fig-stacked.jpg" />
    <img alt="tyro logo" src="docs/_static/imgs/front-fig-stacked.jpg" width="80%"/>
</picture>
</div>

[Project page](https://research.zenseact.com/publications/splatad/)

</div>


# About
This is the official repository for the CVPR 2025 paper [_SplatAD: Real-Time Lidar and Camera Rendering with 3D Gaussian Splatting for Autonomous Driving_](https://arxiv.org/abs/2411.16816). The code in this repository builds upon the open-source library [gsplat](https://github.com/nerfstudio-project/gsplat), with modifications and extensions designed for autonomous driving data. 

While the code contians all components needed to efficiently render camera and lidar data, the SplatAD-model itself, including dataloading, decoders, etc., will be released through [neurad-studio](https://github.com/georghess/neurad-studio).

**We welcome all contributions!**

# Key Features
- Efficient lidar rendering
    - Projection to spherical coordinates
    - Depth and feature rasterization for a non-linear grid of points
- Rolling shutter compensation for camera and lidar


# Installation
Our code introduce no additional dependencies. We thus refer to the original documentation from gsplat for both [installation](https://github.com/nerfstudio-project/gsplat#installation) and [development setup](https://github.com/nerfstudio-project/gsplat/blob/main/docs/DEV.md).

# Usage
See [`rasterization`](gsplat/rendering.py#L22) and [`lidar_rasterization`](gsplat/rendering.py#L453) for entry points to camera and lidar rasterization.
Additionally, we provide example notebooks under [examples](examples) that demonstrate lidar rendering and rolling shutter compensation.
For further examples, check out the [test files](tests).


# Built On
- [gsplat](https://github.com/nerfstudio-project/gsplat) - Collaboration friendly library for CUDA accelerated rasterization of Gaussians with python bindings
- [3dgs-deblur](https://github.com/SpectacularAI/3dgs-deblur) - Inspiration for the rolling shutter compensation

# Citation

You can find our paper on [arXiv](https://arxiv.org/abs/2411.16816).

If you use this code or find our paper useful, please consider citing:

```bibtex
@article{hess2024splatad,
  title={SplatAD: Real-Time Lidar and Camera Rendering with 3D Gaussian Splatting for Autonomous Driving},
  author={Hess, Georg and Lindstr{\"o}m, Carl and Fatemi, Maryam and Petersson, Christoffer and Svensson, Lennart},
  journal={arXiv preprint arXiv:2411.16816},
  year={2024}
}
```

# Contributors

<a href="https://github.com/georghess">
    <img src="https://github.com/georghess.png" width="60px;" style="border-radius: 50%;"/>
</a>
<a href="https://github.com/carlinds">
    <img src="https://github.com/carlinds.png" width="60px;" style="border-radius: 50%;"/>
</a>

\+ [gsplat contributors](https://github.com/nerfstudio-project/gsplat/graphs/contributors)