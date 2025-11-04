# NeRF_PyTorch

Unofficial PyTorch implementation of NeRF: Neural Radiance Fields. ([official Tensorflow implementation repo](https://github.com/bmild/nerf))

## Setup Environment 
```
conda create -n nerf python=3.9
conda activate nerf
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https:/download.pytorch.org/whl/cu113
cd Path/to/starter_code_repo
pip install -r requirements.txt
```

## Rendered Results using Coarse Model

<p align="center">
  <img src="media/coarse_rendered_results.png" alt="coarse_rendered_results" width="720"/>
</p>

## Rendered Results using Coarse + Fine Model
<p align="center">
  <img src="media/fine_rendered_results.png" alt="fine_rendered_results" width="720"/>
</p>