# NeRF_PyTorch

Unofficial PyTorch implementation of NeRF: Neural Radiance Fields. ([official Tensorflow implementation](https://github.com/bmild/nerf))

# Usage
## Environment 
```
conda create -n nerf python=3.9
conda activate nerf
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https:/download.pytorch.org/whl/cu113
cd Path/to/starter_code_repo
pip install -r requirements.txt
```

## Training
Please download the [data](https://drive.google.com/file/d/1OsiBs2udl32-1CqTXCitmov4NQCYdA9g/view?usp=drive_link) to the ```data/``` folder in your repository.

To train the model using only the **coarse** model, run:
```
python run_nerf.py --config configs/lego_coarse.txt
```

To train the model using both **coarse** and **fine** model, run:
```
python run_nerf.py --config configs/lego_fine.txt
```

# Results
## Rendered Results using Coarse Model

<p align="center">
  <img src="media/coarse_rendered_results.png" alt="coarse_rendered_results" width="720"/>
</p>

## Rendered Results using Coarse + Fine Model
<p align="center">
  <img src="media/fine_rendered_results.png" alt="fine_rendered_results" width="720"/>
</p>